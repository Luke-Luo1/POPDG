from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F

from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like


class TemporalEmbedding(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding):
        super(TemporalEmbedding, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = x.transpose(1, 2)  # shape -> [b,h,t]
        x = self.conv(x)
        x = x.transpose(1, 2)  # revert to original shape
        return x

class MusicConvModule(nn.Module):
    def __init__(self,d_model,use_temporal_embedding=True):
        super(MusicConvModule, self).__init__()
        self.temporal_embedding_music = TemporalEmbedding(in_channels=d_model, out_channels=d_model, kernel_size=5, stride=1, padding=2)
        
        self.temporal_music_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
       
        self.music_parameter = nn.Parameter(torch.randn(1, d_model))
        self.use_temporal_embedding = use_temporal_embedding

    def forward(self, x_m, t, cond_drop_prob):
        batch_size, device = x_m.shape[0], x_m.device
        if self.use_temporal_embedding:
            music_temporal_mean = self.temporal_embedding_music(x_m).mean(dim=-2)
        else:
            music_temporal_mean = x_m.mean(dim=-2)
        mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        mask = rearrange(mask, "b -> b 1")
        music_temporal_mean = self.temporal_music_projection(music_temporal_mean)
        null_music_seq = self.music_parameter.to(t.dtype)
        music_temporal_hidden = torch.where(mask, music_temporal_mean, null_music_seq)

        t = t + music_temporal_hidden
        return t
    

class AlignmentModule(nn.Module):
    def __init__(self, d_model, use_temporal_embedding=True):
        super(AlignmentModule, self).__init__()
        self.temporal_embedding_dance = TemporalEmbedding(in_channels=d_model, out_channels=d_model, kernel_size=5, stride=1, padding=2)
        self.embed_channels = d_model
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(d_model, d_model * 2)
        )
        self.use_temporal_embedding = use_temporal_embedding

    def forward(self, x_d, t):
        if self.use_temporal_embedding:
            dance_temporal_mean = self.temporal_embedding_dance(x_d).mean(dim=-2)
        else:
            dance_temporal_mean = x_d.mean(dim=-2)
        t = t + dance_temporal_mean
        t = self.block(t)
        t = rearrange(t, "b c -> b 1 c")
        weight_bias = t.chunk(2, dim=-1)        
        weight, bias = weight_bias
        return (weight + 1) * x_d + bias
    

class DiffTimeEmb(nn.Module):
    def __init__(self,d_model):
        super(DiffTimeEmb, self).__init__()
        self.mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),  
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
        )
        self.time_music = nn.Linear(d_model * 4, d_model)
        self.time_tokens = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),  
            Rearrange("b (r d) -> b r d", r=2),
        )        
    def forward(self, times):
        t_hidden = self.mlp(times)
        t = self.time_music(t_hidden)
        t_tokens = self.time_tokens(t_hidden)        

        return t,t_tokens 

class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t, cond_drop_prob):
        for layer in self.stack:
            x = layer(x, cond, t, cond_drop_prob)
        return x

class MusicEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        rotary=None,
    ) -> None:
        super().__init__()
        self.spatio_attn = nn.MultiheadAttention(
            150, 5, dropout=dropout, batch_first=batch_first
        )
        self.temporal_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        x = x + self.MF_Attn(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self.MT_Attn(self.norm2(x), src_mask, src_key_padding_mask)
        x = x + self.ff_block(self.norm3(x))
        return x

    # self-spacial-attention block
    def MF_Attn(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        qk = qk.permute(0,2,1)
        x = x.permute(0,2,1)
        x_transpose = self.spatio_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x_transpose.permute(0, 2, 1)  
        return self.dropout1(x)
    
    # self-temporal_attention block
    def MT_Attn(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.temporal_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DS_Attention(nn.Module):
    def __init__(self, d_model, d_depth, num_heads):
        super(DS_Attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_depth

        self.qk_proj_weight = nn.Parameter(torch.empty(d_model,2*d_depth*num_heads))
        self.v_proj_weight = nn.Parameter(torch.empty(d_model,d_depth*num_heads))
        
        nn.init.xavier_uniform_(self.qk_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        
        self.linear = nn.Linear(d_depth*num_heads, d_model)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        nframes = query.size(1)
        
        combined_qk = torch.einsum('bnf,fx->bnx',[query, self.qk_proj_weight])
        query, key = torch.split(combined_qk, self.depth * self.num_heads, dim =2 )
        value = torch.einsum('bnf,fx->bnx',[value, self.v_proj_weight])
        
        query, key, value = map(lambda x: self.split_heads(x, batch_size), [query, key, value])

        attn_output = self.scaled_dot_product_attention(query[:,:,:,3:147].view(batch_size,-1,nframes,24,6),
                                                        key[:,:,:,3:147].view(batch_size,-1,nframes,24,6), 
                                                        value[:,:,:,3:147].view(batch_size,-1,nframes,24,6))
        
        value[:,:,:,3:147] = attn_output.view(batch_size,-1,nframes,144)

        value = value.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.depth*self.num_heads)
        return self.linear(value)

    def scaled_dot_product_attention(self, q, k, v):
        d_k = q.size(-1)
        attn_probs = torch.matmul(q, k.transpose(-2, -1))/ math.sqrt(d_k)
        attn_probs = nn.functional.softmax(attn_probs, dim=-1)
        
        attn_probs = self.apply_custom_weighting(attn_probs)
                
        output = torch.matmul(attn_probs, v)
        return output
    
    def apply_custom_weighting(self, attn_probs):
        attn_probs_clone = attn_probs.clone()

        attn_probs_clone[:,:,:,0,6] += attn_probs_clone[:,:,:,0,3] 
        attn_probs_clone[:,:,:,0,6] /= 2 
        attn_probs_clone[:,:,:,6,0] += attn_probs_clone[:,:,:,3,0] 
        attn_probs_clone[:,:,:,6,0] /= 2
        
        attn_probs_clone[:,:,:,0,9] += attn_probs_clone[:,:,:,0,6] 
        attn_probs_clone[:,:,:,0,9] /= 2 
        attn_probs_clone[:,:,:,9,0] += attn_probs_clone[:,:,:,6,0] 
        attn_probs_clone[:,:,:,9,0] /= 2 
        
        indices = [12, 13, 14]
        for idx in indices:
            attn_probs_clone[:,:,:,0,idx] += attn_probs_clone[:,:,:,0,9] 
            attn_probs_clone[:,:,:,0,idx] /= 2
            attn_probs_clone[:,:,:,idx,0] += attn_probs_clone[:,:,:,9,0]
            attn_probs_clone[:,:,:,idx,0] /= 2
        
        map_indices = {13:16, 14:17, 12:15}
        for source, target in map_indices.items():
            attn_probs_clone[:,:,:,0,target] += attn_probs_clone[:,:,:,0,source]
            attn_probs_clone[:,:,:,0,target] /= 2
            attn_probs_clone[:,:,:,target,0] += attn_probs_clone[:,:,:,source,0]
            attn_probs_clone[:,:,:,target,0] /= 2
        
        return attn_probs_clone     
        


class DanceDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.spatio_attn = DS_Attention(d_model,156,4)
        self.temporal_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = activation

        # If there are requirements for GPU memory usage and training speed, it can be set to False.
        self.align1 = AlignmentModule(d_model,False)
        self.align2 = AlignmentModule(d_model,False)
        self.align3 = AlignmentModule(d_model,False)

        self.rotary = rotary
        self.use_rotary = rotary is not None


    # x, cond, t
    def forward(
        self,
        tgt,
        memory,
        t,
        cond_drop_prob,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        
        x_1 = self.DS_Attn(self.norm1(x))
        x = x + x_1
        
        x_2 = self.DT_Attn(self.norm2(x), tgt_mask, tgt_key_padding_mask)
        x = x + self.align1(x_2,t)
        
        x_3 = self.Cross_Attn(self.norm3(x), memory, memory_mask, memory_key_padding_mask)
        x = x + self.align2(x_3,t)
        
        x_4 = self.ff_block(self.norm4(x))
        x = x + self.align3(x_4,t)
        return x

    # self-spacial-attention block
    def DS_Attn(self, x):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        
        x = self.spatio_attn(qk,qk,x)
        
        return self.dropout1(x)
      
    # self-temporal-attention block
    def DT_Attn(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.temporal_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # multihead attention block
    def Cross_Attn(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout3(x)

    # feed forward block
    def ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout4(x)

class Model(nn.Module):
    def __init__(
        self,
        nfeats: int,
        nframes: int = 150,  # 5 seconds, 30 fps
        latent_dim: int = 256,
        ff_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        music_feature_dim: int = 4800,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        self.TimeEmbed = DiffTimeEmb(latent_dim)
        # null embeddings for guidance dropout
        self.null_music_embed = nn.Parameter(torch.randn(1, nframes, latent_dim))
        self.norm_music = nn.LayerNorm(latent_dim)
        # If there are requirements for GPU memory usage and training speed, it can be set to False.
        self.music_temporalcoding = MusicConvModule(latent_dim,False)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.music_encoder = nn.Sequential()
        for _ in range(2):
            self.music_encoder.append(
                MusicEncoder(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        # conditional projection
        self.cond_projection = nn.Linear(music_feature_dim, latent_dim)
 
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                DanceDecoder(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.DecoderSequence = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(latent_dim, output_feats)
        self.variance_layer = nn.Linear(latent_dim,output_feats)

    def guided_forward(self, x, music, times, guidance_weight):
        unc, unc_variance = self.forward(x, music, times, music_drop_prob=1)
        conditioned, conditioned_variance = self.forward(x, music, times, music_drop_prob=0)

        output = unc + (conditioned - unc) * guidance_weight
        variance = unc_variance + (conditioned_variance - unc_variance) * guidance_weight
        return output, variance

    def forward(
        self, x: Tensor, music: Tensor, times: Tensor, music_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device
        
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x)

        # create music conditional embedding with conditional dropout
        mask = prob_mask_like((batch_size,), 1 - music_drop_prob, device=device)
        mask_embed = rearrange(mask, "b -> b 1 1")

        music = music.to(torch.float32)
        music_tokens = self.cond_projection(music)
        music_tokens = self.abs_pos_encoding(music_tokens)
        music_tokens = self.music_encoder(music_tokens)

        null_cond_embed = self.null_music_embed.to(music_tokens.dtype)
        music_tokens = torch.where(mask_embed, music_tokens, null_cond_embed)

        t, t_tokens = self.TimeEmbed(times)

        t = self.music_temporalcoding(music_tokens,t,music_drop_prob)

        # cross-attention conditioning
        c = torch.cat((music_tokens, t_tokens), dim=-2)
        music_tokens = self.norm_music(c)

        output_1 = self.DecoderSequence(x, music_tokens, t, music_drop_prob)

        output = self.final_layer(output_1)
        variance = torch.clamp(self.variance_layer(output_1), min=-1.0, max=1.0)
        return output, variance
