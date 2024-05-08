import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.load_popdanceset import PopDanceSet
from dataset.preprocess import increment_path
from model.adan import Adan
from model.iDDPM import GaussianDiffusion
from model.model import Model
from vis import SMPLSkeleton

class POPDG:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        EMA=True,
        learning_rate=2e-4,
        weight_decay=0.02,
    ):
        self.setup_accelerator()
        self.initialize_models(feature_type, checkpoint_path, learning_rate, weight_decay, EMA)
    
    def setup_accelerator(self):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.state = AcceleratorState()
    
    def initialize_models(self, feature_type, checkpoint_path, learning_rate, weight_decay, EMA):

        use_baseline_feats = feature_type == "baseline"
        pos_dim, rot_dim = 3, 24 * 6
        self.repr_dim = pos_dim + rot_dim + 9
        feature_dim = 35 if use_baseline_feats else 4800

        horizon_seconds, FPS = 5, 30
        self.horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]

        model = Model(
            nfeats=self.repr_dim,
            nframes=self.horizon,
            latent_dim=512,
            ff_dim=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            music_feature_dim=feature_dim,
            activation=F.gelu,
        )

        smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = GaussianDiffusion(
            model,
            self.horizon,
            self.repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            music_drop_prob=0.25,
            guidance_weight=2,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                self.maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    self.state.num_processes,
                )
            )

    def maybe_wrap(self, x, num):
        return x if num == 1 else {f"module.{key}": value for key, value in x.items()}

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)
    
    def train_loop(self, opt):
        
        train_data_loader, test_data_loader = self.setup_data_loaders(opt)
        self.prepare_training_environment(opt)
        self.run_training_epochs(train_data_loader, test_data_loader, opt)
    
    def setup_data_loaders(self, opt):
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
                
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = PopDanceSet(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )
            test_dataset = PopDanceSet(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))
    

        # set normalizer
        self.normalizer = test_dataset.normalizer

        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            # num_workers=10,
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        return self.accelerator.prepare(train_data_loader), test_data_loader
    
    def prepare_training_environment(self, opt):
        # boot up multi-gpu training. test dataloader is only on main process
        self.load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            # To resume training after an interruption, use the next line of code.
            # wandb.init(project=opt.wandb_pj_name, name=opt.exp_name, resume='must', id='')
            save_dir = Path(save_dir)
            self.wdir = save_dir / "weights"
            self.wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
    
    def run_training_epochs(self, train_data_loader, test_data_loader, opt):
        # start_epoch = 1301
        # for epoch in range(start_epoch, opt.epochs + 1):
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_bodyloss = 0
            avg_vlbloss = 0
            # train
            self.train()
            for step, (x, cond, filename, wavnames) in enumerate(
                self.load_loop(train_data_loader)
            ):
                total_loss, (loss, v_loss, fk_loss, body_loss, vlb_loss) = self.diffusion(
                    x, cond, t_override=None
                )
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_bodyloss += body_loss.detach().cpu().numpy()
                    avg_vlbloss += vlb_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
            # Save model
            if (epoch % opt.save_interval) == 0:
            # if epoch == 1 or (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_bodyloss /= len(train_data_loader)
                    avg_vlbloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Body Loss": avg_bodyloss,
                        "Vlb Loss": avg_vlbloss,
                    }
                    wandb.log(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(self.wdir, f"train-{epoch}.pt"))
                    # generate four samples
                    render_count = 4
                    shape = (render_count, self.horizon, self.repr_dim)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x, cond, filename, wavnames) = next(iter(test_data_loader))
                    cond = cond.to(self.accelerator.device)
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=wavnames[:render_count],
                        sound=True,
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process:
            wandb.run.finish()


    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render
        )
