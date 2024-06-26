# POPDG
**Official PyTorch implementation of the paper and PopDanceSet**
![LOGO_final](https://github.com/Luke-Luo1/POPDG/assets/100562982/937c246f-2872-44e7-a8d9-40dd8e6a529f)
[Paper(arXiv version)](https://arxiv.org/abs/2405.03178) | [Paper(CVPR version)](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_POPDG_Popular_3D_Dance_Generation_with_PopDanceSet_CVPR_2024_paper.pdf) | [Project Page](https://luke-luo1.github.io/POPDG/)

**Abstract:** Generating dances that are both lifelike and well-aligned with music continues to be a challenging task in the cross-modal domain. This paper introduces PopDanceSet the first dataset tailored to the preferences of young audiences enabling the generation of aesthetically oriented dances. And it surpasses the AIST++ dataset in music genre diversity and the intricacy and depth of dance movements. Moreover the proposed POPDG model within the iDDPM framework enhances dance diversity and through the Space Augmentation Algorithm strengthens spatial physical connections between human body joints ensuring that increased diversity does not compromise generation quality. A streamlined Alignment Module is also designed to improve the temporal alignment between dance and music. Extensive experiments show that POPDG achieves SOTA results on two datasets. Furthermore the paper also expands on current evaluation metrics. The dataset and code are available at https://github.com/Luke-Luo1/POPDG.
# Environment
```
pytorch 1.12.1+cu116
python 64-bit 3.8.8
at least 16 GB of GPU(Nvidia) memory
```
The successful execution of the entire project depends on the following libraries:
- [jukemirlib](https://github.com/rodrigo-castellon/jukemirlib)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d)
- [accelerate](https://huggingface.co/docs/accelerate/v0.16.0/en/index)
  Run accelerate config after installation.
- [wine](https://www.winehq.org) 
# PopDanceSet
PopDanceSet has been updated! For new dataset information, experimental results, and download methods, please see [docs/PopDanceSet_Description.md](https://github.com/Luke-Luo1/POPDG/blob/main/docs/PopDanceSet_Description.md)

Below is a visualization rendering demo of a small selection of dance genres from the PopDanceSet:

<img src="https://github.com/Luke-Luo1/POPDG/blob/main/docs/demo_1.gif" width="270"> <img src="https://github.com/Luke-Luo1/POPDG/blob/main/docs/demo_3.gif" width="270"> <img src="https://github.com/Luke-Luo1/POPDG/blob/main/docs/demo_4.gif" width="270">
<img src="https://github.com/Luke-Luo1/POPDG/blob/main/docs/demo_5.gif" width="270"> <img src="https://github.com/Luke-Luo1/POPDG/blob/main/docs/demo_6.gif" width="270"> <img src="https://github.com/Luke-Luo1/POPDG/blob/main/docs/demo_7.gif" width="270"> 

# Training
Once the PopDanceSet data has been processed, we can commence with the training:
```
accelerate launch train.py --batch_size <batch_size> --epochs 2000 --exp_name <experiment name> --feature_type jukebox
```
Other configuration parameters are also available in the args.py file, which can be freely adjusted according to training requirements.
- Note: POPDG requires a significant amount of VRAM. We trained in a dual A800 GPU environment. If there is insufficient VRAM available, it is crucial to adjust the batch size accordingly.
# Evaluation
To test our pre-trained model, please download the weights from [Google Drive](https://drive.google.com/file/d/13ZE-x-oKp8SBM7crfANrcZYkW26I3XBZ/view?usp=sharing).

Step 1. Testing dance generation models：
```
python test.py --checkpoint <path to checkpoint> --save_motions<optional>
```
Just like training, you can set other testing arguments as needed.

Step 2. Extract the kinematic and manual features of all PopDanceSet motions(only do it by once)：
```
python eval/extract_features.py
```
- Note: Remember to extract train and test motions, not only training data. 

Step 3. Dance quality evaluations:
Below are the commands to measure the test set metrics PFC, PBC, DIV, and BAS, respectively.
```
python eval/eval_pfc.py --motion_path <path to save_motions>
python eval/eval_pbc.py --motion_path <path to save_motions>
python eval/calculate_scores.py
python eval/calculate_beat_scores.py
```
# Choreographic for wild music
To choreograph any single or multiple pieces of music, simply place the music into the folder you created. The subsequent steps are exactly the same as those in the testing process：
```
python test.py --checkpoint <path to checkpoint> --music_dir <path to test music data> --render_dir <optional, path to rendering file> --save_motions<optional>
```

# Blender
The rendering process in POPDG is essentially the same as with previous models. First, download a model which you prefer from [Mixamo](https://www.mixamo.com/#/) (an example rig ybot.fbx is provided in the SMPL-to-FBX folder). Then, by running the following code, convert the saved dance motions (pkl files) into FBX files, which could then be imported into Blender for rendering.
```
python SMPL-to-FBX/Convert.py --input_dir <path to save_motions> --output_dir <path to outputs>
```
# Citation
```
@InProceedings{Luo_2024_CVPR,
    author    = {Luo, Zhenye and Ren, Min and Hu, Xuecai and Huang, Yongzhen and Yao, Li},
    title     = {POPDG: Popular 3D Dance Generation with PopDanceSet},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {26984-26993}
}
```
# Acknowledgements
We would like to express our deep gratitude to [Li](https://github.com/google-research/mint) for proposing the AIST++ dataset, which served as a template for the construction of PopDanceSet, and we are also immensely thankful to Tseng for  [EDGE ](https://github.com/Stanford-TML/EDGE), which established the foundational framework for POPDG.

