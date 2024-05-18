# POPDG
# Environment
```
pytorch 1.12.1+cu116
python 64-bit 3.8.8
```
The successful execution of the entire project depends on the following libraries:
- [jukemirlib](https://github.com/rodrigo-castellon/jukemirlib)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d)
- [accelerate](https://huggingface.co/docs/accelerate/v0.16.0/en/index)
- [wine](https://www.winehq.org) 
# PopDanceSet
PopDanceSet has been updated! For new dataset information, experimental results, and download methods, please see [docs/PopDanceSet_Description.md](https://github.com/Luke-Luo1/POPDG/blob/main/docs/PopDanceSet_Description.md)
# Training
Once the PopDanceSet data has been processed, we can commence with the training:
```
accelerate launch train.py --batch_size <batch_size> --epochs 2000 --exp_name <experiment name> --feature_type jukebox
```
Other configuration parameters are also available in the args.py file, which can be freely adjusted according to training requirements.
- Note: POPDG requires a significant amount of VRAM. We trained in a dual A800 GPU environment. If there is insufficient VRAM available, it is crucial to adjust the batch size accordingly.
# Evaluation
To test our pre-trained model, please download the weights from [here](https://drive.google.com/file/d/13ZE-x-oKp8SBM7crfANrcZYkW26I3XBZ/view?usp=sharing)(Google Drive).

Step 1. Testing dance generation models：
```
python test.py --checkpoint <path to checkpoint> --music_dir <path to test music data> --render_dir <optional, path to rendering file> --save_motions
```
Just like training, you can set other testing parameters as needed.

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
# Blender
Here, we fully adhere to the 3D rendering method of [EDGE](https://github.com/Stanford-TML/EDGE):
```
python SMPL-to-FBX/Convert.py --input_dir <path to save_motions> --output_dir <path to outputs>
```
# Citation
```
@article{luo2024popdg,
  title={POPDG: Popular 3D Dance Generation with PopDanceSet},
  author={Luo, Zhenye and Ren, Min and Hu, Xuecai and Huang, Yongzhen and Yao, Li},
  journal={arXiv preprint arXiv:2405.03178},
  year={2024}
}
```
# Acknowledgements
We would like to express our deep gratitude to [Li](https://github.com/google-research/mint) for proposing the AIST++ dataset, which served as a template for the construction of PopDanceSet, and we are also immensely thankful to Tseng for  [EDGE ](https://github.com/Stanford-TML/EDGE), which established the foundational framework for POPDG.

