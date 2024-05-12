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
# Citation
# Acknowledgements
