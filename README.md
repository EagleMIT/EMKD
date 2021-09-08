# EMKD
The code repository of IEEE TMI paper [Efficient Medical Image Segmentation Based on Knowledge Distillation](https://ieeexplore.ieee.org/abstract/document/9491090)

# Structure of this repository
This repository is organized as:

- [datasets](/datasets/) contains the dataloader for different datasets
- [networks](/networks/) contains a model zoo for network models
- [scripts](/networks/) coontains scripts for preparing data
- [utils](/networks/) contains api for training and processing data
- [train.py](/train.py) train a single model
- [train_kd.py](/train_kd.py) train with KD

# Usage Guide

## Requirements

 All the codes are tested in the following environment:

- pytorch 1.8.0
- pytorch-lightning >= 1.3.7
- OpenCV
- nibabel

## Dataset Preparation

### KiTS
Download data [here](https://github.com/neheller/kits19)

Please follow the instructions and the data/ directory should then be structured as follows
```
data
├── case_00000
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
├── case_00001
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
...
├── case_00209
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
└── kits.json
```
Cut 3D data into slices using ```scripts/SliceMaker.py``` 

```
python scripts/SliceMaker.py --inpath /data/kits19/data --outpath /data/kits/train --dataset kits --task tumor
```

### LiTS
Similar to KiTS but you may make some adjustments in running ```scripts/SliceMaker.py``` 
```
lits
├── Training_Batch
└── Test-Data
```
```
python scripts/SliceMaker.py --inpath /data/lits/Training-Batch --outpath /data/lits/train --dataset lits --task tumor
```

## Running
### Training Teacher Model
Before knowledge distillation, a well-trained teacher model is required. ```/train.py``` is used to trained a single model without KD(either a teacher model or a student model). 

[RAUNet](https://github.com/nizhenliang/RAUNet) is recommended to be the teacher model.

```
python train.py --model raunet --checkpoint_path /data/checkpoints
```

After training, the checkpoints will be stored in ```/data/checkpoints``` as assigned.

If you want to try different models, use ```--model``` with following choices
```
'deeplabv3+', 'enet', 'erfnet', 'espnet', 'mobilenetv2', 'unet++', 'raunet', 'resnet18', 'unet', 'pspnet'
```
### Training With Knowledge Distillation 
For example, use enet as student model

```
python train_kd.py --tckpt /data/checkpoints/name_of_teacher_checkpoint.ckpt --smodel enet
```

```--tckpt``` refers to the path of teacher model checkpoint. And you can change student model by revising ```--smodel```
