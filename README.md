# GlobalDiff
GlobalDiff is a diffusion-based model designed to enhance sequential recommendation systems. It supports various backbone models like CBIT, Bert4Rec, SASRec and SRGNN, and can be easily adapted to different datasets. This repository includes implementations for ML-1M, KuaiRec, and Beauty datasets.

## Reproduce the results

### ML-1M

```
#CBiT
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m
#Bert4Rec
python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m
#SasRec
python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m
#SRGNN
python -u GlobalDiff_srgnn.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m --pretrain_epoch 40
```

### KuaiRec Data

```
#CBiT
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou
#Bert4Rec
python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou
#SasRec
python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou
#SRGNN
python -u GlobalDiff_srgnn.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou --pretrain_epoch 40
```

### Beauty

```
#CBiT
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty
#Bert4Rec
python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty
#SasRec
python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty
#SRGNN
python -u GlobalDiff_srgnn.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty --pretrain_epoch 40
```

## BackBones

GlobalDiff supports direct transfer from CBiT, Bert4Rec, SASRec and SRGNN. In fact, we recommend using the original model's code for training. The relevant code links are as follows:
https://github.com/hw-du/CBiT/tree/master;
https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch;
https://github.com/pmixer/SASRec.pytorch
https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/srgnn.py

