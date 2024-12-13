# GlobalDiff
GlobalDiff is a diffusion-based model designed to enhance sequential recommendation systems. It supports various backbone models like CBIT, Bert4Rec, and SASRec, and can be easily adapted to different datasets. This repository includes implementations for ML-1M, KuaiRec, and Beauty datasets.

## Reproduce the results

### ML-1M

```
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m 
python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m

python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m
#SRGNN
python -u GlobalDiff_srgnn.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m --pretrain_epoch 40 --min_rating 4 --min_uc 20 --min_sc 1 --cuda 0 --loss_type mse
```

### KuaiRec Data

```
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou 
python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou 
python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou 
```

### Beauty

```
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty 
python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty 
python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty 
```

## BackBone

GlobalDiff supports direct transfer from CBiT, Bert4Rec, and SASRec. In fact, we recommend using the original model's code for training. The relevant code links are as follows:
https://github.com/hw-du/CBiT/tree/master;
https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch;
https://github.com/pmixer/SASRec.pytorch


