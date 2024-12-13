
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m --loss_type mse --min_rating 4 --min_uc 20 --min_sc 1 --cuda 4
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou 
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty --loss_type mse --min_rating 1 --min_uc 12 --min_sc 10 --cuda 4

python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m 
python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou 
python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty 

python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m 
python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou 
python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty 

python -u Globaldiff_atten_adapter.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m



python -u GlobalDiff_srgnn.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m --pretrain_epoch 40 --min_rating 4 --min_uc 20 --min_sc 1 --cuda 2 --loss_type mse
python -u GlobalDiff_srgnn.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou --pretrain_epoch 40 --min_rating 1 --min_uc 20 --min_sc 1 --cuda 2
python -u GlobalDiff_srgnn.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty --pretrain_epoch 8 --min_rating 1 --min_uc 12 --min_sc 10 --cuda 1