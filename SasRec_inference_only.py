# @encodeing = utf-8
# @Author : lmx
# @Date : 2024/5/20
# @description :  test for sasrec
import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import logging
import time as Time
from utility import calculate_hit
from collections import Counter
from Modules_ori import *
import warnings
from tqdm import tqdm
import time
import sys
from dataloaders import dataloader_factory
from pretrain_models.SasRec.SasRec import SASRec

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=3000,
                        help='Number of max epochs.')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='timesteps for diffusion')
    parser.add_argument('--l2_decay', type=float, default=0,
                        help='l2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=6,
                        help='cuda device.')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='dropout ')
    parser.add_argument('--p', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--report_epoch', type=bool, default=True,
                        help='report frequency')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='type of optimizer.')
    #######################dataset#############################
    parser.add_argument('--dataset_code', type=str, default='ml-1m',
                    help='which dataset , choose from[ml-1m,beauty,kuaishou]')
    parser.add_argument('--min_rating', type=int, default=4,
                    help='user min rating')
    parser.add_argument('--min_uc', type=int, default=20,
                    help='N ratings per user for validation and test, should be at least max_len+2')
    parser.add_argument('--min_sc', type=int, default=1,
                    help='N ratings per item for validation and test')
    parser.add_argument('--split', type=str, default='leave_one_out',
                    help='dataset split mode')
    #######################dataloder###########################
    parser.add_argument('--dataloader_code', type=str, default='SasRec',
                    help='which dataloder , choose from[SasRec,Diff]')
    parser.add_argument('--dataloader_random_seed', type=int, default=2024,
                    help=' dataloder random seed ')
    parser.add_argument('--max_len', type=int, default=10,
                    help=' enable max seq len ')
    parser.add_argument('--slide_window_step', type=int, default=1,
                help=' slide window step ')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--mask_prob', type=int, default=0.15,
                    help='dataloder mask_prob')

    #######################train###########################
    parser.add_argument('--num_items', type=int, default=4,
                        help='num_items')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')

    ##########################SasRec#################################
    parser.add_argument('--maxlen', default=10, type=int)
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    # drop_rate 统一设置
    # parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda:0', type=str)

    return parser.parse_args()

args = parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(args.random_seed)




class Tenc(nn.Module):
    def __init__(self, args,device,InitModel):
        super(Tenc, self).__init__()
        self.state_size = args.max_len
        self.item_num = int(args.num_items)
        self.dropout_rate = args.dropout_rate
        self.device = device
        #################################SASRec############################################
        self.SASRec = InitModel

    
    def forward(self, states):
        """
        对整个序列进行pretrain model的前向
            param:
                states : seq序列 形状为(B,S)
            return :
                inputs_emb : 对seq有mask的序列做编码的结果 ，形状为(B,S,D)
        """
        logits,inputs_emb = self.SASRec.predictAll(states)
        #注意！！token id从1开始算，所以这里的输出维度是item_num+1,最大的token id是item_num
        # (B*S) x (V+1)
        logits = logits.view(-1, logits.size(-1))      
        return inputs_emb,logits


    def predict(self, states):
        """
        模型预测
            Args:
                states: 输入的序列，形状为(B,S),类型为torch.LongTensor
            returns:
                prediction: 预测结果，形状为(B,V)
        """
        with torch.no_grad():
            h,logits=self.forward(states)
            # B x V
            scores4En = logits.view(states.size(0),states.size(1),-1)[:, -1, :] 
        return scores4En



def evaluate(model, data_loder, device,epoch_index,is_save):

    evaluated=0
    total_clicks=1.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks=[0,0,0,0]
    ndcg_clicks=[0,0,0,0]
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]
    hit_purchase4En=[0,0,0,0]
    ndcg_purchase4En=[0,0,0,0]
    tqdm_dataloader4eval = tqdm(data_loder)
    # 确保dropout关闭
    model.eval()

    for batch_idx, batch in enumerate(tqdm_dataloader4eval):
        batch = [x.to(device) for x in batch]
        seqs, target ,mask_indice= batch[0], batch[1],batch[2]
        batch_size=seqs.size(0)
        predictionEn= model.predict(seqs)
        ###########################Encoder 的指标######################################
        _, topKEn = predictionEn.topk(100, dim=1, largest=True, sorted=True)
        topKEn = topKEn.cpu().detach().numpy()
        sorted_listEn=np.flip(topKEn,axis=1)
        sorted_listEn = sorted_listEn
        calculate_hit(sorted_listEn,topk,target.tolist(),hit_purchase4En,ndcg_purchase4En)
        total_purchase+=batch_size
 

    print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[0]), 'NDCG@'+str(topk[0]), 'HR@'+str(topk[1]), 'NDCG@'+str(topk[1]), 'HR@'+str(topk[2]), 'NDCG@'+str(topk[2])))
    
    print('#############################Encoder#########################################')
    hr_list = []
    ndcg_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase4En[i]/total_purchase
        ng_purchase=ndcg_purchase4En[i]/total_purchase

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
    
        if i == 0:
            hr_10En = hr_purchase

    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    if is_save:
        save_metrics("./results/encoder.csv",epoch_index,hr_list,ndcg_list,topk)


   

    return hr_10En


def save_metrics(PATH,epcoh_number,hr_list,ndcg_list,topk):
    # 检查路径是否存在
    if not os.path.exists(PATH):
        # 如果路径不存在，创建一个新的 DataFrame
        ## 创建一个空的DataFrame
        df = pd.DataFrame()
        df['epoch']=[epcoh_number]
        for i in range(len(topk)):
            df['HR@'+str(topk[i])]=[hr_list[i]]
            df['NDCG@'+str(topk[i])]=[ndcg_list[i]]
        df.to_csv(PATH, index=False)
        print(f"Created and saved a new CSV file at {PATH}")
    else:
        df = pd.read_csv(PATH)
        curr_index=df.index.max()
        # 使用loc为指定索引位置添加新值
        df.loc[curr_index + 1, 'epoch'] = epcoh_number
        for i in range(len(topk)):
            df.loc[curr_index + 1, 'HR@'+str(topk[i])] = hr_list[i]
            df.loc[curr_index + 1, 'NDCG@'+str(topk[i])] =ndcg_list[i]
        df.to_csv(PATH, index=False)

if __name__ == '__main__':


    ##########################日志#######################################
    #日志文件名按照程序运行时间设置
    log_file_name =  './Log/log-'+args.dataset_code +'-SasRec-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    log_print = open(log_file_name, 'w')
    sys.stdout = log_print
    ##########################cuda##########################################
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    torch.backends.cudnn.enabled = False
    ###########################dataset and loder############################################
    train_loader, val_loader, test_loader, item_num = dataloader_factory(args)
    args.num_items=item_num
    total_loss=0
    num_total=len(val_loader)

    topk=[10, 20, 50]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ##########################加载训练好的SASRec模型##################################################################
    # 因为SASRec模型定义需要itemnum,usernum，但是usernum没有用到，所以这里可以传0进去
    SAS_model = SASRec(0,item_num,args)
    # best_model = torch.load('./pretrain_models/SasRec/ml1m.pth').get('model_state_dict')
    SAS_model.load_state_dict(torch.load('./pretrain_models/SasRec/ml1m.pth', map_location=device))
    # SAS_model.load_state_dict(best_model)
    SAS_model.to(device)
    model = Tenc(args,device,SAS_model)
    model.to(device)

    eval_start = Time.time()
    print('-------------------------- VAL PHRASE --------------------------')
    Valhr_10En = evaluate(model, val_loader, device,0,is_save=False)
    print('-------------------------- TEST PHRASE -------------------------')
    hr_10En = evaluate(model, test_loader,device,0,is_save=True)
    print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-eval_start)))
    print('----------------------------------------------------------------')


