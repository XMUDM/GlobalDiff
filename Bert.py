# @encodeing = utf-8
# @Author : lmx
# @Date : 2024/5/20
# @description : train and test for bert
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
from pretrain_models.Bert.bert_modules.bert import BERT
import time
import sys
from dataloaders import dataloader_factory
import torch.optim as optim
# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=250,
                        help='Number of max epochs.')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='timesteps for diffusion')
    parser.add_argument('--l2_decay', type=float, default=0,
                        help='l2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=7,
                        help='cuda device.')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
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
    parser.add_argument('--dataloader_code', type=str, default='bert',
                    help='which dataloder , choose from[Diff,CBIT,bert]')
    parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
    parser.add_argument('--max_len', type=int, default=10,
                    help=' enable max seq len ')
    parser.add_argument('--slide_window_step', type=int, default=1,
                help=' slide window step ')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--mask_prob', type=int, default=0.15,
                    help='dataloder mask_prob')
    #######################train###########################
    parser.add_argument('--loss_type', type=str, default='ce',
                    help='which loss , choose from[ce,cl-mse,InfoNCE]')
    parser.add_argument('--time_emb_dim', type=int, default=64,
                    help=' time emb dim')
    parser.add_argument('--diffuser_type', type=str, default='Unet',
                        help='choose from[Unet,mlp1,mlp2]')
    parser.add_argument('--num_items', type=int, default=4,
                        help='num_items')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    #######################bert################################
    parser.add_argument('--bert_dropout', type=int, default=0.2,
                    help='bert_dropout')
    parser.add_argument('--bert_hidden_units', type=int, default=256,
                    help=' bert_hidden_units')
    parser.add_argument('--bert_mask_prob', type=int, default=0.15,
                        help='bert_mask_prob')
    parser.add_argument('--bert_num_blocks', type=int, default=2,
                        help='bert_num_blocks')
    parser.add_argument('--bert_num_heads', type=int, default=4,
                        help='bert_num_heads')
    parser.add_argument('--bert_max_len', type=int, default=10,
                        help='bert_max_len')
    #######################diffusion###########################
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=50, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.01, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.05, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.5, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=True, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
    parser.add_argument('--hidden_size', type=int, default=256,help='Number of hidden factors, i.e., embedding size.')
    #######################CBIT###########################
    parser.add_argument('--tau', type=float, default=0.3, help='contrastive loss temperature')
    parser.add_argument('--calcsim', type=str, default='cosine', choices=['cosine', 'dot'])
    parser.add_argument('--projectionhead', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.1, help='loss proportion learning rate')
    parser.add_argument('--lambda_', type=float, default=5, help='loss proportion significance indicator')
    
    # lr scheduler #
    parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
    parser.add_argument('--gamma', type=float, default=1, help='Gamma for StepLR')
    return parser.parse_args()

args = parse_args()

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(args.random_seed)

class Bert(nn.Module):
    def __init__(self, args,device):
        super(Bert, self).__init__()
        self.state_size = args.max_len
        self.hidden_size = args.hidden_size
        self.item_num = int(args.num_items)
        self.dropout_rate = args.dropout_rate
        self.diffuser_type = args.diffuser_type
        self.device = device
        #################################bert############################################
        self.bert = BERT(args)
        # 对于历史序列添加一层linear用来做交叉熵损失
        self.bert_out = nn.Linear(self.hidden_size, self.item_num+1)


    def forward(self, seqs):
        """
        对整个序列进行bert,这里需要注意的是这里是对seq是有mask_token的序列
            args:
                seqs (torch.tensor (B,S)): seq序列,因为这里的seq做了扩展
            return :
                inputs_emb  (torch.tensor (B,S,D)): 对seq序列做编码的结果 
                logits      (torch.tensor (B*S,V+1)): 对seq序列做logits的结果 ,因为token id 从1开始算,所以这里需要囊括0,所以V+1

        """
        inputs_encoding=self.bert(seqs)
        #先对输入序列进行logits,以便进行交叉熵loss
        logits =self.bert_out(inputs_encoding)
        # (B*S) x V
        logits = logits.view(-1, logits.size(-1))   


        return inputs_encoding,logits


    def predict(self, states,target,mask_indice):
        """
        用diffusion model 预测下一个token
            
            Args:
                states (torch.tensor (B,S+1,D)): 对seq序列做编码的结果 ,S+1包括第一个token,因为第一个token代表全局的信息
                target (torch.tensor (B,1)): 目标的token id
                diff: 扩散模型
                mask_indice (torch.tensor (B,1)) : 目标token在原序列中对应的位置
        """
        with torch.no_grad():
            inputs_encoding,seqs_logits=self.forward(states)
            # B x V
            scores4En = seqs_logits.view(states.size(0),states.size(1),-1)[:, -1, :] 
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
    total_loss=0
    num_total=len(data_loder)
    # 确保dropout关闭
    model.eval()

    IsNdcgIncrease=[0,0,0,0]
    IsHrIncrease=[0,0,0,0]

    for batch_idx, batch in enumerate(tqdm_dataloader4eval):
        batch = [x.to(device) for x in batch]
        seqs, target ,mask_indice= batch[0], batch[1],batch[2]
        batch_size=seqs.size(0)
        predictionEn= model.predict(seqs,target,mask_indice)


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
        if i == 1:
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
    log_file_name =  './Log/log-'+args.dataset_code +'-GlobalDiff-' +time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
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
    model = Bert(args,device)

    # 根据命令行参数中的 args.mean_type 的值来设置一个变量 mean_type，该变量将用于构建高斯扩散模型的拟合目标

   
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer =='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

    model.to(device)
    # loss function
    celoss_function=nn.CrossEntropyLoss(ignore_index=0)
    # lr scheduler
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)

    total_step=0
    hr_max = 0
    best_epoch = 0
    best_hr=0
    isStop=False
    # CBIT cl loss
    theta=0
    print('-------------------------- TEST PHRASE -------------------------')
    hr_10En = evaluate(model, test_loader, device,0,is_save=True)
    model.train()

    for i in range(args.epoch):
        start_time = Time.time()
        tqdm_dataloader = tqdm(train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(device) for x in batch]
            tokens,labels = batch
            optimizer.zero_grad()
            loss =0
            inputs_encoding,seqs_logits=model(tokens) 
            loss=celoss_function(seqs_logits.to(device),labels.view(-1))
            loss.backward()
            optimizer.step()

        scheduler.step()
        # 检查并设置最小学习率
        for param_group in optimizer.param_groups:
            if param_group['lr'] < 0.0001:
                param_group['lr'] =  0.0001
        # scheduler.step()
        if args.report_epoch:
            if i % 1 == 0:
                print("Epoch {:03d}; ".format(i) + 'Train loss: {:.10f}; '.format(loss) + "Time cost: " + Time.strftime(
                        "%H: %M: %S", Time.gmtime(Time.time()-start_time)))
            if (i + 1) % 1== 0 and (i+1)>15:
                
                eval_start = Time.time()
                # 验证集早停
                print('-------------------------- VAL PHRASE --------------------------')
                Valhr_10En = evaluate(model, val_loader, device,i,is_save=False)
                if Valhr_10En>best_hr:
                    best_epoch=i+1
                    print(best_epoch ,"update best model!")
                    best_hr=Valhr_10En
                    # 保存模型，按当前日期创建模型文件路径
                    model_path='./experiments' + '/' + Time.strftime("%Y-%m-%d", Time.gmtime()) + '/'
                    os.makedirs(model_path, exist_ok=True)
                    model_name=model_path+'Bert_epoch'+str(i+1)+'.pth'
                    torch.save(model.state_dict(), model_name)
                elif Valhr_10En<best_hr:
                    isStop=True
                print('-------------------------- TEST PHRASE -------------------------')
                hr_10En = evaluate(model, test_loader, device,i,is_save=True)
                print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-eval_start)))
                print('----------------------------------------------------------------')
            if (i+1)>=args.epoch:
                print("best epoch:" ,best_epoch)
                break
        # if(isStop):
        #     print("best epoch:" ,best_epoch)
        #     break

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

