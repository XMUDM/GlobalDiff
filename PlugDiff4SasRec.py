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
from bert_modules.transformer import TransformerBlock
from bert_modules.bert import BERT
import time
import sys
from dataloaders import dataloader_factory
from gaussian_diffusion4SasRec import GaussianDiffusion, ModelMeanType
from pretrain_models.SasRec.model import SASRec

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
    parser.add_argument('--dataloader_code', type=str, default='Diff',
                    help='which dataloder , choose from[Diff]')
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
    parser.add_argument('--loss_type', type=str, default='ce',
                    help='which loss , choose from[ce,cl-mse,InfoNCE,BPR]')
    parser.add_argument('--time_emb_dim', type=int, default=64,
                    help=' time emb dim')
    parser.add_argument('--diffuser_type', type=str, default='mlp2',
                        help='choose from[Unet,mlp1,mlp2]')
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

    #######################diffusion###########################
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=2, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.01, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.05, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.5, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
    


    


    
    
    
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
        self.hidden_size = args.hidden_size
        self.item_num = int(args.num_items)
        self.dropout_rate = args.dropout_rate
        self.diffuser_type = args.diffuser_type
        self.device = device
        self.time_emb_dim=args.time_emb_dim

        # self.item_embeddings = nn.Embedding(
        #     num_embeddings=self.item_num + 1,
        #     embedding_dim=self.hidden_size,
        # )
        # nn.init.normal_(self.item_embeddings.weight, 0, 1)
        #################################SASRec############################################
        self.SASRec = InitModel

        #注意！！token id从1开始算，所以这里的输出维度是item_num+1,最大的token id是item_num
        #对于最后的diffusion生成添加一层linear用来做交叉熵损失
        self.diff_out = nn.Linear(self.hidden_size, self.item_num+1)

        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*2 +self.time_emb_dim, self.hidden_size)
        )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
            nn.Linear(self.hidden_size*2  +self.time_emb_dim, self.hidden_size*2),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.hidden_size*2, self.hidden_size)
        )
        elif self.diffuser_type =='Unet':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*2 +self.time_emb_dim, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, math.ceil(self.hidden_size/2)),
                nn.Tanh(),
                nn.Linear( math.ceil(self.hidden_size/2), self.hidden_size),
            )

        
        #cross-attention preLinear
        self.w1=nn.Linear(self.item_num +1, self.hidden_size)


        self.drop = nn.Dropout(0.5)

    def forward(self, x, seqs, step,random_indice,pretrain=False):
        """
            输入 : x (batch_size, 1, V)
                   seqs (batch_size, seq)
                   random_indice : 采样的位置 (B,1)
            return : 
                predicted_x:(B, 1, V)
                logits (B,V)
                diff_logits (B,V)
        """
        # pretrain model 
        if(pretrain):
            h,logits=self.cacu_seq(seqs)
            diff_logits= x
            predicted_x = diff_logits.unsqueeze(1)
            return predicted_x,logits,diff_logits
        else:
            B,L,V=x.size()
            # x = torch.cat((x,h[:,-1,:].unsqueeze(1)),dim=-1)
            # x= self.w0(x)
            # x = torch.tanh(x)
            #####################cross-attention#############################
            x= F.normalize(x)

            h,logits=self.cacu_seq(seqs)
            # pretrain_logits=logits.view(B,-1,V)
            # #(B,V)
            # pretrain_target_logits= pretrain_logits[torch.arange(B), random_indice, :]
            # #(B,1,V)
            # pretrain_target_logits =pretrain_target_logits.unsqueeze(1)
            # pretrain_target_logits=F.normalize(pretrain_target_logits)
            # # 做个残差连接
            # x= x+ pretrain_target_logits
            # x = self.drop(x)
            x=self.w1(x)        
            # x =torch.tanh(x)
            # pretrain model
            # with torch.no_grad():


            # h=h.view(B,-1)
            # h=self.w2(h)
            # h=h.view(B,-1,D)
            # t = self.step_mlp(step)
            # t=t.unsqueeze(1).repeat(1,L,1)
            # (B,D)
            gathered_h = h [torch.arange(B), random_indice, :]
            #(B,1,2*D)
            cross_attended = torch.cat((x, gathered_h.unsqueeze(1)), dim=-1)
            cross_attended = self.drop(cross_attended)
            # # 计算 Q 和 K 的点积
            # dot_product = torch.bmm(x, h.transpose(1, 2))/ torch.sqrt(torch.tensor(h.size(-1), dtype=torch.float32))  # (batch_size, seq_len, seq_len)
            # # 计算注意力分布（使用 softmax 函数）
            # attention_weights = F.softmax(dot_product, dim=-1)
            # # 应用注意力权重到 Value 张量
            # cross_attended = torch.bmm(attention_weights, h)  # (batch_size, seq_len, embedding_dim)
            
            # cross_attended=x+h
            #time_step embedding
            t = self.timestep_embedding(step, self.time_emb_dim).unsqueeze(1).repeat(1,L,1).to(x.device)
            # t = self.step_mlp(step)
            # t=t.unsqueeze(1).repeat(1,L,1)

            if self.diffuser_type == 'mlp1':
                res = self.diffuser(torch.cat((cross_attended, t), dim=-1).view(B,-1))
            elif self.diffuser_type == 'mlp2':
                res = self.diffuser(torch.cat((cross_attended, t), dim=-1).view(B,-1))
            elif self.diffuser_type == 'Unet':
                res = self.diffuser(torch.cat((cross_attended, t), dim=-1).view(B,-1))
            
            # diffusion logits (B,V)
            diff_logits=self.diff_out(res)
            predicted_x = diff_logits.unsqueeze(1)

            return predicted_x,logits,diff_logits

    
    def cacu_seq(self, states):
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

    # 用于生成时间步长的嵌入表示，通常用于在序列模型中对时间信息进行编码。
    def timestep_embedding(self,timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        # 首先计算嵌入维度的一半，然后生成一组频率。这些频率是通过应用指数函数到一个线性序列来获得的，以便控制正弦波的频率。
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        # 函数创建一个参数矩阵 args，其中每一行是一个时间步长与频率的乘积。这将用于计算正弦和余弦函数值
        args = timesteps[:, None].float() * freqs[None]
        # 函数分别计算每个时间步长对应的正弦和余弦函数值，并将它们连接在一起以形成最终的嵌入张量。
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 如果嵌入维度是奇数，函数会在最后一列添加一个全零的列
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def predict(self, states,target, diff,mask_indice):
        with torch.no_grad():
            h,logits=self.cacu_seq(states)
            En_target_logits=logits.view(states.size(0),states.size(1),-1)[:, -1, :].unsqueeze(1)
            # using diffusion
            # diff_logits= diff.sample(self.forward, En_target_logits, states)
            #using gs_diffusion
            diff_logits = diff.p_sample(self.forward, En_target_logits, states,mask_indice, args.sampling_steps, args.sampling_noise)
            scores4diff =  diff_logits
            # scores[torch.arange(scores.size(0)).unsqueeze(1), states] = 9999
            # B x V
            scores4En = logits.view(states.size(0),states.size(1),-1)[:, -1, :] 
        return scores4diff,scores4En



def evaluate(model, data_loder, diff, device,epoch_index,is_save):

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

    for batch_idx, batch in enumerate(tqdm_dataloader4eval):
        batch = [x.to(device) for x in batch]
        seqs, target ,mask_indice= batch[0], batch[1],batch[2]
        batch_size=seqs.size(0)
        prediction , predictionEn= model.predict(seqs,target, diff,mask_indice)

 
        # _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        # _, topK = prediction.topk(100, dim=1, largest=False, sorted=True)
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        sorted_list2 = sorted_list2
        calculate_hit(sorted_list2,topk,target.tolist(),hit_purchase,ndcg_purchase)
        total_purchase+=batch_size
        # print(hit_purchase)
        # print(ndcg_purchase)
        # print('==========')
        ###########################Encoder 的指标######################################
        _, topKEn = predictionEn.topk(100, dim=1, largest=True, sorted=True)
        topKEn = topKEn.cpu().detach().numpy()
        sorted_listEn=np.flip(topKEn,axis=1)
        sorted_listEn = sorted_listEn
        calculate_hit(sorted_listEn,topk,target.tolist(),hit_purchase4En,ndcg_purchase4En)

 


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

    print('#############################diffusion#########################################')
    hr_list = []
    ndcg_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)

        if i == 0:
            hr_10 = hr_purchase

    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    if is_save:
        save_metrics("./results/diffusion.csv",epoch_index,hr_list,ndcg_list,topk)

   

    return hr_10En,hr_10


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
    log_file_name =  './Log/log-'+args.dataset_code + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
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
    timesteps = args.timesteps
    ##########################加载训练好的SASRec模型##################################################################
    # 因为SASRec模型定义需要itemnum,usernum，但是usernum没有用到，所以这里可以传0进去
    SAS_model = SASRec(0,item_num,args)
    # best_model = torch.load('./pretrain_models/SasRec/ml1m.pth').get('model_state_dict')
    SAS_model.load_state_dict(torch.load('./pretrain_models/SasRec/ml1m.pth', map_location=device))
    # SAS_model.load_state_dict(best_model)
    SAS_model.to(device)
    model = Tenc(args,device,SAS_model)

 # 根据命令行参数中的 args.mean_type 的值来设置一个变量 mean_type，该变量将用于构建高斯扩散模型的拟合目标
    ### Build Gaussian Diffusion ###
    if args.mean_type == 'x0':
        mean_type = ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)
    diff = GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, device,args.num_items).to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=20)
    
    model.to(device)
    # optimizer.to(device)

    total_step=0
    hr_max = 0

    best_epoch = 0
    best_hr=0
    isStop=False

    # print('-------------------------- TEST PHRASE -------------------------')
    # _ = evaluate(model, test_loader, diff, device,0,is_save=False)

    for i in range(args.epoch):
        start_time = Time.time()
        tqdm_dataloader = tqdm(train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(device) for x in batch]
            seqs, target, negative_target, random_indice,seqs_labels = batch
            optimizer.zero_grad()
            # loss, predicted_x = diff.p_losses(model,  ori_target_embedding ,mask_seq_encoding ,negative_target_embedding,seqs_labels,logits, n,curr_epoch=i, loss_type=args.loss_type)
            loss, predicted_x = diff.p_losses(model, seqs,seqs_labels,target, negative_target ,random_indice,curr_epoch=i, loss_type=args.loss_type)
            loss.backward()
            optimizer.step()

        # scheduler.step()
        if args.report_epoch:
            if i % 1 == 0:
                print("Epoch {:03d}; ".format(i) + 'Train loss: {:.10f}; '.format(loss) + "Time cost: " + Time.strftime(
                        "%H: %M: %S", Time.gmtime(Time.time()-start_time)))

            if (i + 1) % 10== 0:
                
                eval_start = Time.time()
                # print('-------------------------- TRAIN PHRASE --------------------------')
                # _ = evaluate(model, train_loader, diff, device)
                print('-------------------------- VAL PHRASE --------------------------')
                Valhr_10En,Valhr_10 = evaluate(model, val_loader, diff, device,i,is_save=False)
                if Valhr_10En<Valhr_10 and Valhr_10>best_hr:
                    best_epoch=i+1
                    print(best_epoch)
                    best_hr=Valhr_10
                elif Valhr_10<best_hr:
                    isStop=True
                print('-------------------------- TEST PHRASE -------------------------')
                hr_10En,hr_10 = evaluate(model, test_loader, diff, device,i,is_save=True)
                print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-eval_start)))
                print('----------------------------------------------------------------')

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        if(isStop):
            print("best epoch:" ,best_epoch)
            break
