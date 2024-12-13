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
from Bert.bert_modules.bert import BERT
import time
import sys
from dataloaders import dataloader_factory
from gaussian_diffusion import GaussianDiffusion
from gaussian_diffusion import ModelMeanType
from unet.unet import Unet
import torch.optim as optim
# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of max epochs.')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='timesteps for diffusion')
    parser.add_argument('--l2_decay', type=float, default=0,
                        help='l2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=2,
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
    parser.add_argument('--min_rating', type=int, default=1,
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
    parser.add_argument('--steps', type=int, default=20, help='diffusion steps')
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

class Tenc(nn.Module):
    def __init__(self, args,device):
        super(Tenc, self).__init__()
        self.state_size = args.max_len
        self.hidden_size = args.hidden_size
        self.item_num = int(args.num_items)
        self.dropout_rate = args.dropout_rate
        self.diffuser_type = args.diffuser_type
        self.device = device
        self.time_emb_dim=args.time_emb_dim

        #################################bert############################################
        self.bert = BERT(args)
        # 对于历史序列添加一层linear用来做交叉熵损失
        self.bert_out = nn.Linear(self.hidden_size, self.item_num+1)

        #对于最后的diffusion生成添加一层linear用来做交叉熵损失
        self.diff_out = nn.Linear(self.hidden_size, self.item_num+1)

        self.item_embeddings=self.bert.embedding.token


        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3 +self.time_emb_dim, self.hidden_size)
        )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
            nn.Linear(self.hidden_size*3  +self.time_emb_dim, self.hidden_size*2),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.hidden_size*2, self.hidden_size)
        )


        
        #cross-attention preLinear
        self.w1=nn.Linear(self.item_num+1 , self.hidden_size)
        self.drop = nn.Dropout(self.dropout_rate)

        # Unet modules
        self.linear_for_unet_input=nn.Linear(self.hidden_size*3  +self.time_emb_dim,256)
        self.unet=Unet().to(self.device)
        self.unet_output=nn.Linear(256, self.hidden_size)


    def forward(self, x, seqs, step,random_indice,pretrain=False):
        """
            args
                x (torch.tensor (B,1,V+1)): target item的模拟logits,这里V+1是因为token id从1开始算的,但是要算上0 
                seqs (torch.tensor (B,S+1): 历史序列的token id,这里S+1是因为在序列的最前面拼接了一个global token
                random_indice (torch.tensor (B)): target item对应在原seq中的位置
            return : 
                predicted_x (torch.tensor (B,1,V+1)): 模型预测的target item的模拟logits
                seqs_logits (torch.tensor (B*S,V+1)): bert预测的历史序列的logits
                diff_logits (torch.tensor (B,V+1)): 模型预测的target item的logits

        """

        B,L,V=x.size()
        #####################cross-attention#############################
        x= F.normalize(x)
        inputs_encoding,seqs_encoding,seqs_logits,global_encoding=self.cacu_seq(seqs)
        x=self.w1(x)        
        # (B,D)
        target_encoding = seqs_encoding [torch.arange(B), random_indice, :]
        # #(B,1,2*D)
        # cross_attended_encoding = torch.cat((x,global_encoding.unsqueeze(1)), dim=-1)
        #(B,1,3*D)
        cross_attended_encoding = torch.cat((x,global_encoding.unsqueeze(1),target_encoding.unsqueeze(1)), dim=-1)
        cross_attended_encoding = self.drop(cross_attended_encoding)
        #time_step embedding
        t = self.timestep_embedding(step, self.time_emb_dim).unsqueeze(1).repeat(1,L,1).to(x.device)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((cross_attended_encoding, t), dim=-1).view(B,-1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((cross_attended_encoding, t), dim=-1).view(B,-1))
        elif self.diffuser_type == 'Unet':
                res=torch.cat((cross_attended_encoding, t), dim=-1).view(B,-1)
                res=self.linear_for_unet_input(res)
                res=res.view(B,1,16,16)
                res=self.unet(res)
                res=res.view(B,-1)
                res=self.unet_output(res)
        # diffusion logits (B,V)
        diff_logits=self.diff_out(res)
        predicted_x = diff_logits.unsqueeze(1)
        return predicted_x,seqs_logits,diff_logits,inputs_encoding


    def cacu_seq(self, seqs):
        """
        对整个序列进行bert,这里需要注意的是这里是对seq是有mask_token的序列
            args:
                seqs (torch.tensor (B,S+1)): seq序列,因为这里的seq做了扩展,有一个special token 来获得全局的信息
            return :
                inputs_emb  (torch.tensor (B,S,D)): 对seq序列做编码的结果 ,不包括第一个token
                logits      (torch.tensor (B*S,V+1)): 对seq序列做logits的结果 ,因为token id 从1开始算,所以这里需要囊括0,所以V+1
                global_emb  (torch.tensor (B,D)): 对seq序列做全局编码的结果 ,就是第一个token的编码
        """
        inputs_encoding=self.bert(seqs)
        seqs_encoding=inputs_encoding[:,1:,:].clone()
        #先对输入序列进行logits,以便进行交叉熵loss,这里inputs_emb不包括第一个token,因为它代表全局的信息
        logits =self.bert_out(seqs_encoding)
  
        # (B*S) x V
        logits = logits.view(-1, logits.size(-1))   
        # 获取全局信息,这里inputs_emb包括第一个token,因为它代表全局的信息
        global_encoding = inputs_encoding[:,0]

        return inputs_encoding,seqs_encoding,logits,global_encoding

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
        """
        用diffusion model 预测下一个token
            
            Args:
                states (torch.tensor (B,S+1,D)): 对seq序列做编码的结果 ,S+1包括第一个token,因为第一个token代表全局的信息
                target (torch.tensor (B,1)): 目标的token id
                diff: 扩散模型
                mask_indice (torch.tensor (B,1)) : 目标token在原序列中对应的位置
        """
        with torch.no_grad():
            inputs_encoding,seqs_encoding,seqs_logits,global_encoding=self.cacu_seq(states)
            # target=torch.randint(0, item_num , (h.shape[0],1)).to(h.device)
            # x = self.cacu_x(target)
            x = seqs_encoding[:,-1,:].unsqueeze(1)
            assert x.shape == (seqs_encoding.size(0),1,seqs_encoding.size(2))
            # 这里的取序列中最后一个item的logits,注意这里是states.size(1)-1因为states中第一个token代表全局信息，所以states的长度比seqs的长度多1
            En_target_logits=seqs_logits.view(states.size(0),states.size(1)-1,-1)[:, -1, :].unsqueeze(1)
            #using gs_diffusion
            diff_logits = diff.p_sample(self.forward, En_target_logits, states, mask_indice, args.sampling_steps, args.sampling_noise)
            scores4diff =  diff_logits
            scores4diff[torch.arange(scores4diff.size(0)).unsqueeze(1), states[:,1:-1]] = -9999
            # B x V,这里的states.size(1)-1就是seq_len，去掉了pandding golbal token
            scores4En = seqs_logits.view(states.size(0),states.size(1)-1,-1)[:, -1, :] 
            scores4En[torch.arange(scores4En.size(0)).unsqueeze(1), states[:,1:-1]] = -9999
        return scores4diff,scores4En

class NTXENTloss(nn.Module):
    """
    对比学习的模型
    """
    def __init__(self, args, device , temperature=1.):
        super(NTXENTloss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.projection_dim = args.bert_hidden_units
        self.device = device
        self.w1 = nn.Linear(self.projection_dim, self.projection_dim, bias=False).to(self.device)
        self.bn1 = nn.BatchNorm1d(self.projection_dim).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.w2 = nn.Linear(self.projection_dim, self.projection_dim, bias=False).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False).to(self.device)
        #self.cossim = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def project(self, h):
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(h)))))
    def cosinesim(self,h1,h2):
        h = torch.matmul(h1, h2.T)
        h1_norm2 = h1.pow(2).sum(dim=-1).sqrt().view(h.shape[0],1)
        h2_norm2 = h2.pow(2).sum(dim=-1).sqrt().view(1,h.shape[0])
        return h/(h1_norm2@h2_norm2)
    def forward(self, h1, h2,calcsim='dot'):
        """
        注意这里用(self.args.bert_max_len+1)而不是self.args.bert_max_len是因为序列添加了special token来捕获全局信息,具体就是在seq前面添加了固定的token
        """
        b = h1.shape[0]
        if self.args.projectionhead:
            z1, z2 = self.project(h1.view(b*(self.args.bert_max_len+1),self.args.bert_hidden_units)), self.project(h2.view(b*(self.args.bert_max_len+1),self.args.bert_hidden_units))
        else:
            z1, z2 = h1, h2
        z1 = z1.view(b, (self.args.bert_max_len+1)*self.args.bert_hidden_units)
        z2 = z2.view(b, (self.args.bert_max_len+1)*self.args.bert_hidden_units)


        if calcsim=='dot':
            sim11 = torch.matmul(z1, z1.T) / self.temperature
            sim22 = torch.matmul(z2, z2.T) / self.temperature
            sim12 = torch.matmul(z1, z2.T) / self.temperature
        elif calcsim=='cosine':
            sim11 = self.cosinesim(z1, z1) / self.temperature
            sim22 = self.cosinesim(z2, z2) / self.temperature
            sim12 = self.cosinesim(z1, z2) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
        ntxentloss = self.criterion(raw_scores, targets)
        return ntxentloss

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
    hit_purchase4fusion=[0,0,0,0]
    ndcg_purchase4fusion=[0,0,0,0]
    tqdm_dataloader4eval = tqdm(data_loder)
    total_loss=0
    num_total=len(data_loder)
    # 确保dropout关闭
    model.eval()


    En_win_ndcg=[0,0,0,0]
    En_win_hr=[0,0,0,0]

    Diff_win_ndcg=[0,0,0,0]
    Diff_win_hr=[0,0,0,0]

    for batch_idx, batch in enumerate(tqdm_dataloader4eval):
        batch = [x.to(device) for x in batch]
        seqs, target ,mask_indice= batch[0], batch[1],batch[2]
        batch_size=seqs.size(0)
        prediction , predictionEn= model.predict(seqs,target, diff,mask_indice)

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

        ####################################Borda Count法 fusion logits##########################################
        alpha=0.5
        # 对每个样本计算加权Borda得分
        # 计算每个样本的排名
        ranks1 = prediction.argsort(dim=1, descending=False).argsort(dim=1)
        ranks2 = predictionEn.argsort(dim=1, descending=False).argsort(dim=1)
        # 计算加权排名,因为要融合，所以要标准化
        weighted_ranks1 = ranks1 * F.softmax(prediction,dim=1)
        weighted_ranks2 = ranks2 * F.softmax(predictionEn,dim=1)
        # 加权排名相加
        weighted_borda_scores = (weighted_ranks1*alpha) + weighted_ranks2*(1-alpha)
        ###########################fusion两个logitis之后的指标######################################
        _, topK_fusion = weighted_borda_scores.topk(100, dim=1, largest=True, sorted=True)
        topK_fusion = topK_fusion.cpu().detach().numpy()
        sorted_list_fusion=np.flip(topK_fusion,axis=1)
        calculate_hit(sorted_list_fusion,topk,target.tolist(),hit_purchase4fusion,ndcg_purchase4fusion)

        def CountNdcgHR(sorted_list2,sorted_listEn,true_items):
            for i in range(len(topk)):
                rec_listEn = sorted_listEn[:, -topk[i]:]
                rec_list = sorted_list2[:, -topk[i]:]
                for j in range(len(true_items)):
                    # 如果都在看哪个排名高
                    if true_items[j] in rec_list[j] and true_items[j] in rec_listEn[j]:
                        rankEn = topk[i] - np.argwhere(rec_listEn[j] == true_items[j])
                        rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                        if rank > rankEn:
                            En_win_ndcg[i]+=1
                        elif rank < rankEn:
                            Diff_win_ndcg[i]+=1
                        else:
                            continue
                    # 如果看命中呢
                    if true_items[j] in rec_list[j] and true_items[j] not in rec_listEn[j]:
                        Diff_win_hr[i]+=1
                    elif true_items[j] not in rec_list[j] and true_items[j] in rec_listEn[j]:
                        En_win_hr[i]+=1
        CountNdcgHR(sorted_list2,sorted_listEn,target.tolist())
 
    print("#############################分别对比encoder以及diffusion的logitis指标是否增加######################################")
    
    print("En_win_hr",En_win_hr)
    print("Diff_win_hr",Diff_win_hr)
    print("En_win_ndcg",En_win_ndcg)
    print("Diff_win_ndcg",Diff_win_ndcg)

        
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
        save_metrics("./results/encoder1.csv",epoch_index,hr_list,ndcg_list,topk)

    print('#############################diffusion#########################################')
    hr_list = []
    ndcg_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)

        if i == 1:
            hr_10 = hr_purchase

    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    if is_save:
        save_metrics("./results/diffusion1.csv",epoch_index,hr_list,ndcg_list,topk)


    print('#############################fuse diffsion and Encoder#########################################')
    hr_list = []
    ndcg_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase4fusion[i]/total_purchase
        ng_purchase=ndcg_purchase4fusion[i]/total_purchase

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)

        if i == 1:
            hr_10fusion = hr_purchase

    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    if is_save:
        save_metrics("./results/fusion1.csv",epoch_index,hr_list,ndcg_list,topk)


    return hr_10En,hr_10,hr_10fusion


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
    timesteps = args.timesteps
    model = Tenc(args,device)

    ntxentloss_model = NTXENTloss(args,device,args.tau)
 # 根据命令行参数中的 args.mean_type 的值来设置一个变量 mean_type，该变量将用于构建高斯扩散模型的拟合目标
    ### Build Gaussian Diffusion ###
    if args.mean_type == 'x0':
        mean_type = ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)
    diff = GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, device,item_num).to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
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
    hr_10En,hr_10,hr_10fusion = evaluate(model, test_loader, diff, device,0,is_save=True)
    model.train()

    for i in range(args.epoch):
        start_time = Time.time()
        tqdm_dataloader = tqdm(train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(device) for x in batch]
            seqs, target, negative_target, random_indice,seqs_labels,target_logits,pos_tokens,pos_labels = batch
            optimizer.zero_grad()
            # loss, predicted_x = diff.p_losses(model,  ori_target_embedding ,mask_seq_encoding ,negative_target_embedding,seqs_labels,logits, n,curr_epoch=i, loss_type=args.loss_type)
            loss, predicted_x = diff.p_losses(model, seqs,seqs_labels,target,target_logits,negative_target,random_indice,curr_epoch=i,loss_type=args.loss_type)
            # 添加上CBIT的对比学习loss
            cl_loss = 0
            pos_loss = 0
            pos_pairs=[]
            num_pos=pos_tokens.shape[1]

            inputs_encoding,seqs_encoding,seqs_logits,global_encoding=model.cacu_seq(seqs) 
            #编码器的交叉熵损失
            pos_loss+=celoss_function(seqs_logits.to(device),seqs_labels.view(-1))
            pos_pairs.append(inputs_encoding)
            for j in range(num_pos):
                inputs_encoding,seqs_encoding,seqs_logits,global_encoding=model.cacu_seq(pos_tokens[:,j,:]) 
                #编码器的交叉熵损失
                curr_labels=pos_labels[:,j,:].clone().view(-1)
                pos_loss+=celoss_function(seqs_logits.to(device),curr_labels)
                pos_pairs.append(inputs_encoding)
            for j in range(len(pos_pairs)):
                for k in range(len(pos_pairs)):
                    if j!=k:
                        cl_loss = ntxentloss_model(pos_pairs[j], pos_pairs[k], calcsim=args.calcsim) + cl_loss
            # loss += pos_loss
            # num_main_loss = loss.detach().data.item() 
            # num_cl_loss = cl_loss.detach().data.item()
            # theta_hat = num_main_loss/(num_main_loss+args.lambda_*num_cl_loss)
            # theta = args.alpha*theta_hat+(1-args.alpha)*theta
            # loss = loss + theta*cl_loss
            # print("pos_loss:",pos_loss.detach().data.item(),"cl_loss:",cl_loss.detach().data.item())
            loss = loss + cl_loss + pos_loss
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
            if (i + 1) % 1== 0 and (i+1)>5:
                
                eval_start = Time.time()
                # 验证集早停
                print('-------------------------- VAL PHRASE --------------------------')
                Valhr_10En,Valhr_10 ,Valhr_10Fusion= evaluate(model, val_loader, diff, device,i,is_save=False)
                if Valhr_10Fusion>best_hr:
                    best_epoch=i+1
                    print(best_epoch ,"update best model!")
                    best_hr=Valhr_10Fusion
                    # 保存模型，按当前日期创建模型文件路径
                    model_path='./experiments' + '/' + Time.strftime("%Y-%m-%d", Time.gmtime()) + '/'
                    os.makedirs(model_path, exist_ok=True)
                    model_name=model_path+'GlobalDiff_epoch'+str(i+1)+'.pth'
                    torch.save(model.state_dict(), model_name)
                elif Valhr_10Fusion<best_hr:
                    isStop=True
                print('-------------------------- TEST PHRASE -------------------------')
                hr_10En,hr_10,hr_10Fusion = evaluate(model, test_loader, diff, device,i,is_save=True)
                print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-eval_start)))
                print('----------------------------------------------------------------')
            if (i+1)>=args.epoch:
                print("best epoch:" ,best_epoch)
                break
        # 验证集早停
        # if(isStop):
        #     print("best epoch:" ,best_epoch)
        #     break

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

