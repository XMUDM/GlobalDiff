# _*_ encoding = utf-8 _*_
# @Author  : lmx
# @Time : 2024/5/6

from .base import AbstractDataloader
import torch
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import random
class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.max_len = args.max_len
        self.CLOZE_MASK_TOKEN = self.item_count+1
        self.GLOBAL_MASK_TOKEN = self.item_count+2
        # self.GLOBAL_MASK_TOKEN = 0
        self.mask_prob = args.mask_prob
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        # 训练用滑动窗口划分
        self.train_slidewindow, self.train_slidewindow_by_user, self.user_count_slidewindow = self.get_train_dataset_slidewindow(args.slide_window_step)
        # self.generate_dataset4DreamRec()
        self.items_cos_similarity=self.generate_items_cos_similarity_based_on_train_ui_matrix_()
    @classmethod
    def code(cls):
        return 'bert'
    


    def generate_dataset4DreamRec(self):
        """
        生成对齐DreamRec的数据集
        """
        print("generate_dataset4DreamRec...")
        data_statis_file_name =  './generated_dataset/data_statis.df'
        train_file_name='./generated_dataset/train_data.df'
        val_file_name='./generated_dataset/val_data.df'
        test_file_name='./generated_dataset/test_data.df'
        ########################data_statis########################
        data_statis = {'seq_size': [self.args.max_len], 'item_num': [self.item_count]}
        df = pd.DataFrame(data_statis)
        df.to_pickle(data_statis_file_name)
        # ########################train_data 定长生成########################
        # train_list=[]
        # target_list=[]
        # for user in range(self.user_count):
        #     seq = self.train[user]
        #     seq_len = len(seq)
        #     beg_idx = list(range(seq_len-self.max_len-1, 0, -self.args.slide_window_step))
        #     assert seq_len>=self.max_len
        #     beg_idx.append(0)
        #     for i in beg_idx:
        #         temp_seq = seq[i:i + self.max_len]
        #         target = seq[i + self.max_len]
        #         train_list.append(temp_seq)
        #         target_list.append(target)
        # train_data = {'seq': train_list, 'len_seq': [self.args.max_len]*len(train_list), 'next': target_list }
        # df = pd.DataFrame(train_data)
        # # 将 DataFrame 存储为 .df 文件
        # df.to_pickle(train_file_name)
        ########################train_data########################
        train_list=[]
        target_list=[]
        train_len=[]
        for user in range(self.user_count):
            seq = self.train[user]
            seq_len = len(seq)
            beg_idx = list(range(seq_len-self.max_len-1, 0, -self.args.slide_window_step))
            assert seq_len>=self.max_len
            beg_idx.append(0)
            for i in beg_idx:
                temp_seq = [self.item_count] * self.max_len
                train_len.append(1)
                for j in range(self.max_len):
                    train_list.append(temp_seq.copy())
                    target = seq[i+j]
                    target_list.append(target) 
                    temp_seq[j] = seq[i + j]
                    train_len.append(j+1)
                train_list.append(temp_seq)
                target = seq[i + self.max_len]
                target_list.append(target)
        
        train_data = {'seq': train_list, 'len_seq': train_len, 'next': target_list }
        df = pd.DataFrame(train_data)
        # 将 DataFrame 存储为 .df 文件
        df.to_pickle(train_file_name)
        print(df.head(20))
        ########################val_data########################
        val_list=[]
        target_list=[]
        for user in range(self.user_count):
            seq = self.train[user]
            target = self.val[user][0]
            temp_seq = seq[-self.max_len:]
            val_list.append(temp_seq)
            target_list.append(target)
        
        val_data = {'seq': val_list, 'len_seq': [self.args.max_len]*len(val_list), 'next': target_list }
        df = pd.DataFrame(val_data)
        # 将 DataFrame 存储为 .df 文件
        df.to_pickle(val_file_name)
        ########################test_data########################
        test_list=[]
        target_list=[]
        for user in range(self.user_count):
            seq = self.train[user] + self.val[user]
            target = self.test[user][0]
            temp_seq = seq[-self.max_len:]
            test_list.append(temp_seq)
            target_list.append(target)
        test_data = {'seq': test_list, 'len_seq': [self.args.max_len]*len(test_list), 'next': target_list }
        df = pd.DataFrame(test_data)
        # 将 DataFrame 存储为 .df 文件
        df.to_pickle(test_file_name)


    def generate_train_ui_matrix(self):
        """
        生成训练集的ui矩阵
            return :
                ui_matrix (np.array) : 训练集的ui矩阵
        """
        print("generate_train_ui_matrix...")
        # item编号从1开始算
        ui_matrix=np.zeros((self.user_count, self.item_count+1))
        for user in tqdm(range(self.user_count),desc='generate_train_ui_matrix'):
            seq = self.train[user]
            for item in seq:
                ui_matrix[user][item] = 1
        return ui_matrix


    def generate_items_cos_similarity_based_on_train_ui_matrix_(self):
        """
            计算items的余弦相似度
            return :
                items_cos_similarity (np.array) : 记录items的余弦相似度
        """
        print("generate_items_cos_similarity_based_on_train_ui_matrix...")
        ui_matrix = self.generate_train_ui_matrix()
        items_cos_similarity = cosine_similarity(ui_matrix.T)
        print(items_cos_similarity.shape)
        return items_cos_similarity
    
    def get_train_dataset_slidewindow(self, step=10):
        """
        对于训练数据进行滑动窗口划分,滑动窗口大小为max_len,值得注意的是这里做滑动窗口就已经保证了训练序列的长度一定为max_len
            param:
                step : 滑动窗口滑动步长
            return:
                train_slidewindow : 记录slidewindow 划分后的 {user : seqs} 字典,seqs为一个数组
                train_slidewindow_by_user : 记录slidewindow 划分前的 {user : [seqs,...]} 字典,seqs为一个数组
                real_user_count : 滑动窗口划分后的真实样本个数
        """
        real_user_count=0
        # 记录slidewindow 划分后的 user : seqs
        train_slidewindow={}
        # 记录slidewindow 划分前的 user ：[seqs,...]
        train_slidewindow_by_user = {}
        for user in range(self.user_count):
            seq = self.train[user]
            seq_len = len(seq)
            beg_idx = list(range(seq_len-self.max_len, 0, -step))
            assert seq_len>=self.max_len
            beg_idx.append(0)
            for i in beg_idx:
                temp = seq[i:i + self.max_len]
                train_slidewindow[real_user_count] = temp
                l = train_slidewindow_by_user.get(user,[])
                l.append(temp)
                train_slidewindow_by_user[user] = l
                real_user_count+=1
        return train_slidewindow, train_slidewindow_by_user, real_user_count
    
    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.batch_size,
                                           shuffle=True, pin_memory=True,drop_last=False)
        return dataloader

    def _get_train_dataset(self):
        """
            生成训练集的dataset
            return:
                dataset (DiffTrainDataset) : 训练集的dataset
        """
        dataset = DiffTrainDataset(self.train,self.max_len,self.CLOZE_MASK_TOKEN,self.GLOBAL_MASK_TOKEN, self.item_count,self.mask_prob,self.items_cos_similarity,self.rng)
        # print(len(dataset))
        # dataset = DiffTrainDataset(self.train_slidewindow,self.max_len, self.CLOZE_MASK_TOKEN,self.GLOBAL_MASK_TOKEN, self.item_count,self.mask_prob,self.items_cos_similarity)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.batch_size if mode == 'val' else self.args.batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=False,drop_last=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            answers = self.val 
            dataset = DiffEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN,self.GLOBAL_MASK_TOKEN,self.item_count)
        else :
            answers = self.test
            val = self.val
            dataset = DiffTestDataset(self.train, answers,val, self.max_len, self.CLOZE_MASK_TOKEN,self.GLOBAL_MASK_TOKEN,self.item_count)

        return dataset


class DiffTrainDataset(data_utils.Dataset):
    """
        随机mask序列的dataset
            Attributes:
                u2seq (dict) : 记录每个user的序列
                max_len (int) : 序列的最大长度
                mask_token (int) : 随机mask的目标
                num_items (int) : 序列中item的个数
                mask_prob (float) : mask概率
                items_cos_similarity (np.array) : 记录items的余弦相似度

    """
    def __init__(self, u2seq, max_len,  mask_token,global_token,num_items,mask_prob,items_cos_similarity,rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_token = mask_token
        self.global_token = global_token
        self.num_items = num_items
        self.mask_prob =mask_prob
        self.items_cos_similarity = items_cos_similarity
        self.rng = rng


    def __len__(self):
        return len(self.users)


    def __getitem__(self, index):
        """
        从dataset中获得某个样本
            Args :
                index : 样本编号 
            return :
                tokens (torch.tensor (S+1)): 随机MASK后的序列,因为在最前面拼接了全局的特殊token,所以这里+1 
                target : MASK掉的目标
                negative_target : 负样本
                random_indice : MASK的目标的索引
                labels (torch.tensor (S)):  对原本的seq中mask掉的部分置位0,用于交叉熵损失中避免随机mask的label也进行计算.也就是整个seqs的label
                target_logits (torch.tensor (V+1)): 利用余弦相似度构建一个目标item的logits,这里+1是因为原本item的编号从1开始算
        """
        user = self.users[index]
        seq = self._getseq(user)[-self.max_len:]
        assert len(seq)==self.max_len
        tokens = torch.LongTensor(seq).clone()
        labels = torch.LongTensor(seq).clone()

        # 类似于bert，这里需要对序列做一个随机mask，也就是cloze 任务
        for index in range(len(tokens)):
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens[index]=self.mask_token
                elif prob < 0.9:
                    tokens[index]=torch.randint(low=1, high=self.num_items, size=(1,)).item()
                else:
                    tokens[index]=tokens[index]
            else:
                tokens[index]=tokens[index]
                labels[index]=0


        return tokens, labels
    
    def _getseq(self, user):
        return self.u2seq[user]



class DiffEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token,global_token,item_count):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.global_token = global_token
        self.item_count=item_count


    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        """
        从dataset中获得某个样本
            param :
                index : 样本编号 
            return :
                tokens : 随机MASK后的序列 
                target : MASK掉的目标
                mask_indice : MASK的目标的索引,这里的mask固定为最后一个token
        """
        user = self.users[index]
        seq = self.u2seq[user]
        answers = self.u2answer[user]
        answer= answers[0]
        answer=np.array(answer)
        tokens = torch.LongTensor(seq).clone()[-self.max_len:]
        mask_indice = self.max_len-1
        tokens[0:self.max_len-1] = tokens[1:].clone()
        tokens[-1] = self.mask_token


        return tokens, torch.from_numpy(answer), mask_indice

class DiffTestDataset(data_utils.Dataset):
    def __init__(self, u2seq,u2eval, u2answer, max_len, mask_token,global_token,item_count):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.u2eval = u2eval
        self.max_len = max_len
        self.mask_token = mask_token
        self.global_token = global_token
        self.item_count=item_count


    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        """
        从dataset中获得某个样本
            param :
                index : 样本编号 
            return :
                tokens : 随机MASK后的序列 
                target : MASK掉的目标
                mask_indice : MASK的目标的索引,这里的mask固定为最后一个token
        """
        user = self.users[index]
        seq = self.u2seq[user] + self.u2eval[user]
        answers = self.u2answer[user]
        answer= answers[0]
        answer=np.array(answer)
        tokens = torch.LongTensor(seq).clone()[-self.max_len:]
        mask_indice = self.max_len-1
        tokens[0:self.max_len-1] = tokens[1:].clone()
        tokens[-1] = self.mask_token

        return tokens, torch.from_numpy(answer), mask_indice
