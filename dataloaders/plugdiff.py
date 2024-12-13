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
from sklearn.cluster import KMeans
import torch.nn.functional as F
import random
class PlugDiffDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.max_len = args.max_len
        self.CLOZE_MASK_TOKEN = self.item_count+1
        # self.GLOBAL_MASK_TOKEN = 0
        self.mask_prob = args.mask_prob
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        # 训练用滑动窗口划分
        self.train_slidewindow, self.train_slidewindow_by_user, self.user_count_slidewindow = self.get_train_dataset_slidewindow(args.slide_window_step)
        # self.generate_dataset4DreamRec()
        self.items_cos_similarity=self.generate_items_cos_similarity_based_on_train_ui_matrix_()
        print(self.items_cos_similarity)
        # self.train_ui_matrix=self.generate_train_ui_matrix()
    @classmethod
    def code(cls):
        return 'GlobalDiff'
    


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
            for item in seq[-self.max_len:]:
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
        # print(ui_matrix)
        items_cos_similarity = cosine_similarity(ui_matrix.T)
        # print("#"*100)
        # print(items_cos_similarity)
        # print("NXN:",np.max(items_cos_similarity))
        # non_zero_counts = np.count_nonzero(items_cos_similarity, axis=1)
        # zero_rows_index = np.where(np.all(items_cos_similarity == 0, axis=1))[0]

        # print("Number of non-zero elements in each row:", non_zero_counts)
    

        # num_bins=1000
        # # Step 2: 使用 KMeans 将 items 分成 num_bins 类
        # kmeans = KMeans(n_clusters=num_bins, random_state=42)
        # kmeans.fit(ui_matrix.T)
        # # 获取每一类的中心 items
        # centers = kmeans.cluster_centers_
        # labels = kmeans.labels_
        # # Step 3: 保留每个 item 关于中心 items 的相似度
        # # 获取中心 items 在原矩阵中的索引
        # center_indices = [np.where(labels == i)[0][0] for i in range(num_bins)]
        # # 创建一个矩阵来存储处理后的相似度
        # filtered_similarity = np.zeros_like(items_cos_similarity)
        # for i in range(items_cos_similarity.shape[0]):
        #     filtered_similarity[i, center_indices] = items_cos_similarity[i, center_indices]

        # print("N*K:",filtered_similarity[0])
        # print("N*K:",filtered_similarity[1])

        # # # # 1. 使用K-means进行聚类
        # kmeans = KMeans(n_clusters=1000, random_state=42)
        # kmeans.fit(ui_matrix.T)
        # # 2. 获取聚类中心
        # centroids = kmeans.cluster_centers_
        # center_index = np.unique(kmeans.labels_)
        # # 3. 计算聚类中心之间的相似度
        # matrix = cosine_similarity(centroids)
        # # 4. 创建最终的相似度矩阵：根据item所在的聚类计算聚类中心之间的相似度
        # item_cluster = kmeans.predict(ui_matrix.T)
        # # 使用矩阵操作代替循环
        # final_matrix = matrix[item_cluster][:, item_cluster]
        # # 将zero_rows_index的行列置为0
        # final_matrix[zero_rows_index, :] = 0
        # final_matrix[:, zero_rows_index] = 0
        # mask = np.zeros((final_matrix.shape[1]))
        # mask[center_index] = 1
        # final_matrix = final_matrix * mask 



        # 如果利用点积距离呢
        # 计算物品之间的点积距离
        items_dot_product = np.dot(ui_matrix.T, ui_matrix)
        # 计算每个物品中1的和
        items_sum = np.sum(ui_matrix, axis=0)
        # 避免除以0的错误，将0替换为1（假设这些物品没有用户交互）
        items_sum[items_sum == 0] = 1
        # 归一化
        items_dot_product_normalized = items_dot_product / items_sum[:, np.newaxis]
        # print(items_dot_product_normalized.shape)

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
        dataset = DiffTrainDataset(self.train,self.max_len,self.CLOZE_MASK_TOKEN,self.item_count,self.mask_prob,self.items_cos_similarity,self.rng)
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
            dataset = DiffEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN,self.item_count)
        else :
            answers = self.test
            val = self.val
            dataset = DiffTestDataset(self.train, answers,val, self.max_len, self.CLOZE_MASK_TOKEN,self.item_count)

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
    def __init__(self, u2seq, max_len,  mask_token,num_items,mask_prob,items_cos_similarity,rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_token = mask_token
        self.num_items = num_items
        self.mask_prob =mask_prob
        self.items_cos_similarity = items_cos_similarity
        self.rng = rng


    def __len__(self):
        return len(self.users)


    def get_positive_sample(self,tokens,num_positive=8):

        """
        根据tokens序列重新采样生成对应的正样本
            Args:
                tokens (torch.tensor(S)) :  采样的原始tokens
                num_positive (int) : 正样本数量
            Return:
                pos_tokens (torch.tensor (K*S)) : 生成的正样本tokens,对应的是不同的序列
                pos_labels (torch.tensor(K*S)) : 生成的正样本labels
        """
        pos_tokens_list=[]
        pos_labels_list= []
        for i in range(num_positive):
            pos_tokens = tokens.clone()
            pos_labels = tokens.clone()

            # 类似于bert，这里需要对序列做一个随机mask，也就是cloze 任务
            for index in range(len(pos_tokens)):
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob
                    if prob < 0.8:
                        pos_tokens[index]=self.mask_token
                    elif prob < 0.9:
                        pos_tokens[index]=self.rng.randint(1, self.num_items)
                    else:
                        pos_tokens[index]=pos_tokens[index]
                else:
                    pos_labels[index]=0

            pos_tokens_list.append(pos_tokens)
            pos_labels_list.append(pos_labels)

            # 将pos_tokens_list里的tensor堆叠在一起
            pos_tokens = torch.stack(pos_tokens_list)
            pos_labels = torch.stack(pos_labels_list)

        return pos_tokens,pos_labels


    def __getitem__(self, index):
        """
        从dataset中获得某个样本
            Args :
                index : 样本编号 
            return :
                tokens (torch.tensor (S)): 随机MASK后的序列,因为在最前面拼接了全局的特殊token,所以这里+1 
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

        pos_tokens_list,pos_labels_list = self.get_positive_sample(tokens)
        #因为要cloze mask，所以先把目标label取出来
        # 生成随机整数索引
        random_indice = torch.randint(high=self.max_len,size=())
        # random_indice = self.max_len-1
        target = tokens[random_indice].clone()

        # # 直接生成一个正态分布的采样做对比
        # target_logits =torch.randn(self.num_items + 1)
        # target_logits[target.item()]=1

        # 模拟target的logits，这里用相似度简单模拟
        target_logits = torch.from_numpy(self.items_cos_similarity[target]).clone().to(torch.float32)

        # target_logits = F.normalize(target_logits, p=2, dim=-1)
        top_values, top_indices = torch.topk(target_logits, 10,  largest=True, sorted=True)
        # print(top_values)
        # 光用相似度不行，因为存在一些序列中的item跟目标的相似度很低，但是这些item有可能跟target一起出现的几率很大，所以这里提升一下seq中出现的item的logits，这里用索引距离作为权重就好
        for indice,token in enumerate(tokens):
            scale_factor = 1.5  # 缩放因子
            offset = 1  # 偏移量
            ratio = scale_factor/(abs(indice-random_indice)+1) + offset
            target_logits[token.item()] *= ratio
        target_logits=torch.clamp(target_logits, min=0, max=1)

        # 把target mask掉
        # target_logits [target.item()]=0

        # target_logits = F.normalize(target_logits, p=2, dim=-1)
        # print(target_logits)
        # # 模拟target的logits，这里用seqs简单模拟
        # # 创建大小为 item_num + 1 的张量，并初始化为 -1
        # target_logits = torch.full((self.num_items + 1,), -1, dtype=torch.float32)
        # # 遍历 seqs 并在张量中设置对应位置的值
        # for idx, token in enumerate(tokens):
        #     if token <= self.num_items + 1:
        #         target_logits[token] = (idx+1)/self.max_len

        
        # torch.bernoulli 利用伯努利分布生成01分布，用于生成target_logits，这里待定，可以先不要
        retain_target_mask = torch.bernoulli(target_logits)
        # retain_target_mask = (target_logits > 0.5).float()
        # target_logits= target_logits * retain_target_mask
        target_logits= retain_target_mask
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
        # 采样一些负样本
        negative_target=torch.from_numpy(np.array(np.random.choice(np.setdiff1d(np.arange(self.num_items+1), self._getseq(user)), size=100, replace=False)))
        # mask 目标的值，这里用mask token就行
        tokens[random_indice] = self.mask_token
        # 为了确保random_indice的label不为0，所以这里需要重新赋值
        labels[random_indice] = target

        assert len(tokens)==self.max_len

        return tokens, torch.LongTensor(target), negative_target, random_indice, labels, target_logits,pos_tokens_list,pos_labels_list
    
    def _getseq(self, user):
        return self.u2seq[user]



class DiffEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token,item_count):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
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

        assert len(tokens)==self.max_len

        return tokens, torch.from_numpy(answer), mask_indice

class DiffTestDataset(data_utils.Dataset):
    def __init__(self, u2seq,u2eval, u2answer, max_len, mask_token,item_count):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.u2eval = u2eval
        self.max_len = max_len
        self.mask_token = mask_token
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

        assert len(tokens)==self.max_len
        return tokens, torch.from_numpy(answer), mask_indice
