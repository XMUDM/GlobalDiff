# _*_ encoding = utf-8 _*_
# @Author  : lmx
# @Time     : 2024/5/6

from .utils import *
from config import RAW_DATASET_ROOT_FOLDER
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle


class AbstractDataset(metaclass=ABCMeta):
    
    """
        DataSet抽象类
            Attributes:
                args: 超参数
                min_rating: 最小评分
                min_uc: 最小用户数
                min_sc: 最小商品数
                split: 划分方式

    """
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split
        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    def load_dataset(self):
        """
        加载数据集并预处理
        """
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))


        #统计数据集情况
        num_unique_uids = df['uid'].nunique()
        # 统计sid的unique数量
        num_unique_sids = df['sid'].nunique()
        # 统计每个uid的交互数量并计算交互的平均长度
        avg_interaction_length = df.groupby('uid').size().mean()
        # 统计一共有多少个交互
        total_interactions = len(df)
        # 计算数据集的稀疏度 (Sparsity)
        sparsity = 1 - (total_interactions / (num_unique_uids * num_unique_sids))
        # 输出结果
        print(f"Unique UID数量: {num_unique_uids}")
        print(f"Unique SID数量: {num_unique_sids}")
        print(f"交互的平均长度: {avg_interaction_length}")
        print(f"总交互数量: {total_interactions}")
        print(f"数据集稀疏度: {sparsity:.6f}") 
        print("processed item count :"+ str(max( df['sid'])))
        from Utils.Utils4GenDataset import GenDataset4SASrecFromDF
        # result_df = df.groupby('uid').apply(lambda x: x.sort_values(by='timestamp').iloc[:-5]).reset_index(drop=True)
        # print(f"Unique UID数量: {result_df['uid'].nunique()}")
        # print(f"Unique SID数量: {result_df['sid'].nunique()}")
        GenDataset4SASrecFromDF(df,len(umap))
        
        # from Utils.Utils4GenDataset import GenDataset_Gru4Rec_from_dataframe
        # GenDataset_Gru4Rec_from_dataframe(df,len(umap),self.args.max_len)
        from Utils.Utils4GenDataset import GenDataset_RecBole_from_dataframe
        GenDataset_RecBole_from_dataframe(df,len(umap),self.args.max_len)

        print("processed item count :"+ str(max( df['sid'])))
        print("maxsid:",max(list(smap.values())))
        print("minsid:",min(list(smap.values())))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw file doesn't exist. Downloading...")
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    def make_implicit(self, df):
        print('Turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating]
        # return df[['uid', 'sid', 'timestamp']]
        return df

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df

    def densify_index(self, df):
        print('Densifying index')

        umap = {u: i for i, u in enumerate(sorted(set(df['uid'])))}
        # 从1开始算sid
        smap = {s: i+1 for i, s in enumerate(sorted(set(df['sid'])))}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        
        return df, umap, smap


    def get_TrainUserAndTestUser(self):
        print('-'*18)
        print(os.getcwd()) # 获取当前工作目录路径
        train_file = os.getcwd()+'/datasets/train.txt'
        test_file = os.getcwd()+'/datasets/test.txt'
        train_uid=[]
        test_uid=[]

        # 打开训练集文件,遍历每一行。
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    #删除换行符
                    l = l.strip('\n').split(' ')
                    #记录 uid
                    uid = int(l[0])
                    #记录uid set
                    train_uid.append(uid)
        # 打开训练集文件,遍历每一行。
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    #删除换行符
                    l = l.strip('\n').split(' ')
                    #记录 uid
                    uid = int(l[0])
                    #记录uid set
                    test_uid.append(uid)
        return train_uid,test_uid

    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = user2items[user]           
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

