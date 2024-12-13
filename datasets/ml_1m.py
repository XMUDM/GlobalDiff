# _*_ encoding : utf-8 _*_
# @Author : lmx
# @Time : 2024/5/6
from .base import AbstractDataset
import pandas as pd
from datetime import date


class ML1MDataset(AbstractDataset):
    """
        导入ml1m数据集
    """
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    def load_ratings_df(self):
        """
        加载ratings.dat文件
            
            returns:
                a pandas.DataFrame including rating information
                columns:
                    uid: user id
                    sid: item id
                    rating: rating score
                    timestamp: timestamp
        """
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


