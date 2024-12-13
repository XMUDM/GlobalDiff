from .base import AbstractDataset

import pandas as pd

from datetime import date


class BeautyDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'beauty'

    @classmethod
    def url(cls):
        return 'https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['ratings_Beauty.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings_Beauty.csv')
        df = pd.read_csv(file_path,header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


