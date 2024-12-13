from .base import AbstractDataset

import pandas as pd

from datetime import date


class YOOCHOOSEDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'yoochoose'

    @classmethod
    def url(cls):
        return '0'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['ratings.dat']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep=':', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']

        return df
