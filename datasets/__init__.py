from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from .beauty import BeautyDataset
from .kuaishou import KUAISHOUDataset
DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset,
    BeautyDataset.code() : BeautyDataset,
    KUAISHOUDataset.code() :KUAISHOUDataset,
}


def dataset_factory(args):
    
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
