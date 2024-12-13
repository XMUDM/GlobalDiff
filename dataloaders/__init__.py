from datasets import dataset_factory
from .diff import DiffDataloader
from .cbit import CBITDataloader
from .bert import BertDataloader
from .plugdiff import PlugDiffDataloader

DATALOADERS = {
    DiffDataloader.code(): DiffDataloader,
    CBITDataloader.code(): CBITDataloader,
    BertDataloader.code(): BertDataloader,
    PlugDiffDataloader.code(): PlugDiffDataloader,

}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    print("---After processing---")
    print("user_count :"+ str(dataloader.user_count))
    print("item_count :"+ str(dataloader.item_count))
    
    return train, val, test, dataloader.item_count
