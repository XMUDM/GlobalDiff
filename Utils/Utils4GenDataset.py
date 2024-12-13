"""
@purpose : 基于bert的数据集处理对齐各个模型的数据集
@author : mxluo
"""

import pandas as pd
from tqdm import tqdm
tqdm.pandas()
def GenDataset4SASrecFromDict(DataDict):
    """
    为SASRec生成数据集
        param:
            DataDict : {uid : [sid1,sid2]}
    """
    print(f'generating dataset for SASRec')
    for uid,sids in tqdm(DataDict.items()):
        for sid in sids:
            with open('./generated_dataset/SASRecDataSet.txt','a') as fin:
                fin.write(str(uid)+' '+str(sid)+'\n')

def GenDataset4SASrecFromDF(df,user_count):
    """
    为SASRec生成数据集
        param:
            df : dataframe
            user_count : uid数量
    """
    user_group = df.groupby('uid')
    user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
    print("processed item count :"+ str(max( df['sid'])))
    data = {}
    for user in range(user_count):
        items = user2items[user]           
        data[user]= items
    print(max(max(sid_list) for sid_list in data.values()))
    GenDataset4SASrecFromDict(data)


def GenDataset_Gru4Rec_from_dataframe(df,user_count,maxlen):
    """
    为Gru4Rec生成数据集
        param:
            df : dataframe [uid,sid,rating,timestamp]
            user_count : uid数量
            maxlen : 最大序列长度,因为我这里用到的知识序列里面最后的maxlen长度,所以这里要保持一致,只将最后的序列分别划分为train,val,test
    """
    print(f'generating dataset for Gru4Rec...')
    train_dataframe = df.groupby('uid').apply(lambda x: x.sort_values(by='timestamp').iloc[:-2].tail(maxlen)).reset_index(drop=True)
    val_dataframe = df.groupby('uid').apply(lambda x: x.sort_values(by='timestamp').iloc[:-1].tail(maxlen)).reset_index(drop=True)
    test_dataframe = df.groupby('uid').apply(lambda x: x.sort_values(by='timestamp').tail(maxlen)).reset_index(drop=True)

    # 保存至csv
    train_dataframe.to_csv('./generated_dataset/train_dataframe_gru4rec.csv',index=False,sep='\t')
    val_dataframe.to_csv('./generated_dataset/val_dataframe_gru4rec.csv',index=False,sep='\t')
    test_dataframe.to_csv('./generated_dataset/test_dataframe_gru4rec.csv',index=False,sep='\t')


def GenDataset_RecBole_from_dataframe(df,user_count,maxlen):
    """
    为RecBole生成数据集
        param:
            df : dataframe [uid,sid,rating,timestamp]
            user_count : uid数量
            maxlen : 最大序列长度,因为我这里用到的知识序列里面最后的maxlen长度,所以这里要保持一致,只将最后的序列分别划分为train,val,test,并且val,test使用的留一法划分的,因此对于保留的长度应该是maxlen+2
    """
    print(f'generating dataset for Gru4Rec...')
    inter_dataframe = df.groupby('uid').apply(lambda x: x.sort_values(by='timestamp').tail(maxlen+2)).reset_index(drop=True)
    #重命名列
    inter_dataframe.rename(columns={'uid': 'user_id', 'sid': 'item_id', 'rating': 'rating', 'timestamp': 'timestamp'}, inplace=True)
    # 调整顺序
    inter_dataframe = inter_dataframe[['user_id', 'item_id', 'rating', 'timestamp']]
    output_file = "./generated_dataset/beauty.inter"
    # 定义文件头
    header = "user_id:token\titem_id:token\trating:float\ttimestamp:float"
    with open(output_file, 'w') as f:
        # 写入头
        f.write(header + "\n")
        # 将 DataFrame 保存为制表符分隔的文件
        inter_dataframe.to_csv(f, sep='\t', index=False, header=False)