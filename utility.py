import os
import copy
import math
import numpy as np
import pandas as pd
from collections import deque
import torch.nn as nn
import torch



def calculate_hit(sorted_list,topk,true_items,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                # total_reward[i] += rewards[j]
                # if rewards[j] == r_click:
                #     hit_click[i] += 1.0
                #     ndcg_click[i] += 1.0 / np.log2(rank + 1)
                # else:
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += (1.0 / math.log2(rank + 1))/(1.0 / math.log2(2))
                





