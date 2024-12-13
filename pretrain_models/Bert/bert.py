from pretrain_models.Bert.base import BaseModel
from pretrain_models.Bert.bert_modules.bert import BERT

import torch
import torch.nn as nn



class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.args=args
        self.bert_hidden_units = args.bert_hidden_units
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert_hidden_units, args.num_items + 1)
    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x):
        c_i = self.bert(x)[-1]
        rec_output = self.out(c_i)
        return rec_output,c_i

# class BERTModel(BaseModel):
#     def __init__(self, args):
#         super().__init__(args)
#         self.bert = BERT(args)
#         self.out_before = nn.Sequential(
#             nn.Linear(self.bert.hidden, self.bert.hidden),
#             nn.ReLU()
#         )
#         self.out = nn.Linear(self.bert.hidden, args.num_items + 1)

#     @classmethod
#     def code(cls):
#         return 'bert'

#     def forward(self, x):
#         x = self.bert(x)
#         # return self.out(x)
#         return self.out(self.out_before(x))


