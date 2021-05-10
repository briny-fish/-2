import torch
import torch.nn.functional as F
import transformers
from transformers import BertTokenizer, BertForNextSentencePrediction,BertModel
from transformers import AdamW
import pickle
import re
import pandas as pd
#import Metrics as mc
import torch.nn as nn
#import Loss
import dataloader
import random
import numpy as np
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')
bertmodel = BertModel.from_pretrained('../bert-base-uncased', return_dict=True)

class clsModel(nn.Module):
    def __init__(self,global_feature_dim):
        super(clsModel,self).__init__()
        self.bert = bertmodel
        #self.rnn = nn.GRU(input_size=768, hidden_size=768, batch_first=True, bidirectional=True)
        self.cls0 = nn.Linear(768+global_feature_dim, 768+global_feature_dim)
        self.cls00 = nn.Linear(768+global_feature_dim,3)
        self.cls1 = nn.Linear(global_feature_dim,global_feature_dim)
        self.cls2 = nn.Linear(global_feature_dim,3)
        self.soft = nn.Softmax(dim=-1)
    def forward(self,input_ids,attention_mask,token_type_ids,global_feature):
        bertout = self.bert(input_ids,attention_mask,token_type_ids)['pooler_output']
        bertout = self.cls0(torch.cat((bertout,global_feature),dim=-1))
        bertout = self.cls00(F.sigmoid(bertout))
        modelout = self.soft(bertout.view(-1,3))
        #out = F.sigmoid(self.cls1(global_feature))
        #lossout = self.cls2(out)
        #modelout = self.soft(lossout)
        return bertout.view(-1,3),modelout#返回bertout用来作为loss的输入，因为Crossentropyloss带了softmax，返回modelout作为模型预测的结果用来计算模型的acc
