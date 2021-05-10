import torch
import torch.nn.functional as F
import transformers
from transformers import BertTokenizer, BertForNextSentencePrediction,BertModel
from transformers import AdamW
import pickle
from torch.optim import lr_scheduler
import re
import dataloader
import model
import pandas as pd
#import Metrics as mc
import torch.nn as nn
#import Loss
import numpy as np
import random
from tqdm import tqdm
torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')
bertmodel = BertModel.from_pretrained('../bert-base-uncased', return_dict=True)
def _convert_to_transformer_inputs(question, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy,
                                       truncation=True
                                       )

        input_ids = inputs["input_ids"]
        # print(tokenizer.decode(input_ids))
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids_q, input_masks_q, input_segments_q = return_id(
        question, 'longest_first', max_sequence_length)

    return [input_ids_q, input_masks_q, input_segments_q]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    for instance in tqdm(df[columns]):
        if(instance != instance):
            q = ''
        else:
            q = instance
        #print(q)
        ids_q, masks_q, segments_q = \
            _convert_to_transformer_inputs(q, tokenizer, max_sequence_length)

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32),
            np.asarray(input_masks_q, dtype=np.int32),
            np.asarray(input_segments_q, dtype=np.int32)]

def compute_output_arrays(df, columns):
    tmp = df[columns]
    tmp = list(tmp)
    tmpdict = {'small':0,'fit':1,'large':2}
    return np.array([tmpdict[x] for x in tmp])

#评论文本token化
input_categories = 'review_text'
MAX_SEQUENCE_LENGTH = 200
trainDatas = dataloader.readTrain()
train_inputs = compute_input_arrays(trainDatas, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
train_outputs = compute_output_arrays(trainDatas,'fit')


train_global_feature =[list(row) for index,row in trainDatas[['age','body type','bust size','category','height','rating','rented for','size','weight']].iterrows()]


max_age = 0.0
bust_dict = {}
body_type = {}
category = {}
max_height = 0.0
max_rating = 0.0
rent_for = {}
max_size = 0.0
max_weight = 0.0

#获取数据中不同columns 的最值以及类别数量
for i in range(len(train_global_feature)):
    train_global_feature[i] = [x if x==x else -1 for x in train_global_feature[i]]
    max_age = max(max_age,train_global_feature[i][0])
    if(train_global_feature[i][1] not in body_type.keys()):
        body_type[train_global_feature[i][1]] = 1
    else:
        body_type[train_global_feature[i][1]] += 1

    if (train_global_feature[i][2] not in bust_dict.keys()):
        bust_dict[train_global_feature[i][2]] = 1
    else:
        bust_dict[train_global_feature[i][2]] += 1

    if (train_global_feature[i][3] not in category.keys()):
        category[train_global_feature[i][3]] = 1
    else:
        category[train_global_feature[i][3]] += 1

    if(train_global_feature[i][4]!=-1):

        tmp = train_global_feature[i][4].split("' ")
        train_global_feature[i][4] = (float(tmp[0]) * 12 + float(tmp[1][:-1])) * 2.54
        max_height = max(max_height,train_global_feature[i][4])

    max_rating = max(max_rating, train_global_feature[i][5])

    if (train_global_feature[i][6] not in rent_for.keys()):
        rent_for[train_global_feature[i][6]] = 1
    else:
        rent_for[train_global_feature[i][6]] += 1

    max_size = max(max_size, train_global_feature[i][7])
    if(train_global_feature[i][8]!=-1):
        train_global_feature[i][8] = float(train_global_feature[i][8][:-3])

    max_weight = max(max_weight, train_global_feature[i][8])

train_inputs_gf = []
#利用上面得到的数据进行向量化（数值归一化以及类别的向量化）
for i in range(len(train_global_feature)):
    tmp = []
    if(train_global_feature[i][0] != -1):
        train_global_feature[i][0] = float(train_global_feature[i][0]) / max_age
    else:
        train_global_feature[i][0] = -1.0
    if train_global_feature[i][2] not in body_type.keys():
        train_global_feature[i][2] = -1
    train_global_feature[i][2] = list(body_type.keys()).index(train_global_feature[i][2])

    if train_global_feature[i][3] not in category.keys():
        train_global_feature[i][3] = -1
    train_global_feature[i][3] = list(category.keys()).index(train_global_feature[i][3])

    if (train_global_feature[i][4] != -1):
        train_global_feature[i][4] = float(train_global_feature[i][4]) / max_height
    else:
        train_global_feature[i][4] = -1.0

    if (train_global_feature[i][5] != -1):
        train_global_feature[i][5] = float(train_global_feature[i][5]) / max_rating
    else:
        train_global_feature[i][5] = -1.0

    if (train_global_feature[i][6] not in rent_for.keys()):
        train_global_feature[i][6] = -1
    train_global_feature[i][6] = list(rent_for.keys()).index(train_global_feature[i][6])

    if (train_global_feature[i][7] != -1):
        train_global_feature[i][7] = float(train_global_feature[i][7]) / max_size
    else:
        train_global_feature[i][7] = -1.0

    if (train_global_feature[i][8] != -1):
        train_global_feature[i][8] = float(train_global_feature[i][8]) / max_weight
    else:
        train_global_feature[i][8] = -1.0

    tmp.append(train_global_feature[i][0])
    tmp.append(train_global_feature[i][4])
    tmp.append(train_global_feature[i][5])
    tmp.append(train_global_feature[i][7])
    tmp.append(train_global_feature[i][8])
    #类别的向量化，变成onehot形式
    tmp.extend([1.0 if x == train_global_feature[i][2] else 0.0 for x in range(len(list(body_type.keys())))])
    tmp.extend([1.0 if x == train_global_feature[i][3] else 0.0 for x in range(len(list(category.keys())))])
    tmp.extend([1.0 if x == train_global_feature[i][5] else 0.0 for x in range(len(list(rent_for.keys())))])
    train_inputs_gf.append(tmp)
train_inputs_gf = np.array(train_inputs_gf)
print(len(train_inputs_gf[0]))
print(train_inputs_gf[:10])
#模型初始化
model = model.clsModel(89)
#model.load_state_dict(torch.load('/data0/data_ti4_c/zongwz/gov/parameter.pkl'))
model = model.to(device)
learning_rate = 5e-4
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
#scheduler = lr_scheduler.StepLR(optim, step_size=2, gamma=0.1)

#训练
def run_model(epoch_num, train_batch_size, test_batch_size):
    losstrain_list = []
    losstest_list = []
    for epoch in range(epoch_num):
        print('epoch:{}'.format(epoch))
        # print(train_inputs[0])
        idx = list(range(int(len(train_inputs[0])*0.7)))
        #idx = list(range(3500))
        idx_test = range(idx[-1],len(train_inputs[0]))
        #idx_test = list(range(idx[-1],5000))
        acc,loss = one_epoch( 0, train_inputs,train_inputs_gf, train_outputs, train_batch_size,idx)
        losstrain_list.append(loss)
        
        testacc,testloss = one_epoch( 1, train_inputs, train_inputs_gf,train_outputs, test_batch_size,idx_test)
        losstest_list.append(testloss)
        print('trainacc:%s,trainloss:%s'%(str(acc),str(loss)))
        print('testacc:%s,testloss:%s'%(str(testacc),str(testloss)))
        if(epoch>3 and testloss>losstest_list[epoch-2] and losstest_list[epoch-1]>losstest_list[epoch-2]):
            break
        #scheduler.step()





#每一轮的训练过程，state=0表示train，否则evaluation
def one_epoch( state, inputs,gf, labels, batch_size,idx):
    true_num = 0.0
    loss_rec = 0.0
    totnum = 0.0
    if state == 0:
        model.train()
    else:
        model.eval()
    loss = ''
    # print(inputs[:10])
    # print(inputs[0])
    model_output = ''
    for cnt in tqdm(range(0, len(idx), batch_size)):
        # if cnt>400:break

        # print(cnt)
        if (cnt + batch_size < len(idx)):
            tmpidx = idx[cnt:cnt+batch_size]
        else:
            break

        reviews_simples = [[inputs[i][id] for id in tmpidx] for i in range(3)]
        reviews_simples = torch.LongTensor(reviews_simples).to(device)
        gf_simples = gf[tmpidx]
        # print(idx)
        # print(labels)
        tmplabels = labels[tmpidx]
        # print(labels)
        # print(tmp)
        tmplabels = torch.tensor(tmplabels).to(device)
        loss = ''

        if state == 0:

            loss_output,model_output = model(input_ids=reviews_simples[0], attention_mask=reviews_simples[1],
                                              token_type_ids=reviews_simples[2],global_feature=torch.FloatTensor(gf_simples).to(device))
            # print(loss_output.shape)
            # print(tmplabels.view(-1).shape)
            #print(tmplabels.view(-1))
            loss = criterion(loss_output, tmplabels.view(-1))
            loss_rec+=float(loss)
            # print(loss)
            loss.backward()
            optim.step()
            optim.zero_grad()
        else:
            with torch.no_grad():
                loss_output, model_output = model(input_ids=reviews_simples[0], attention_mask=reviews_simples[1],
                                                  token_type_ids=reviews_simples[2], global_feature=torch.FloatTensor(gf_simples).to(device))
                # print(loss_output.shape)
                loss = criterion(loss_output, tmplabels.view(-1))
                loss_rec += float(loss)
                # print(loss_output)

        # print(model_output)
        pred = torch.argmax(model_output, dim=-1).squeeze()
        #print(pred)

        target = tmplabels.view(-1)
        #print(target)
        true_num += float(torch.sum(pred == target))
        totnum += float(batch_size)
    if state==0:
        torch.save(model.state_dict(), r'/home/zongwz/ass2/baseparameter5e4seq.pkl')
    return true_num/totnum,loss_rec/totnum
    

run_model(40,24,24)









