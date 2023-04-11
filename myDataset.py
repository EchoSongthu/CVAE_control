#!/usr/bin/env python
# coding=utf-8
import os
import torch
import setproctitle
setproctitle.setproctitle("cvae@s")
import numpy as np
import json
import pickle
import torch
import pdb
import os
from utils import merge,identify_home_

class myDataset_mobile(object):
    def __init__(self, split:str, control:str):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grid = 180
        self.split_data_dir = '/data/zmy/dataset/mobile/split_label'
        # self.split_data_dir = '/data/zmy/dataset/tencent/split_label'
        self.max_pos = 168
        self.label_type_num = 1
        self.seq_len = self.max_pos + self.label_type_num + 4
        self.control = control
        self.read_split_data(split)

    def read_split_data(self,split:str):
        path = os.path.join(self.split_data_dir, split+'.pkl')
        self.split_data = pickle.load(open(path, 'rb'))
        print(split+" data loaded...")

    def __getitem__(self, id):
        item = np.zeros((self.max_pos, 3), dtype=int) # (169,3)
        idx = 0
        traj_data = self.split_data[id]['traj'] # mobile
        revenue, gender, edu, age = self.split_data[id]['profile'] # mobile
        # traj_data = self.split_data[id] # tencent
        
        '''traj信息'''
        for j in range(len(traj_data)): # 天数*24
            for i in range(24):
                if idx < self.max_pos + 1:
                    x = traj_data[j][i][0]
                    y = traj_data[j][i][1]
                    item[idx][0] = x * self.max_grid +y +1 # grid id
                    item[idx][1] = i + 1 # hour
                    item[idx][2] = traj_data[j][i][2] +1 # y # week
                    idx += 1
                else:
                    break
        
        '''home实验'''
        traj = item[:,0,...].copy() # traj (168,)

        label = {}
        if self.control == 'home':
            traj_temp = merge(traj)
            home = identify_home_(traj_temp)
            label['home'] = home

        elif self.control == 'revenue':
            if revenue<40:
                revenue_label = 1 # 32392
            elif revenue<80:
                revenue_label = 2
            elif revenue<110:
                revenue_label = 3
            else:
                revenue_label = 4
            revenue_label += 32391
            label['revenue'] = revenue_label

        elif self.control == 'edu':
            if edu.startswith('初中'):
                edu_label = 1
            elif edu.startswith('高中'):
                edu_label = 2
            elif edu.startswith('本科'):
                edu_label = 3
            else:
                edu_label = 4
            edu_label += 32391
            label['edu'] = edu_label

        elif self.control == 'age':
            if age<30:
                age_label = 1
            elif age<40:
                age_label = 2
            elif age<60:
                age_label = 3
            else:
                age_label = 4
            age_label += 32391
            label['age'] = age_label

        elif self.control == 'gender':
            if int(gender)==1:
                gender_label = 1
            else:
                gender_label = 2
            gender_label += 32391
            label['gender'] = gender_label

        labels = np.zeros((self.label_type_num,), dtype=int) # (1,)
        labels[-1] = label[self.control]
        token = np.array([32411], dtype=int)
        start_token = np.array([32412,32412], dtype=int)

        x = np.concatenate([start_token,labels,token], axis=0)
        y = np.concatenate([start_token,labels,token,traj,token], axis=0)
        input = np.concatenate([token,start_token,labels,token,traj], axis=0)

        # mask
        x_mask = torch.full((self.label_type_num+1+2, ), 1).to(self.device)
        y_mask = torch.full((self.seq_len, ), 1).to(self.device)
        x = torch.tensor(x).to(self.device)
        y = torch.tensor(y).to(self.device)
        input = torch.tensor(input).to(self.device)

        dict = {"x_mask":x_mask,
                "x_tokens":x,
                "y_mask":y_mask,
                "y_tokens":y,
                "input_tokens":input,
                "target_tokens":y,
                "mask":y_mask}
        return dict


    def __len__(self):
        return len(self.split_data)


class myDataset_age(object):
    def __init__(self, split:str):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grid = 180
        self.split_data_dir = '/data/zmy/dataset/chinamobile/split_label'
        # self.split_data_dir = '/data/zmy/dataset/tencent/split'
        self.max_pos = 168
        self.label_type_num = 5
        self.seq_len = self.max_pos + self.label_type_num + 2
        
        self.read_split_data(split)

    def read_split_data(self,split:str):
        path = os.path.join(self.split_data_dir, split+'.pkl')
        self.split_data = pickle.load(open(path, 'rb'))
        print(split+" data loaded...")

    def __getitem__(self, id):
        item = np.zeros((self.max_pos+1, 3), dtype=int) # (169,3)
        idx = 1 # 0为start token

        traj_data = self.split_data[id]['traj'] # mobile
        revenue, gender, edu, age = self.split_data[id]['profile'] # mobile
        # traj_data = self.split_data[id] # tencent
        
        '''traj信息'''
        for j in range(len(traj_data)): # 天数*24
            for i in range(24):
                if idx < self.max_pos + 1:
                    x = traj_data[j][i][0]
                    y = traj_data[j][i][1]
                    item[idx][0] = x * self.max_grid +y +1 # grid id
                    item[idx][1] = i +1 # hour
                    item[idx][2] = traj_data[j][i][2] +1 # y # week
                    idx += 1
                else:
                    break
        
        '''home实验'''
        # this is for home control
        traj = item[:,0,...].copy() # traj (169,)
        traj_temp = merge(traj[1:])
        home = identify_home_(traj_temp)

        labels = np.zeros((self.label_type_num,), dtype=int) # (5,)
        pdb.set_trace()
        labels[-1] = home # home


        # mask
        mask_long = torch.full((self.seq_len, ), 1).to(self.device)
        mask_short = torch.full((self.label_type_num+1, ), 1).to(self.device)




        traj = np.append(traj,int(home)) # 带label的traj (170,)
        labels = traj.copy()
        labels[1:self.max_pos+1] = -100
        
        traj = torch.tensor(traj)
        traj = traj.to(self.device)
        labels = torch.tensor(labels)
        labels = labels.to(self.device)
        '''
        # profile信息
        

        if revenue<50:
            revenue_label = 1
        elif revenue<100:
            revenue_label = 2
        elif revenue<150:
            revenue_label = 3
        elif revenue<200:
            revenue_label = 4
        else:
            revenue_label = 5

        if edu.startswith('初中'):
            edu_label = 1
        elif edu.startswith('高中'):
            edu_label = 2
        elif edu.startswith('本科'):
            edu_label = 3
        elif edu.startswith('研究生'):
            edu_label = 4
        else:
            raise ValueError("edu error")

        if age<20:
            age_label = 1
        elif age<30:
            age_label = 2
        elif age<40:
            age_label = 3
        elif age<50:
            age_label = 4
        elif age<60:
            age_label = 5
        else:
            age_label = 6
        '''

        # return {"data":item,
        #         "length":len(traj_data),
        #         "age_label":age_label,
        #         "revenue_label":revenue_label,
        #         "gender_label":int(gender),
        #         "edu_label":edu_label}
        return {"input_ids":traj,
                "labels":labels}

    def __len__(self):
        return len(self.split_data)