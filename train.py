# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 12:07:53 2022

@author: gwsur
"""


import train_env as env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from datetime import datetime
import sys
import numpy as np
import pickle
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

date = datetime.now().strftime('%m%d_%H_%M')
# Hyperparameters
n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
max_episodes = sys.argv[1]
learning_rate = sys.argv[2]
N = sys.argv[3]
mul = sys.argv[4]
kc = sys.argv[5]
gamma = sys.argv[6]
mode_op = sys.argv[7]
load_op = sys.argv[8]
train_file = sys.argv[9]
train_pt = sys.argv[10]
max_episodes = int(max_episodes)
learning_rate = float(learning_rate)   
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N = int(N)
mul = int(mul)
kc = int(kc)
gamma = float(gamma)
mode_op = int(mode_op)
load_op = int(load_op)
input_size = N*2 + 3 + kc
num_of_ac = N

class ActorCritic(nn.Module):
    def __init__(self,input_size, output_size, itemsize, knapsize):
        super(ActorCritic, self).__init__()
        embedsize = 512
        kernelsize = 2
        stride = 2
        cvnout = (embedsize - kernelsize)/stride + 1
        cvnout = int(cvnout)
        # if torch.cuda.is_available():
        #     self.Embedding = nn.Linear(input_size, embedsize*2).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)
        #     self.conv = nn.Conv1d(1, 1, 1, 1).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)        
        #     self.hidden_1 = nn.Linear(embedsize*2, embedsize*2).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)
        #     self.hidden_2 = nn.Linear(embedsize*2, embedsize*2).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)
        #     self.Embedding_pi = nn.Linear(embedsize*2, output_size).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)
        #     self.Embedding_v = nn.Linear(embedsize*2, 1).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)
        # else:
        self.Embedding = nn.Linear(input_size, embedsize, dtype=torch.float64)
        # self.conv = nn.Conv1d(1, 1, 1, 1, dtype=torch.float64)  
        self.conv = nn.Conv1d(1,1,kernelsize,stride,dtype=torch.float64 )
        self.hidden_1 = nn.Linear(cvnout, embedsize, dtype=torch.float64)
        # self.hidden_2 = nn.Linear(embedsize, embedsize, dtype=torch.float64)
        self.Embedding_pi = nn.Linear(embedsize, output_size, dtype=torch.float64)
        self.Embedding_v = nn.Linear(embedsize, 1, dtype=torch.float64)
        
        self.itemsize = itemsize
        self.knapsize = knapsize
        self._initialize_weights( -0.08,  0.08)
    def _initialize_weights(self, init_min = -0.08, init_max = 0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def pi(self, x, device, softmax_dim=1):
 
        
        data = x.clone().detach()

        data = data.to(device)
        # data_conv = self.conv(data)
        embeded = self.Embedding(data)
        conved = self.conv(embeded)
        embeded_out = torch.relu(self.hidden_1(conved))

        u = torch.sigmoid(self.Embedding_pi(embeded_out).squeeze(1))

        return u          
    
    def v(self, x, device, softmax_dim=1):
 
        
        data = x.clone().detach()

        data = data.to(device)

        
        embeded = self.Embedding(data)
        conved = self.conv(embeded)
        embeded_out = torch.relu(self.hidden_1(conved))
        # embeded_out = torch.relu(self.hidden_2(embeded_out))
        u = self.Embedding_v(embeded_out).squeeze(1)
        # u = u + e6
        return u  
    def pi_cpu(self, x, softmax_dim=1):
 
        
        data = x.clone().detach()


        embeded = self.Embedding(data)
        conved = self.conv(embeded)
        embeded_out = torch.relu(self.hidden_1(conved))
        # embeded_out = torch.relu(self.hidden_2(embeded_out))
        u = torch.sigmoid(self.Embedding_pi(embeded_out).squeeze(1))

        return u          
    
    def v_cpu(self, x,  softmax_dim=1):
 
        
        data = x.clone().detach()

        
        embeded = self.Embedding(data)
        conved = self.conv(embeded)
        embeded_out = torch.relu(self.hidden_1(conved))

        u = self.Embedding_v(embeded_out).squeeze(1)
        # u = u + e6
        return u 

def train(global_model, rank):

    date = datetime.now().strftime('%m%d_%H_%M')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_of_ac = N
 
    rList = []
    sList = []
    lList = []
    envs = env.ENV(N,kc, train_file)
    local_model = ActorCritic(input_size, num_of_ac, N, kc)
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.SGD(global_model.parameters(), lr=learning_rate)
    each_peri_value_list = []
    total_value_list = []
    for n_epi in range(max_episodes*mul):
        
        total_reward = 0.0;  
        total_reward_list = []
        # selected_list = []
        total_loss = []
        done = False
        p = n_epi%max_episodes
        s = envs.reset(p)
        step_count  = 0 
        s_lst, a_lst, r_lst = [], [], []
        while not done:
            
            for j in range(update_interval):
                s2 = np.expand_dims(s, 1)
                prob = local_model.pi_cpu(torch.from_numpy(s2).double())
                a = Categorical(prob).sample().numpy()
                s_prime, r, done = envs.step(a)

                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                total_reward +=r
                total_reward_list.append(r)
                # selected_list.append(envs.selected*envs.item_val)
                s = s_prime
                step_count +=1
                if done:
                    break
                    
                # s_final = torch.tensor(s_prime, dtype=torch.float)
            s_final = np.expand_dims(s_prime, 1)
            R = 0.0 if done else local_model.v_cpu(torch.from_numpy(s_final).double()).detach().clone().numpy()
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                td_target_lst.append([R])
            td_target_lst.reverse()
            td_target = torch.tensor(td_target_lst)
            td_target = td_target.reshape(-1)            
            s_vec = torch.tensor(s_lst).reshape(-1, input_size).double()  # input_size == Dimension of state

            a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1)

            advantage = td_target - local_model.v_cpu(s_vec.unsqueeze(1)).reshape(-1)
            pi = local_model.pi_cpu(s_vec.unsqueeze(1),  softmax_dim=1)
            pi_a = pi.gather(1, a_vec).reshape(-1)
            loss = -torch.log(pi_a) * advantage.detach() + \
                F.mse_loss(local_model.v_cpu(s_vec.unsqueeze(1)).reshape(-1), td_target.double())
            total_loss.append(sum(loss.detach().numpy().reshape(-1)))
            optimizer.zero_grad()
            loss.mean().backward()
            # local_model = local_model.to("cpu")
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())

            s_lst, a_lst, r_lst = [], [], []
        lList.append(sum(total_loss))
        rList.append(sum(total_reward))
        sList.append(step_count)   

        if (n_epi+1)%(max_episodes) == 0 :
            torch.save(global_model.state_dict(),  './Pt/a%dc_train_%s_item_%d_knap_%d_epi_%d_rank_%d_epi_%d_act.pt'%(mode_op+2,date,  global_model.itemsize, global_model.knapsize, max_episodes, int(rank) ,int(n_epi)))#'cfg.model_dir = ./Pt/'     
            data = {
                'lList': lList,
                'rList': rList,
                'sList': sList,
                'learning_rate': learning_rate,
        
        
            }         
        
            with open('train_%s_item_%d_knap_%d_epi_%d_data.pickle_'%(date,  global_model.itemsize, global_model.knapsize, int(n_epi)), 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    global_model  = ActorCritic(input_size, num_of_ac, N, kc)
    if load_op == 1:
        #train_pt = 'a3c_512_cpu_step_20end_0129_01_35_item_50_knap_3_epi_1000_rank_1_epi_19999_act.pt'
        global_model.load_state_dict(torch.load('./Pt/' + train_pt))

    processes = []
    # use just call train in a2c
    if mode_op == 0: #a2c
      train(global_model,0)

    else: #a3c
        global_model.share_memory()
        for rank in range(n_train_processes):
                p =  mp.Process(target=train, args=(global_model, rank,))       
                p.start()
                processes.append(p)
        for p in processes:
            p.join()