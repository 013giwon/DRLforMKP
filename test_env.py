# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 19:33:57 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:55:48 2021

@author: User
"""


import numpy as np
import math
import random
import pickle
import copy
from datetime import datetime
date = datetime.now().strftime('%m%d_%H_%M')
class ENV:

    #print(item_val[1])
    

    knap_capa = 3;
    learning_rate = 1e-1
#    input_size = a.shape[0]
    output_size = 1
    
    rList= []

    def __init__(self,n,k, file, name="main"):
        self.net_name = name

        self.N = n 
        self.K = k
        # name = input("please write file name to open: (source item)")
        with open(file,'rb') as f:
            data = pickle.load(f)
        self.name = file
        self.overall_item_value = np.array(data.get('value'))
        self.overall_item_weight = np.array(data.get('weight'))
        self.overall_knap_capa = np.array(data.get('knapsack'))     
        self.ovrl_max = max(np.max(self.overall_item_value, axis=-1))
    def build_stat(self,i):

            self.idx = np.asarray(np.arange(self.N), dtype='int64').reshape(-1)
           # self.capa_idx = np.asarray(np.arange(self.K), dtype='int64')
            self.capa_idx = np.asarray(np.arange(self.K), dtype='int64').reshape(-1)

            problems_ratio = self.overall_item_value[i].copy()/self.overall_item_weight[i].copy()

            sort_index = np.argsort(-problems_ratio)            
            self.sort_index = sort_index
            self.sorted_item_val = self.overall_item_value[i,self.sort_index]
            self.sorted_item_weight= self.overall_item_weight[i,self.sort_index]
            self.item_size = self.sorted_item_weight.shape[0] 
            self.num_of_ac = self.item_size
            # self.knap_capa = self.problems[i,2,0]
            self.knap_capa = self.overall_knap_capa[i,self.capa_idx]

    
    def step(self, action):
        state = self.curr_state.copy()
        # state = self.curr_agre_state.copy()
       
        

#        item = action

# 담았던 아이템을 또 담으려 했을 때
        if action > state[0,0] - 1:
            # print('you tried to do again %d, %d'%(state[0,0],item+1))
                return state,  False
        done = False
        #item_idx = np.nonzero(self.idx == self.curr_idx[0,item])[0]
        selected_item = self.curr_idx[action].copy()
        # lefted[item_idx] = 1
        k = 0 #math.floor(action/self.item_size)

        k_idx2 = self.curr_capa_idx[k].copy()

        temp_s = state.copy()

        #selected_item = item_idx2

        
      #  wFlag = False
        # 담으려고 시도한 아이템의 무게가 초과될때
        if state[0,1 + k] < self.sorted_item_weight[selected_item]:
            
           # wFlag = True
            self.sorted_passed[selected_item] = 1
        else:
            self.sorted_selected[selected_item] = 1
            self.curr_capa[k_idx2] = self.curr_capa[k_idx2] - self.sorted_item_weight[selected_item]


        #안되는 케이스 끝나고 되는 케이
        temp_s[0,0] = temp_s[0,0] - 1
            
        

        valid_k = np.argwhere(self.curr_capa > 0).reshape(-1)

        if valid_k.size == 0:
            sidx = np.argwhere(self.sorted_selected == 1).reshape(-1) 
            self.selected[self.sort_index[sidx]] = 1
            return temp_s, True
          #  capa = np.asarray(np.zeros(self.K))
            
        else:
          
            capa_ratio = self.curr_capa[valid_k]
            valid_k_sort =  np.argsort(-capa_ratio)
    

            capa = self.curr_capa[valid_k[valid_k_sort]]
            
            if valid_k.size < self.K:
                pad = np.asarray(np.zeros( self.K- valid_k.size))
                capa = np.append(capa, pad)  
            # capa2 = np.asarray(capa)    

        temp_s[0,1:1 + self.K] = capa

        lefted = self.sorted_selected.copy() + self.sorted_passed.copy()

        unSelected = np.argwhere(lefted == 0).reshape(-1)

        if unSelected.size == 0:

             temp_s[0,1 + self.K:3 + self.K + 2*self.N] = 0
             self.curr_idx = 0
             self.curr_state = temp_s.copy() 
             sidx = np.argwhere(self.sorted_selected == 1).reshape(-1) 
             self.selected[self.sort_index[sidx]] = 1
             return temp_s,  True

        temp_s[0,1 + self.K] = sum(self.sorted_item_val[unSelected])
        temp_s[0,2 + self.K] = sum(self.sorted_item_weight[unSelected])
        


        if max(self.curr_capa) < min(self.sorted_item_weight[unSelected]):
        # if temp_s[0,1] < min(self.item_weight[unSelected[0]]):
            sidx = np.argwhere(self.sorted_selected == 1).reshape(-1) 
            self.selected[self.sort_index[sidx]] = 1
            return temp_s, True         
        
        self.curr_capa_idx = self.capa_idx[valid_k[valid_k_sort]].reshape(-1)   
                                    

        temp_s[0, 3 + self.K : 3 + self.K + 2*self.N] = np.zeros(2*self.N)
        temp_s[0, 3 + self.K : 3 + self.K + 2*unSelected.size:2] = self.sorted_item_val[unSelected]/self.ovrl_max
        temp_s[0, 4 + self.K : 4 + self.K + 2*unSelected.size:2] = self.sorted_item_weight[unSelected]/max(self.curr_capa)

        self.curr_state = temp_s.copy() 
        
        self.curr_idx = unSelected

        return temp_s, done
    
    def reset(self,i):
        self.build_stat(i)

        normal_capa = np.asarray(self.knap_capa.copy())

        arr1 = np.zeros(2*self.N+3 + self.K)
        arr1[0]= self.item_size
        arr1[1:self.K + 1] = normal_capa
        arr1[self.K + 1] = sum(self.sorted_item_val)
        arr1[self.K + 2] = sum(self.sorted_item_weight)
        arr1[self.K + 3:self.K + 3+2*self.N:2] = self.sorted_item_val/self.ovrl_max
        arr1[self.K + 4:self.K + 4+2*self.N:2] = self.sorted_item_weight/max(self.knap_capa)
        arr1 = np.array(arr1)
        arr1 = arr1.reshape(1,arr1.size)  
        self.curr_capa_idx = np.asarray(copy.deepcopy(self.capa_idx))

        self.curr_idx = np.asarray(copy.deepcopy(self.idx))

        self.curr_state = arr1.copy()
        self.sorted_selected = np.zeros(self.item_size)
        self.selected = np.zeros(self.item_size)
        self.sorted_passed = np.zeros(self.item_size)
        self.curr_capa = copy.deepcopy(self.knap_capa)
        
        return arr1

