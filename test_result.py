# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 18:02:49 2021

@author: User
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import pdb
import sys
result = []


string_list = ['ffh_mul','random_sol_knap','gurobi_op_sol_knap','test']
folder = r'./' 
repeat_axis = int(sys.argv[1])
ran_ratio = np.zeros((1,repeat_axis))
ffh_ratio = np.zeros((1,repeat_axis))
gurobi_ratio = np.zeros((1,repeat_axis))
proposed_ratio = np.zeros((1,repeat_axis))
for k in range(repeat_axis):
    name = input("please write file name to open in {}: ".format(k+1))
    # name = 'ep_1000_item_50_knap_3_R_10_R2_80_data.pickle_221103_12_34'

    with open(name,'rb') as f:
        data = pickle.load(f)

    overall_item_value = np.array(data.get('value'))
    overall_item_weight = np.array(data.get('weight'))
    overall_knap_capa = np.array(data.get('knapsack'))
    item_size = overall_item_value.shape[1]
    epi_len = overall_item_value.shape[0]
    pdb.set_trace()
    for a in range(4):

        print("/"+ string_list[a]+"*item_"+str(item_size)+ "_knap_" + str( overall_knap_capa.shape[1]) + "_data.pickle*")
        file_list = glob.glob(f"{folder}/"+ string_list[a]+"*item_"+str(item_size)+ "_knap_" + str(overall_knap_capa.shape[1]) + "_data.pickle*")


        if k == 0:
         result.append(file_list[0])
        else:
            result[a] = (file_list[0])
    
    epi_ratio_sum = np.zeros((4,epi_len))
    last_state = np.zeros((4,item_size))
    result_data = []
    # result = input("please write file name to open: ")
    for j in range(4):
        with open(result[j],'rb') as f:
            result_data = pickle.load(f)
         #overall_knap_capa.shape[0]  
        
        t_state = np.array(result_data.get('last_state'))
        t_state = np.squeeze(t_state)
        isItem_map = np.zeros((epi_len,item_size))
        isItem_unmap = np.zeros((epi_len,item_size))
        epi_sum = np.zeros((1,epi_len))
        
        for i in range(epi_len):
        
            isItem_map[i,:] = np.add(isItem_map[i,:], t_state[i])
            unSelected = np.nonzero(isItem_map[i] == 0 )[0]
            isItem_unmap[i,unSelected] = 1
             # print(unSelected)
            epi_sum[0,i] = sum(np.squeeze(np.multiply(isItem_map[i].reshape(1, len(isItem_map[i])), overall_item_value[i].reshape(1,len(overall_item_value[i])))))
            epi_ratio_sum[j,i] = epi_sum[0,i]/sum(overall_item_value[i])
            # print(np.multiply(isItem_map[i].reshape(1, len(isItem_map[i])), overall_item_value[i].reshape(1,len(overall_item_value[i]))))        
    x = np.arange(repeat_axis)
    x = x*1 + 1
    print(epi_ratio_sum[1].mean())
    ran_ratio[0,k] = epi_ratio_sum[1].mean()
    ffh_ratio[0,k] = epi_ratio_sum[0].mean()
    gurobi_ratio[0,k] = epi_ratio_sum[2].mean()
    proposed_ratio[0,k] = epi_ratio_sum[3].mean()
# plt.bar(x, np.squeeze(epi_sum), color='blue')
# plt.bar(x,np.squeeze(epi_sum2), color='red')
print("random", ran_ratio.mean())
print("greedy", ffh_ratio.mean())
print("gurobi", gurobi_ratio.mean())
print("proposed",proposed_ratio.mean())
plt.plot(x,np.squeeze(ran_ratio), color='red', marker= 'o', label='random')
plt.plot(x,np.squeeze(ffh_ratio), color='orange',  marker= 'o',label='ffh')
plt.plot(x,np.squeeze(gurobi_ratio), color='blue',  marker= 'o',label='gurobi')
plt.plot(x,np.squeeze(proposed_ratio), color='green',  marker= 'o',label='proposed')
plt.title(str(item_size)+' items RI')
plt.xlabel('Number of knapsacks')
plt.ylabel('Averate map ratio of values')
plt.ylim(0.0,1)
plt.xlim(0.5,repeat_axis+ 0.5)
plt.xticks(np.arange(0, repeat_axis+1, step=1))
plt.legend(loc=0)
# plt.savefig('50_RI.eps', format='eps')
plt.savefig(str(item_size)+'_RI.png', format='png')
# plt.xticks(x, years)

plt.show