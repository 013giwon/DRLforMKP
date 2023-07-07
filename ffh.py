# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:13:17 2021

@author: User
"""



import numpy as np
import matplotlib.pyplot as plt
import random
import collections 
from torch.distributions import Categorical
import sys
import pickle 
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
date = datetime.now().strftime('%m%d_%H_%M')
# num_episodes = 3
# print('sys.argv 길이 : ', len(sys.argv))                                             
 
 
# for arg in sys.argv: 
#     print('arg value = ', arg) 


#num_of_st = pow(2,num_of_ac);



rList= []
sList = []

  
def rargmax(vector):
    
    m = np.amax(vector)
    indices = np.nonzero(vector == m )[0]
    return random.choice(indices)

def main(max_episodes, N, kc):   

    max_episodes = int(max_episodes)
  
    
    N = int(N)

    kc = int(kc)

    name = input("please write file name to open: (source item)")
    
    with open(name,'rb') as f:
        data = pickle.load(f)

    overall_item_value = np.array(data.get('value'))
    overall_item_weight = np.array(data.get('weight'))
    overall_knap_capa = np.array(data.get('knapsack'))     
 
    

    last_state = []

    rList = []
    idx = []

    i = 0

    t1 = time()
    #Q = np.zeros([num_of_st, num_of_ac])
    for i in range(max_episodes): 

        step_count = 0

        selected = np.zeros((1,  N))
        capa = np.asarray(overall_knap_capa[i].copy()).reshape(1,overall_knap_capa[i].copy().size)
        desorder = np.asarray(np.argsort(-overall_item_value[i]/overall_item_weight[i])).reshape(1,overall_item_value[i].copy().size)
        if kc > 1:
            randknap = np.random.randint(0, int(kc), size=(1,int(N)))
        else:
            randknap = np.zeros((1,N)).astype(int)
        for j in range(N):
            k_desorder =  np.argsort(-capa)
            for k in range(kc):

                if capa[0,k_desorder[0,k]] >= overall_item_weight[i,desorder[0,j]]:
                    capa[0,k_desorder[0,k]] = capa[0,k_desorder[0,k]] - overall_item_weight[i,desorder[0,j]]
                    selected[0,desorder[0,j]] = 1
                    break
           
        last_state.append(selected) 

    # plt.figure()
    # plt.plot(np.array(rList)/np.array(sList)) 
    # plt.show()
    t2 = time()

    computime = t2-t1
    data = {

        'last_state' : last_state,
        'compute_time':computime
    }
    with open('ffh_mul_end_%s_item_%d_knap_%d_data.pickle_'%(date,  N, kc), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)    
   # fla = 0
    # knap_play(mainDQN , max_episodes,fla, step_max)
    # fla = 1
    # knap_play(mainDQN , max_episodes,fla, step_max)
         
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])
 #   main(50,7,5,7)
