# DRLforMKP
**A Deep Reinforcement Learning-Based Scheme for Solving Multiple Knapsack Problems

This project shows the official codes that used in A Deep Reinforcement Learning-Based Scheme for Solving Multiple Knapsack Problems

Appl. Sci. 2022, 12(6), 3068; https://doi.org/10.3390/app12063068



![image](https://user-images.githubusercontent.com/69515626/199708217-af268d7a-d9eb-4502-979b-0aa87880aca7.png)
<Figure in the paper>

-Create item and knapsack instances

  python RI.py 1000 50 3 10 80

  python LI.py 1000 50 3 10 10

  python QI.py 1000 50 1 10 20

  args : # of episode | # of items | # of knapsack 
  | maximum size (v,w) of item | size of the knapsack

-Train and test  (in here, the train file should be hard coded in a3c mode)

  python train.py 1000 0.0001 50 1000 5 0.9999999 0 0 
  
  python test.py 1000 0.0001 50 1 5 0.9999999
  
  args : # of episode learning rate  | # of items | # of repeat the episode 

  |mode op 0 -a2c 1 a3c |  load_op 0 new 1 start from certain model
                                                                      
-Comparison algorithm

  python random_sol.py 1000 50 1
  
  python ffh.py 1000 50 1
  
  args : # of episode |  # of items  |  # number of knapsack (gamma)


  python gurobi_op_mul.py
  
  To run gurobi, you need a license and install as follows (https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-)


-test example
(base) PS C:\Users\Giwon Sur\Documents\DRLforMKP> python test.py 1000 0.0001 50 1000 3 0.9999999
please write test_File name to open: (Source item)ep_1000_item_50_knap_3_R_10_R2_80_data.pickle_221103_12_34
please write pt file name to open: a3c_train_1103_18_51_item_50_knap_3_epi_1000_rank_0_epi_999_act.pt

I will delete the redundant part ASAP, but the code works well in here.
**
