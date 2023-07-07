# DRLforMKP
**A Deep Reinforcement Learning-Based Scheme for Solving Multiple Knapsack Problems

This project shows the official codes that used in A Deep Reinforcement Learning-Based Scheme for Solving Multiple Knapsack Problems

Appl. Sci. 2022, 12(6), 3068; https://doi.org/10.3390/app12063068



![image](https://user-images.githubusercontent.com/69515626/199708217-af268d7a-d9eb-4502-979b-0aa87880aca7.png)
<Figure in the paper>

I used it in spyder IDE, and the scripts are as follow
creating item and knapsack instances


python RI.py 1000 50 3 10 80
python LI.py 1000 50 3 10 10
python QI.py 1000 50 1 10 20

args : # of episode # of items # of knapsack maximum size (v,w) of item size of the knapsack
train and test  (in here, the train file should be hard coded in a3c mode)

python train.py 1000 0.0001 50 1000 5 0.9999999 6 4 0
python test.py 1000 0.0001 50 1 5 0.9999999 6 4

args : # of episode learning rate  # of items # of repeat the episode  ( reward op state option)  load_op
                                                                        ------------------------
comparison algorithm
To run, gurobi, you need a license
python random_algorithm.py 1000 0.001 50 1 1 0.99

python gurobi_op_mul.py

python ffh_mul.py 1000 0.001 50 1 1 0.99
# of episode (learning rate) # of items (#?) # number of knapsack (gamma)
I will delete the redundant part ASAP, but the code works well in here.
**
