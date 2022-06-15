# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:05:50 2022

@author: Gert
"""
import numpy as np

from roct.milp import OptimalRobustTree

from groot.toolbox import Model
from groot.datasets import load_all,load_breast_cancer,load_cylinder_bands,load_banknote_authentication

from sklearn.tree import DecisionTreeClassifier as boom
from sklearn import *
from sklearn.preprocessing import MinMaxScaler

import copy as copy

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from scipy.stats import linregress
from groot.model import GrootTreeClassifier
datasets = set((
    "banknote-authentication",
    "blood-transfusion",
    "breast-cancer",
    "cylinder-bands",
    "diabetes",
    "haberman",
    "ionosphere",
    "wine",
))
epsilons = {
    "banknote-authentication": [0.07,0.09,0.11],
    "blood-transfusion": [0.01,0.02,0.03],
    "breast-cancer": [0.28,0.39,0.45],
    "cylinder-bands": [0.23,0.28,0.45],
    "diabetes": [0.05,0.07,0.09],
    "haberman": [0.02,0.03,0.05],
    "ionosphere": [0.2,0.28,0.36],
    "wine": [0.02,0.03,0.04],
}

depths =  { 
    "banknote-authentication": [4,3,3] ,
    "blood-transfusion": [3,3,3],
    "breast-cancer": [3,2,1],
    "cylinder-bands": [2,2,2],
    "diabetes": [0,0,0],
    "haberman": [0,0,0],
    "ionosphere": [2,2,1],
    "wine": [1,0,2],
}

def perturb_CART_adversarially(X,y,depth):

    cart = boom(max_depth = depth, max_leaf_nodes = 2**depth)
    cart.fit(X,y)
    m = Model.from_sklearn(cart)
    X_adv = m.adversarial_examples(X,y,options = {"disable_progress_bar" : True})

    if np.isnan(X_adv).any():
        print( " Couldnt find adv. examples")
        return False,X_adv
    

        
    else: 
        cart_adv = boom(max_depth = depth, max_leaf_nodes = 2**depth)
        cart_adv.fit(X_adv,y)
        return Model.from_sklearn(cart_adv),X_adv


sns.set_style("darkgrid")
sns.color_palette("colorblind")

slope_list = []
end_list   = []
begin_list = []
diff_list  = []
for i in range(3):

    for  name,X,y in load_all():
        
        if name in datasets:
    
            epsilon = epsilons[name][i]
            depth = depths[name][i] 
            if depth == 0 :
                depth = 1
# =============================================================================
#             if depth < 3: 
#                 continue
# =============================================================================
            
            print()
            print(name)
            print(f"(Epsilon, depth): ({epsilon} , {depth})")
            
            scaler = MinMaxScaler()
            
            border_80 = int(np.floor(len(y)*0.8))
            X_test  = X[border_80:]
            X       = X[0:border_80]
            X       = scaler.fit_transform(X)
            X_test  = scaler.transform(X_test)
            y_test  = y[border_80:]
            y       = y[0:border_80]
          
            attack_model = [epsilon]*len(X[1])
            
            acc_list     = [] 
            adv_acc_list = []
            
            tre = GrootTreeClassifier(max_depth=depth, attack_model=attack_model)
            tre.fit(X, y)
            
            M = Model.from_groot(tre)
            X_adv = X
            j = 0 

            while True:
                j += 1
                
                print(j % 10, end = "")
                if j % 10 == 0:
                    print(" ", end = "")
                
# =============================================================================
#                 if len(adv_acc_list) > 0:
#                     prev_adv_acc_X = C.adversarial_accuracy(X, y, attack="tree", epsilon=epsilon,options = {"disable_progress_bar" : True})
#                 
# =============================================================================
                
                C,X_adv  = perturb_CART_adversarially(X_adv,y,depth)
# =============================================================================
#                 if name == "banknote-authentication" and i == 1:
#                     print(X[1])
#                     
#                     print(X_adv[1])
# =============================================================================
                
                if C is False or j >50:
                    break
                
                
                adv_acc = C.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=epsilon,options = {"disable_progress_bar" : True})
# =============================================================================
#                 
#                 
#                 if len(adv_acc_list) > 0:
#                     adv_acc_X = C.adversarial_accuracy(X, y, attack="tree", epsilon=epsilon,options = {"disable_progress_bar" : True})
# 
#                     if adv_acc_X/prev_adv_acc_X < 1: 
#                         
#                         rows,cols = np.shape(X_adv)
#                         eps = np.random.uniform(low = -epsilon, high = epsilon, size=(rows,cols))
#                         X_adv = X_adv + eps
#                         print(" e ", end = "")
#                         continue 
#                         
# =============================================================================
                adv_acc_list.append( adv_acc )
            

            slope_list.append([name,linregress(pd.Series( np.arange(len(adv_acc_list))),adv_acc_list).slope])
            end_list.append(adv_acc_list[-1])
            begin_list.append(adv_acc_list[0])
            diff_list.append([name,M.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=epsilon,options = {"disable_progress_bar" : True}) - np.max(adv_acc_list)])
            
            sns.regplot(x = pd.Series( np.arange(len(adv_acc_list)),name = "# Perturbations")
                        ,y = adv_acc_list , scatter_kws={'s':1}
                        ,label = name+  " e = " + str(epsilon)  + " d = " + str(depth)
                        , scatter = True, ci = None  , line_kws={'linewidth':2},
                        )
    
        
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.title("Adversial Accuracy score")
plt.show()

slope_mean = np.mean(np.array(slope_list)[:,1].astype(float))
incr       = np.array(end_list) - np.array(begin_list)
print()
print(f"Slope mean = {slope_mean}")
print(f"Average increase: {np.mean( incr )}")

adv_perturb_end_list = [0.7419354838709677,
 0.6933333333333334,
 0.4107142857142857,
 0.6428571428571429,
 0.0,
 0.18545454545454546,
 0.0,
 0.693076923076923,
 0.7419354838709677,
 0.6866666666666666,
 0.03571428571428571,
 0.6428571428571429,
 0.0,
 0.6218181818181818,
 0.0,
 0.693076923076923,
 0.7419354838709677,
 0.68,
 0.0,
 0.6428571428571429,
 0.1267605633802817,
 0.02909090909090909,
 0.0,
 0.6330769230769231]