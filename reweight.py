import numpy as np


from roct.milp import OptimalRobustTree

from groot.toolbox import Model
from groot.datasets import load_all,load_breast_cancer,load_cylinder_bands,load_banknote_authentication

from sklearn.tree import DecisionTreeClassifier as boom
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from scipy.stats import linregress

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

p_list = []
for p in range(0,51,1):
    
    def perturb_CART_adversarially(X,y,depth):
    
        cart = boom(max_depth = depth, max_leaf_nodes = 2**depth)
        cart.fit(X,y)
        m = Model.from_sklearn(cart)
    
        X_adv = ((1-m.attack_distance(X,y,options = {"disable_progress_bar" : True}))**p).reshape(-1,1)*X

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
    
    for i in range(3):
    
        for  name,X,y in load_all():
            
            if name in datasets:
        
                epsilon = epsilons[name][i]
                depth = depths[name][i] 
                
                if depth == 0: 
                    depth = 1
                
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
                
                X_adv = X
                j = 0 
                while True:
                    j += 1
                    
                    print(j % 10, end = "")
                    if j % 10 == 0:
                        print(" ", end = "")
                    
                    C,X_adv  = perturb_CART_adversarially(X_adv,y,depth)
                    
                    if C is False or j > 20:
                        break
    
                    adv_acc = C.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=epsilon,options = {"disable_progress_bar" : True})

                    adv_acc_list.append( adv_acc )
        
                slope_list.append([name,linregress(pd.Series( np.arange(len(adv_acc_list))),adv_acc_list).slope])
                end_list.append(adv_acc_list[-1])
                begin_list.append(adv_acc_list[0])
                
                sns.regplot(x = pd.Series( np.arange(len(adv_acc_list)),name = "# Perturbations")
                            ,y = adv_acc_list , scatter_kws={'s':1}
                            ,label = name+  " e = " + str(epsilon)  + " d = " + str(depth)
                            , scatter = True, ci = None  , line_kws={'linewidth':2},
                            )
        
            
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

    
    slope_mean = np.mean(np.array(slope_list)[:,1].astype(float))
    incr       = np.array(end_list) - np.array(begin_list)
    print()
    print(f"Slope mean = {slope_mean}")
    print(f"Average increase: {np.mean( incr )}")
    
    plt.title("Adversial Accuracy score " + str(p) + " " + str(np.mean(incr)))
    plt.show()
    
    p_list.append(np.mean(incr))
    
sns.regplot(x = pd.Series( np.arange(len(p_list)),name = "# Perturbations")
            ,y = p_list , scatter_kws={'s':1}
            ,label = name+  " e = " + str(epsilon)  + " d = " + str(depth)
            , scatter = True  , line_kws={'linewidth':2},
            )
plt.title("Mean increase in adversarial accuracy score N_perturb = 20")
plt.ylim(0,0.4)
plt.show()

