
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

from scipy.stats import uniform 

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



sns.set_style("darkgrid")
sns.color_palette("colorblind")

slope_list = []
end_list   = []
begin_list = []
for  name,X,y in load_all():
    
    if name in datasets:
        
        for i in range(3):
            
            epsilon = epsilons[name][i]
            depth = depths[name][i] 
            
            if depth == 0 : 
                depth =1 
            
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
            
    # =============================================================================
    #             
    #             ROCT = OptimalRobustTree(max_depth=depth, attack_model=attack_model, time_limit = 1)
    #             ROCT.fit(X,y)
    #             
    # =============================================================================
            acc_list     = [] 
            adv_acc_list = []



            N_perturb = 20
            rows,cols = np.shape(X)

            eps    = np.random.uniform(low = -epsilon, high = epsilon, size=(rows*N_perturb,cols))
            X_perturb  = np.tile(X,(N_perturb,1)) + eps
            X_perturb = np.concatenate((X, X_perturb), axis=0)
            

            y_conc = np.tile(y,N_perturb+1)
            
            
            for j in range(N_perturb):

               
                X_corr      = X_perturb[:rows + rows*j]
                y_corr      = y_conc[:rows + rows*j]
                
                print(j % 10, end = "")
                if j % 10 == 0:
                    print(" ", end = "")
                    
                cart = boom(max_depth = depth)
                cart.fit(X_corr,y_corr)
                
                if cart.n_classes_ == 1:
                    continue
                
                C       = Model.from_sklearn(cart)
                acc     = C.accuracy(X_test,y_test)
                adv_acc = C.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=epsilon,options = {"disable_progress_bar" : True})
                
                
                m = Model.from_sklearn(cart)
                adv_acc = m.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=epsilon,options = {"disable_progress_bar" : True})
               
                adv_acc_list.append( adv_acc )
            
            print()
            
            if cart.n_classes_ == 1:
                    continue


            slope_list.append([name,linregress(pd.Series( np.arange(len(adv_acc_list))),adv_acc_list).slope])
            end_list.append(adv_acc_list[-1])
            begin_list.append(adv_acc_list[0])
            sns.regplot(x = pd.Series( np.arange(len(adv_acc_list)),name = "# Perturbations")
                        ,y = adv_acc_list , scatter_kws={'s':1}
                        ,label = name+  " e = " + str(epsilon)  + " d = " + str(depth)
                        , scatter = True, ci = None  , line_kws={'linewidth':2})
        
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.title("Adversial Accuracy score uniformly sampled perturbations")
plt.show()

slope_mean = np.mean(np.array(slope_list)[:,1].astype(float))
incr       = np.array(end_list) - np.array(begin_list)
print()
print(f"Slope mean = {slope_mean}")
print(f"Average increase: {np.mean( incr )}")

end_list_uniform = [0.7419354838709677,
 0.7346938775510204,
 0.5897435897435898,
 0.8933333333333333,
 0.5583333333333333,
 0.9375,
 0.19642857142857142,
 0.4444444444444444,
 0.16666666666666666,
 0.6883116883116883,
 0.7723577235772358,
 0.5656565656565656,
 0.8169014084507042,
 0.4642857142857143,
 0.5111111111111111,
 0.6181818181818182,
 0.3,
 0.8832116788321168,
 0.8,
 0.8181818181818182,
 0.6838461538461539,
 0.7067307692307693,
 0.4675480769230769]
