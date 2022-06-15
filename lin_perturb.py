
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
    "banknote-authentication": [4,3,3],
    "blood-transfusion": [4,3,2],
    "breast-cancer": [3,2,1],
    "cylinder-bands": [3,2,0],
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
diff_list  = []
diff_listR = np.array([['blood-transfusion', 0.9],
 ['blood-transfusion', 0.6166666666666667],
 ['cylinder-bands', 0.5714285714285714 ],
 ['banknote-authentication', 0.6072727272727273],
 ['banknote-authentication', 0.15454545454545454],
 ['breast-cancer', 0.9343065693430657]])
diff_listR = diff_listR[:,1].astype(float)
acc_listR  = [0.9, 0.625, 0.5714285714285714, 0.8072727272727273, 0.33636363636363636, 1.0]
accs_list  = []
GROOT_list = []

for  name,X,y in load_all():

    for i in range(3):
            
        if name in datasets:
            
                epsilon = epsilons[name][i]
                depth = depths[name][i] 
                
# =============================================================================
#                 if depth < 3:
#                     continue
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


                N_perturb = 20
                
                rows,cols = np.shape(X)
    
    
                y_conc = np.tile(y,N_perturb+1)
    
                for j in range(1,N_perturb+1):
    
                    lineps      = np.linspace(-epsilon,epsilon,j)
                    
                    X_perturb = X
                    y_corr      = y_conc[:rows + rows*j]
                     
                    for eps in lineps:
                        X_eps = X + eps 
                        X_perturb  = np.concatenate((X_perturb,X_eps),axis = 0)   
                        
                   
                    
                    print(j % 10, end = "")
                    if j % 10 == 0:
                        print(" ", end = "")
                        
                    cart = boom(max_depth = 4 , random_state = 1)
                    if j ==0:
                        cart.fit(X_perturb,y_corr)
                    else:
                        cart.fit(X_perturb,y_corr)
                        
                    if cart.n_classes_ == 1:
                        continue
                    
                    C       = Model.from_sklearn(cart)
                    acc     = C.accuracy(X_test,y_test)
                    adv_acc = C.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=epsilon,options = {"disable_progress_bar" : True})
                    
                    adv_acc_list.append( adv_acc )
                    acc_list.append(acc)
                    
                
                if cart.n_classes_ == 1:
                        continue
                
            
                slope_list.append([name,linregress(pd.Series( np.arange(len(adv_acc_list))),adv_acc_list).slope])
                end_list.append(adv_acc_list[-1])
                begin_list.append(adv_acc_list[0])
    
                sns.regplot(x = pd.Series( np.arange(len(adv_acc_list)),name = "# Perturbations")
                            ,y = adv_acc_list , scatter_kws={'s':1}
                            ,label = name+  " e = " + str(epsilon)  + " d = " + str(depth) 
                            , scatter = True,ci = 80  , line_kws={'linewidth':2}
                            )
                

                accs_list.append(acc_list)
                GROOT_list.append([M.accuracy(X_test,y_test),M.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=epsilon,options = {"disable_progress_bar" : True})])
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.title("Adversial Accuracy score linear perturbations")
plt.ylim([0,1])
plt.show()

slope_mean = np.mean(np.array(slope_list)[:,1].astype(float))
incr       = np.array(end_list) - np.array(begin_list)
print()
print(f"Slope mean = {slope_mean}")
print(f"Average increase: {np.mean( incr )}")

#print(f"ROCT - perturbed CART {np.mean(diff_listR-end_list)}")

end_list_lin_perturb = [0.7258064516129032,
 0.7755102040816326,
 0.6410256410256411,
 0.9,
 0.6333333333333333,
 0.9375,
 0.375,
 0.4666666666666667,
 0.3888888888888889,
 0.538961038961039,
 0.4796747967479675,
 0.6262626262626263,
 0.6619718309859155,
 0.26785714285714285,
 0.4,
 0.7018181818181818,
 0.2545454545454545,
 0.8905109489051095,
 0.8181818181818182,
 0.5,
 0.67,
 0.675,
 0.40865384615384615]