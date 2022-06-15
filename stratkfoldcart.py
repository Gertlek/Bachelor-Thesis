
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
from sklearn.model_selection import StratifiedKFold


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

for  name,X,y in load_all():
    
    for i in range(3):
        
        best_adv_accuracy = 0
        validation_scores = []
        
        for depth in range(0,5):
            
            total_adv_accuracy = 0
            split_optimality = []
            split_models = []
            skf = StratifiedKFold(n_splits = 3)
            
        if name in datasets:
            
                epsilon = epsilons[name][i]


                scaler = MinMaxScaler()
                
                border_80 = int(np.floor(len(y)*0.8))
                X_test  = X[border_80:]
                X       = X[0:border_80]
                X       = scaler.fit_transform(X)
                X_test  = scaler.transform(X_test)
                y_test  = y[border_80:]
                y       = y[0:border_80]
                
                X_train = np.concatenate((X,X_test),axis = 0)
                y_train = np.concatenate((y,y_test),axis = 0)
                
                for train_index, test_index in skf.split(X_train,y_train):
                    X_train_cv, X_val_cv = X_train[train_index], X_train[test_index]
                    y_train_cv, y_val_cv = y_train[train_index], y_train[test_index]
                
                    N_perturb = 10
                    
                    rows,cols = np.shape(X)
        
                    y_conc = np.tile(y,N_perturb+1)
        
                    for j in range(1,N_perturb+1):
        
                        lineps      = np.linspace(-epsilon,epsilon,j)
                        
                        X_perturb = X
                        y_corr      = y_conc[:rows + rows*j]
                         
                        for eps in lineps:
                            X_eps = X + eps 
                            X_perturb  = np.concatenate((X_perturb,X_eps),axis = 0)   
                       

                            
                        cart = boom(max_depth = depth , random_state = 1)
                        if j ==0:
                            cart.fit(X_perturb,y_corr)
                        else:
                            cart.fit(X_perturb,y_corr)
                    if cart.n_classes_ == 1:
                        continue
                        
                    C = Model.from_sklearn(cart)
                    adv_acc = C.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=epsilon,options = {"disable_progress_bar" : True})
                    total_adv_accuracy += adv_acc
                    
                avg_adv_accuracy = total_adv_accuracy / 3
                validation_scores.append(avg_adv_accuracy)
                
                if avg_adv_accuracy > best_adv_accuracy:
                    best_adv_accuracy = avg_adv_accuracy
                    best_depth = depth
                
        print(name,epsilon,best_depth,best_adv_accuracy)