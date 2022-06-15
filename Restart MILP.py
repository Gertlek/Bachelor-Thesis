import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.preprocessing import MinMaxScaler

import math

from groot.datasets import load_all,load_breast_cancer,load_cylinder_bands,load_banknote_authentication
from groot.adversary import DecisionTreeAdversary
from groot.model import (
    GrootTreeClassifier,
    NumericalNode,
    Node,
    _TREE_LEAF,
    _TREE_UNDEFINED,
)

from groot.toolbox import Model

from roct.base import BaseOptimalRobustTree
from roct.upper_bound import samples_in_range
from roct.milp import OptimalRobustTree

import gurobipy as gp
from gurobipy import GRB


from scipy.spatial.distance import pdist

import time

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
    "blood-transfusion": [4,3,2],
    "breast-cancer": [3,2,1],
    "cylinder-bands": [3,2,0],
    "diabetes": [0,0,0],
    "haberman": [0,0,0],
    "ionosphere": [2,2,1],
    "wine": [1,0,2],
}

def check_features_scaled(X):
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    if np.any(min_values < 0 - 1e-04) or np.any(max_values > 1 + 1e-04):
        print()
        print(min(min_values),max(max_values))
        print("not scaled")
        print()
# =============================================================================
#         warnings.warn(
#             "found feature values outside of the [0, 1] range, "
#             "features should be scaled to [0, 1] or ROCT will ignore their "
#             "values for splitting."
#         )
# 
# 
# =============================================================================

    
def __build_model_gurobi2(self, X, y):

    p = self.n_features_in_
    n = self.n_samples_
    T_B = range(1, (self.T // 2) + 1)
    T_L = range((self.T // 2) + 1, self.T + 1)

    
    model = gp.Model("Optimal_Robust_Tree_Fitting")
    
    a = model.addVars(range(1, p + 1), T_B, vtype=GRB.BINARY, name="a")
    z = model.addVars(
        range(1, n + 1), T_L, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="z"
    )
    s = model.addVars(range(1, n + 1), T_B, range(2), vtype=GRB.BINARY, name="s")
    b = model.addVars(T_B, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="b")
    c = model.addVars(T_L, vtype=GRB.BINARY, name="c")
    e = model.addVars(range(1, n + 1), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="e")
    
    # The objective is to minimize the sum of errors (weighted by 1 each)
    model.setObjective(gp.quicksum(e[i] for i in range(1,n+1)), GRB.MINIMIZE)
    

    # Let the nodes only split on one feature
    for t in T_B:
        model.addConstr(gp.quicksum(a[j, t] for j in range(1, p + 1)) == 1)
    
    epsilon = 1e-08
    self.epsilon_ = epsilon
    M_left = M_right = 2 + epsilon
    for i in range(1, n + 1):
        for m in T_B:
            model.addConstr(
                gp.quicksum(a[j, m] * X[i - 1, j - 1] for j in range(1, p + 1))
                >= epsilon + b[m] - M_left * s[i, m, 0]
            )
            model.addConstr(
                gp.quicksum(a[j, m] * X[i - 1, j - 1] for j in range(1, p + 1))
                <= b[m] + M_right * s[i, m, 1]
            )
    
    for i in range(1, n + 1):
        for t in T_L:
            A_l, A_r = self._OptimalRobustTree__ancestors(t)
    
            model.addConstr(
                gp.quicksum(s[i, m, 0] for m in A_l)
                + gp.quicksum(s[i, m, 1] for m in A_r)
                - len(A_l + A_r)
                + 1
                <= z[i, t]
            )
    
    for i in range(1, n + 1):
        for t in T_L:
            if y[i - 1] == 0:
                model.addConstr(z[i, t] + c[t] - 1 <= e[i])
            else:
                model.addConstr(z[i, t] - c[t] <= e[i])
    

    tolerance = 10 ** (int(math.log10(epsilon)) - 1)
    self.tolerance = tolerance
    return model, (a, z, e, s, b, c), tolerance

def __solve_model_gurobi2(self, model , variables, warm_start, tolerance, restart = False):
   a, z, e, s, b, c = variables
   
   self.variables = variables
   p = self.n_features_in_
   n = self.n_samples_
   T_B = range(1, (self.T // 2) + 1)
   T_L = range((self.T // 2) + 1, self.T + 1)

   if restart :
  
        X_train,y_train,attack_index_list = restart

        curr_e_len = len(e)
        
        e.update( model.addVars(range(curr_e_len + 1, curr_e_len+len(attack_index_list)+1),lb=0, ub=1, vtype=GRB.CONTINUOUS, name="e", obj=1))

        z.update(model.addVars(range(curr_e_len + 1, curr_e_len+len(attack_index_list)+1),T_L,lb=0, ub=1, vtype=GRB.CONTINUOUS, name="z"))

        s.update(model.addVars(range(curr_e_len + 1, curr_e_len+len(attack_index_list)+1),T_B,range(2),vtype=GRB.BINARY, name="s"))

        for  count,attacking_index in enumerate(attack_index_list):

            model.addConstr(e[curr_e_len + count + 1] == e[attacking_index + 1])
       

   if warm_start:
       a_warm, b_warm, c_warm, e_warm, s_warm, z_warm = warm_start

       for j in range(1, p + 1):
           for t in T_B:
               a[j, t].start = a_warm[t - 1][
                   j - 1
               ]  # a_warm's indices are reversed

       for t in T_B:
           b[t].start = b_warm[t - 1]

       for i, t in enumerate(T_L):
           label = c_warm[i]
           c[t].start = label

       for i in range(1, n + 1):
           e[i].start = e_warm[i - 1]

       for key in s_warm:
           i, t, side = key
           s[i, t, side].start = s_warm[key]

       for key in z_warm:
           i, t = key
           z[i, t].start = z_warm[key]


   model.write("model.lp")

   output_flag = 1 if self.verbose else 0
   options = [
       ("OutputFlag", output_flag),
       ("IntFeasTol", tolerance),
       ("MIPGap", tolerance),
       ("Presolve", 2),
       ("MIPFocus", 1),
       ("Cuts", 2),
       ("Method", 0),
   ]
   if self.time_limit:
       options.append(("TimeLimit", self.time_limit))
   if self.cpus:
       options.append(("Threads", self.cpus))
   
   for option in options:
       model.setParam(*option)
   self.Incumbent_counter = 0
   def callback(model, where):
       
        if where == GRB.Callback.MIP:
            if time.time() >= time_cut_off:
                print("Time's up")
                model.terminate()
        if where == GRB.Callback.MIPSOL:

            self.Incumbent_counter += 1
        
            if self.Incumbent_counter % 3 == 0:
                print("Restart", end = " ")
                            
                print(f"{int(model.cbGet(GRB.Callback.MIPSOL_OBJ))}/{len(e)}", end = "|")

                model.terminate()
            if time.time() >= time_cut_off:
                print("Time's up")
                model.terminate()
            
   model.optimize(callback)

   self.model = model
   
   
   objective = sum([e[i+1].X for i in range(self.n_samples_)])/self.n_samples_

   self.objective = objective
   # Create branching nodes with their feature and splitting threshold
   nodes = []
   for t in T_B:
       a_values = [a[j, t].X for j in range(1, p + 1)]

       if 1 in a_values:
           feature = a_values.index(1)

           candidates = self.thresholds[feature]
           i = np.abs(candidates - b[t].X).argmin()

           if math.isclose(b[t].X, candidates[i]):
               if i == len(candidates) - 1:
                   # Prevent out of bounds
                   threshold = candidates[i] + self.epsilon_
               else:
                   threshold = (candidates[i] + candidates[i + 1]) * 0.5
           elif b[t].X < candidates[i]:
               if i == 0:
                   # Prevent out of bounds
                   threshold = candidates[i] - self.epsilon_
               else:
                   threshold = (candidates[i - 1] + candidates[i]) * 0.5
           else:
               if i == len(candidates) - 1:
                   # Prevent out of bounds
                   threshold = candidates[i] + self.epsilon_
               else:
                   threshold = (candidates[i] + candidates[i + 1]) * 0.5
       else:
           # If there is no a_j == 1 then this node is a dummy that should
           # not apply a split. Threshold = 1 enforces this.
           feature = 0
           threshold = 2.0

       if self.verbose:
           print(f"Node: {t} feature: {feature}, threshold: {threshold}")
       node = NumericalNode(
           feature, threshold, _TREE_UNDEFINED, _TREE_UNDEFINED, _TREE_UNDEFINED
       )
       nodes.append(node)

   # Create leaf nodes with their prediction values
   for t in T_L:
       value = np.array([round(1 - c[t].X), round(c[t].X)])
       if self.verbose:
           print(f"Leaf: {t} value: {value}")
       leaf = Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)
       nodes.append(leaf)

   # Hook up the nodes to each other
   for t in T_B:
       node = nodes[t - 1]
       node.left_child = nodes[(t * 2) - 1]
       node.right_child = nodes[t * 2]

   self.root_ = nodes[0]
   self.optimal_ = model.Status == GRB.OPTIMAL
   
def _fit_solver_specific2(self, X, y):
        self.thresholds = [
            self._OptimalRobustTree__determine_thresholds(samples, feature)
            for feature, samples in enumerate(X.T)
        ]

        warm_start = self._OptimalRobustTree__generate_warm_start(X, y, self.warm_start_tree)

        model, variables, tolerance = self.__build_model_gurobi2(X, y)

        self.__solve_model_gurobi2(model, variables, warm_start, tolerance)
        
def fit2(self, X, y):
       """
       Fit the optimal robust decision tree on the given dataset.
       """
       check_features_scaled(X)

       self.n_samples_, self.n_features_in_ = X.shape

       # If no attack model is given then train a regular decision tree
       if self.attack_model is None:
           self.attack_model = [0.0] * X.shape[1]

       self.Delta_l, self.Delta_r = self._BaseOptimalRobustTree__parse_attack_model(self.attack_model)
       self.T = (2 ** (self.max_depth + 1)) - 1

       self._fit_solver_specific2(X, y)


OptimalRobustTree.__build_model_gurobi2 = __build_model_gurobi2
OptimalRobustTree.__solve_model_gurobi2 = __solve_model_gurobi2
OptimalRobustTree._fit_solver_specific2 = _fit_solver_specific2
OptimalRobustTree.fit2                  = fit2

def bfs(node):
    visited = [] # List to keep track of visited nodes.
    queue = []     #Initialize a queue
    
    def bfs_inside(visited, node):
      visited.append(node)
      queue.append(node)
    
      while queue:
        s = queue.pop(0) 
        if s != -1 or -2:
            for neighbour in [s.left_child,s.right_child]:
              if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    bfs_inside(visited,node)           
    return visited

acc_list = [] 
adv_acc_list = [] 
time_list = []
obj_lists = []
for name,X,y in load_all():

    border_80 = int(np.floor(len(y)*0.8))
    X_test  = X[border_80:]
    X       = X[0:border_80]
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
            
    y_test  = y[border_80:]
    y       = y[0:border_80]

    for i in range(3):
 
        if name in datasets:
            
            epsilon = epsilons[name][i]
            depth = depths[name][i] 

            if depth < 3:
                continue

            attack_model = [epsilon]*len(X[1])

            ROCT = OptimalRobustTree(
                max_depth=depth,
                attack_model=None,
                warm_start_tree=None,
                )
            
            X_train = X
            y_train = y
            intersecting_samples_indices = set(np.unique(np.array(samples_in_range(X,y,epsilon,epsilon))) )
  
            t = time.time()
            
            time_border = float("inf")
            iterations = 0
  
            global time_cut_off 
            time_cut_off = t + time_border
            obj_list = []
            
            ROCT.fit2(X_train, y_train)
            m  = Model.from_groot(ROCT)
            obj_list.append([1-ROCT.objective,m.adversarial_accuracy(X_test,y_test,epsilon = epsilon,options = {"disable_progress_bar" : True})])
            print(obj_list[0], end = "\n")
            
            samesame = 0 
            while True:

                m  = Model.from_groot(ROCT)
                
                feasible_attacks = m.attack_feasibility(X,y,epsilon = epsilon,options = {"disable_progress_bar" : True})
# =============================================================================
#   
#                 for j,_ in enumerate(feasible_attacks):
#                     if j in intersecting_samples_indices:
#                         
# =============================================================================

                X_feasible = X[feasible_attacks]
                y_feasible = y[feasible_attacks]
                
                attack_index_list = np.where(feasible_attacks == True)[0]
                
                adversarial_examples = m.adversarial_examples(X_feasible,y_feasible,options = {"disable_progress_bar" : True})

                X_train = np.vstack((X_train,adversarial_examples))
                y_train = np.concatenate((y_train,y_feasible))

                restart = X_train,y_train,attack_index_list
         
                ROCT.__solve_model_gurobi2(ROCT.model, ROCT.variables, warm_start = False, tolerance = ROCT.tolerance, restart = restart)
                iterations += 1
                obj_list.append([1-ROCT.objective,m.adversarial_accuracy(X_test,y_test,epsilon = epsilon,options = {"disable_progress_bar" : True})])
    

                               
                print("  error acc, adv acc: " ,obj_list[-1], end = "\n")
                
                if  obj_list[-2][0] >= 1-ROCT.objective and obj_list[-2][1] >= obj_list[-1][1]:
                   
                    samesame +=1 
                    
                    if samesame >= 2:
                        print("Objective reached, terminate")
                        obj_list.pop(-1)
                        break
                        
                

            obj_lists.append(obj_list) 
            ROCT_normal = OptimalRobustTree(
                max_depth= depth,
                attack_model= attack_model,
                warm_start_tree= None,
                time_limit = 5 #int( time.time()- t )
                )
            
            ROCT_normal.fit(X,y)
            
            m2  = Model.from_groot(ROCT_normal)
            
            adv_acc_list.append([name,epsilon,depth,m.adversarial_accuracy(X_test,y_test,epsilon = epsilon,options = {"disable_progress_bar" : True}),m2.adversarial_accuracy(X_test,y_test,epsilon = epsilon,options = {"disable_progress_bar" : True})])
            acc_list.append([name,epsilon,depth,m.accuracy(X_test,y_test),m2.accuracy(X_test,y_test)])

print(f"Mean time: {np.mean(time_list)}")

print("Restart ROCT, ROCT mean acc: ", np.mean(np.array(acc_list)[:,3].astype(float)) ,np.mean( np.array(acc_list)[:,4].astype(float)))

print("Restart ROCT, ROCT mean adv acc: ", np.mean(np.array(adv_acc_list)[:,3].astype(float)) ,np.mean( np.array(adv_acc_list)[:,4].astype(float)))

obj_lists.pop(2)
fig, axs = plt.subplots(nrows = 3, ncols = 2)
k = 0
for  ar, ax in zip(obj_lists,axs.ravel()):

    if k > len(obj_lists):
        break
    el = np.array(ar)

    ax.plot(range(np.shape(el)[0]),el[:,0],label = str(name_list[k]) + " \n objective" )
    ax.plot(range(np.shape(el)[0]),el[:,1],label = str(name_list[k]) + " \n adv acc" )
    ax.legend(loc = "center left", prop={'size': 6},frameon=False)
    ax.set_xlabel("Iterations")

    k += 1
fig.tight_layout()

plt.show()

obj_lists= [[[0.6454849498327759, 0.82],
  [0.6471571906354515, 0.82],
  [0.81438127090301, 0.76],
  [0.8377926421404682, 0.9066666666666666],
  [0.8444816053511706, 0.9066666666666666],
  [0.9130434782608696, 0.9066666666666666],
  [0.9180602006688963, 0.9066666666666666],
  [0.9481605351170569, 0.9066666666666666],
  [0.959866220735786, 0.8066666666666666],
  [0.9615384615384616, 0.9066666666666666],
  [0.9632107023411371, 0.9066666666666666],
  [0.9665551839464883, 0.9066666666666666],
  [0.9682274247491639, 0.9066666666666666],
  [0.9698996655518395, 0.9066666666666666],
  [0.9715719063545151, 0.9066666666666666],
  [0.774247491638796, 0.9066666666666666],
  [0.7759197324414716, 0.5066666666666667],
  [0.7792642140468228, 0.42],
  [0.7909698996655519, 0.4666666666666667],
  [0.7993311036789298, 0.44],
  [0.8193979933110368, 0.4666666666666667],
  [0.8294314381270903, 0.4666666666666667],
  [0.8377926421404682, 0.4666666666666667],
  [0.8394648829431438, 0.56],
  [0.939799331103679, 0.44],
  [0.959866220735786, 0.6866666666666666],
  [0.9698996655518395, 0.7066666666666667],
  [0.9715719063545151, 0.7266666666666667],
  [0.9732441471571907, 0.7266666666666667],
  [0.9749163879598662, 0.7266666666666667],
  [0.9765886287625418, 0.9066666666666666],
  [0.979933110367893, 0.78],
  [0.9816053511705686, 0.7266666666666667],
  [0.9832775919732442, 0.5466666666666666],
  [0.9866220735785953, 0.7266666666666667],
  [0.9882943143812709, 0.7266666666666667],
  [0.9899665551839465, 0.5533333333333333],
  [0.9765886287625418, 0.7266666666666667],
  [0.9782608695652174, 0.8133333333333334],
  [0.979933110367893, 0.8],
  [0.9816053511705686, 0.8466666666666667],
  [0.9832775919732442, 0.8533333333333334],
  [0.9849498327759197, 0.8133333333333334],
  [0.9882943143812709, 0.8666666666666667],
  [0.9933110367892977, 0.7533333333333333],
  [0.9933110367892977, 0.9066666666666666]],
 [[0.919732441471572, 0.8533333333333334],
  [0.9214046822742474, 0.8533333333333334],
  [0.9280936454849499, 0.8533333333333334],
  [0.931438127090301, 0.8533333333333334],
  [0.9364548494983278, 0.8533333333333334],
  [0.9381270903010034, 0.8533333333333334],
  [0.939799331103679, 0.8533333333333334],
  [0.9414715719063546, 0.8533333333333334],
  [0.9247491638795986, 0.8533333333333334],
  [0.9297658862876255, 0.7466666666666667],
  [0.931438127090301, 0.7466666666666667],
  [0.9331103678929766, 0.7333333333333333],
  [0.9364548494983278, 0.7466666666666667],
  [0.9381270903010034, 0.7466666666666667],
  [0.939799331103679, 0.76],
  [0.9414715719063546, 0.8],
  [0.9431438127090301, 0.8],
  [0.9565217391304348, 0.8133333333333334],
  [0.9581939799331104, 0.4866666666666667],
  [0.959866220735786, 0.4866666666666667]],
 [[1.0, 0.5178571428571429], [1.0, 0.5178571428571429]],
 [[0.7028258887876025, 0.0],
  [0.7037374658158615, 0.0],
  [0.7055606198723792, 0.0],
  [0.8012762078395624, 0.0],
  [0.8067456700091158, 0.18545454545454546],
  [0.8094804010938924, 0.18545454545454546],
  [0.8158614402917046, 0.18545454545454546],
  [0.8113035551504102, 0.18545454545454546],
  [0.918869644484959, 0.4036363636363636],
  [0.9252506836827712, 0.4109090909090909],
  [0.927073837739289, 0.4109090909090909],
  [0.9279854147675478, 0.4109090909090909],
  [0.9288969917958068, 0.39636363636363636],
  [0.9325432999088423, 0.39636363636363636],
  [0.9334548769371012, 0.41818181818181815],
  [0.9343664539653601, 0.4],
  [0.935278030993619, 0.4],
  [0.9361896080218779, 0.39636363636363636],
  [0.9371011850501367, 0.39636363636363636],
  [0.9398359161349134, 0.4],
  [0.9425706472196901, 0.44363636363636366],
  [0.9434822242479489, 0.45454545454545453],
  [0.9443938012762079, 0.45454545454545453],
  [0.9453053783044667, 0.45454545454545453],
  [0.9462169553327257, 0.44363636363636366],
  [0.9471285323609845, 0.45454545454545453],
  [0.9489516864175023, 0.44363636363636366],
  [0.9498632634457611, 0.46545454545454545],
  [0.9516864175022789, 0.45454545454545453],
  [0.9525979945305378, 0.4109090909090909],
  [0.9535095715587967, 0.4690909090909091],
  [0.9544211485870556, 0.5236363636363637],
  [0.9562443026435734, 0.4690909090909091],
  [0.9571558796718322, 0.5781818181818181],
  [0.959890610756609, 0.509090909090909],
  [0.9626253418413856, 0.4727272727272727],
  [0.9635369188696445, 0.4727272727272727],
  [0.9644484958979034, 0.5272727272727272],
  [0.96718322698268, 0.4727272727272727],
  [0.968094804010939, 0.48363636363636364],
  [0.9690063810391978, 0.5381818181818182],
  [0.9735642661804923, 0.5418181818181819],
  [0.9753874202370101, 0.5818181818181818],
  [0.9772105742935278, 0.5745454545454546],
  [0.9781221513217867, 0.5818181818181818],
  [0.9799453053783045, 0.5781818181818181],
  [0.9808568824065633, 0.4763636363636364],
  [0.9826800364630811, 0.4763636363636364],
  [0.98359161349134, 0.5054545454545455],
  [0.98359161349134, 0.5272727272727272],
  [0.9881494986326345, 0.509090909090909],
  [0.9890610756608933, 0.6181818181818182],
  [0.99179580674567, 0.6218181818181818],
  [0.9927073837739289, 0.6181818181818182],
  [0.9936189608021878, 0.5854545454545454],
  [0.9945305378304466, 0.6],
  [0.9954421148587056, 0.5963636363636363],
  [0.9963536918869644, 0.5781818181818181],
  [0.9963536918869644, 0.6072727272727273],
  [0.9963536918869644, 0.6509090909090909]],
 [[0.773017319963537, 0.15272727272727274],
  [0.8140382862351869, 0.15272727272727274],
  [0.8158614402917046, 0.15272727272727274],
  [0.8185961713764813, 0.15272727272727274],
  [0.8195077484047402, 0.14909090909090908],
  [0.821330902461258, 0.14909090909090908],
  [0.8222424794895169, 0.14909090909090908],
  [0.8195077484047402, 0.15272727272727274],
  [0.8204193254329991, 0.33454545454545453],
  [0.8231540565177757, 0.33454545454545453],
  [0.8477666362807658, 0.33454545454545453],
  [0.8514129443938012, 0.6181818181818182],
  [0.8559708295350957, 0.6181818181818182],
  [0.8587055606198724, 0.6181818181818182],
  [0.8632634457611668, 0.6181818181818182],
  [0.8659981768459435, 0.6181818181818182],
  [0.8678213309024613, 0.38545454545454544],
  [0.8732907930720146, 0.4109090909090909],
  [0.8751139471285323, 0.4109090909090909],
  [0.8769371011850502, 0.38545454545454544],
  [0.877848678213309, 0.38545454545454544],
  [0.8787602552415679, 0.4109090909090909],
  [0.8814949863263446, 0.44727272727272727],
  [0.886052871467639, 0.43272727272727274],
  [0.886964448495898, 0.44],
  [0.8979033728350045, 0.44727272727272727],
  [0.9161349134001823, 0.5236363636363637],
  [0.9170464904284412, 0.5236363636363637],
  [0.9179580674567, 0.5236363636363637],
  [0.9179580674567, 0.5236363636363637],
  [0.918869644484959, 0.5236363636363637],
  [0.9197812215132178, 0.5236363636363637],
  [0.9206927985414768, 0.5236363636363637],
  [0.9216043755697356, 0.5527272727272727],
  [0.9225159525979946, 0.5527272727272727]],
 [[0.773017319963537, 0.09818181818181818],
  [0.796718322698268, 0.09818181818181818],
  [0.8058340929808568, 0.09818181818181818],
  [0.8550592525068368, 0.09818181818181818],
  [0.8587055606198724, 0.46545454545454545],
  [0.8659981768459435, 0.4690909090909091],
  [0.8687329079307201, 0.46545454545454545],
  [0.8705560619872379, 0.46545454545454545],
  [0.8732907930720146, 0.46545454545454545],
  [0.8760255241567912, 0.46545454545454545],
  [0.8851412944393802, 0.46545454545454545],
  [0.886052871467639, 0.4690909090909091],
  [0.8887876025524157, 0.4690909090909091],
  [0.8906107566089334, 0.4690909090909091],
  [0.8915223336371924, 0.4690909090909091],
  [0.8924339106654512, 0.48],
  [0.6508659981768459, 0.48],
  [0.6517775752051048, 0.6363636363636364],
  [0.6554238833181404, 0.6363636363636364],
  [0.6599817684594349, 0.6363636363636364],
  [0.6645396536007293, 0.6363636363636364],
  [0.6690975387420237, 0.6363636363636364],
  [0.6700091157702825, 0.6509090909090909],
  [0.845031905195989, 0.6363636363636364],
  [0.845943482224248, 0.12363636363636364],
  [0.8833181403828624, 0.13818181818181818],
  [0.8915223336371924, 0.38181818181818183],
  [0.8933454876937101, 0.3563636363636364],
  [0.894257064721969, 0.3563636363636364],
  [0.8951686417502279, 0.3563636363636364],
  [0.8960802187784868, 0.3563636363636364],
  [0.8969917958067457, 0.3563636363636364],
  [0.8979033728350045, 0.3563636363636364]],
 [[0.8772893772893773, 0.7445255474452555],
  [0.9597069597069597, 0.7445255474452555],
  [0.9926739926739927, 0.021897810218978103],
  [0.9945054945054945, 0.0],
  [0.9981684981684982, 0.014598540145985401],
  [1.0, 0.021897810218978103],
  [1.0, 0.19708029197080293],
  [1.0, 0.21897810218978103],
  [1.0, 0.21897810218978103]]]

obj_lists.pop(2)


lines_acc_advacc = [ [0.9,0.9,0.784280936454849 ], [0.9,0.9,0.7625418060200669] , [0.7672727272727272,0.5781818181818181,0.8951686417502279] ,[ 0.3781818181818182,0.27636363636363637,0.8076572470373746], [ 0.37454545454545457, 0.24727272727272728,0.8076572470373746 ] , [1,0.9343065693430657 ,0.9377289377289377] ]
name_list = [ 'blood-transfusion','blood-transfusion', 'banknote-authentication','banknote-authentication','banknote-authentication','breast-cancer']
fig, axs = plt.subplots(nrows = 3, ncols = 2)
k = 0
for  ar, ax in zip(obj_lists,axs.ravel()):
    
    if k > len(obj_lists):
        break
    el = np.array(ar)
    ax.axhline(y=lines_acc_advacc[k][1], color='orange', linestyle='--')
    ax.axhline(y=lines_acc_advacc[k][2], color='b', linestyle='--')
    ax.plot(range(np.shape(el)[0]),el[:,0],label = str(name_list[k]) + " \n objective" )
    ax.plot(range(np.shape(el)[0]),el[:,1],label = str(name_list[k]) + " \n adv acc" )
    ax.legend(loc = "lower left", prop={'size': 5})
    k += 1
plt.show()