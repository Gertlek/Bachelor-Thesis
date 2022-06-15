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

import warnings
from scipy.spatial.distance import pdist

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

def presort_data_indices(X):

    ind = np.argsort(X, axis=0)
    return ind
        
def check_features_scaled(X):
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    if np.any(min_values < 0) or np.any(max_values) > 1:
        warnings.warn(
            "found feature values outside of the [0, 1] range, "
            "features should be scaled to [0, 1] or ROCT will ignore their "
            "values for splitting."
        )


def same_class_samples_in_range(X, y, Delta_l, Delta_r):
    """
    Returns a list of tuples (i, j) where sample i (of class 0 or 1) is in
    range of sample j (of class 0 or 1)
    """
    diff_in_range = []
    diff_set = set()
    Delta_l = np.array(Delta_l)
    Delta_r = np.array(Delta_r)
    equal_in_range = []
    
    for i, (sample, label) in enumerate(zip(X, y)):
        sample_l = sample - Delta_l
        sample_r = sample + Delta_r

        for j in range(i + 1, len(X)):
            other_sample = X[j]
            
            if label != y[j]:
                if np.all(
                    (other_sample + Delta_r > sample_l)
                    & (other_sample - Delta_l <= sample_r)
                ):
                    diff_set.add(i)
                    diff_set.add(j)
                    
                    if label == 0:
                        diff_in_range.append((i, j))
                    else:
                        diff_in_range.append((j, i))
                        
    for i, (sample, label) in enumerate(zip(X, y)):
        sample_l = sample - Delta_l
        sample_r = sample + Delta_r
        
        for j in range(i + 1, len(X)):
            other_sample = X[j]
            
            if label == y[j]:

                if np.all(
                    (other_sample + Delta_r > sample_l)
                    & (other_sample - Delta_l <= sample_r)
                ):
                    if not (i in diff_set or j in diff_set):
                        if label == 0:
                            equal_in_range.append((i, j))
                        else:
                            equal_in_range.append((j, i))
                    
    return equal_in_range, diff_in_range


def __determine_epsilon(self, X):
        best_epsilon = 1.0
        for feature in range(self.n_features_in_):
            values = np.concatenate(
                (
                    X[:, feature],
                    X[:, feature] - self.Delta_l[feature],
                    X[:, feature] + self.Delta_r[feature],
                )
            )
            values = np.sort(values)
            differences = np.diff(values)

            # Determine the smallest non-zero difference
            epsilon = np.min(differences[np.nonzero(differences)])
            if epsilon < best_epsilon:
                best_epsilon = epsilon

        if best_epsilon > 1e-04:
            best_epsilon = 1e-04

        if best_epsilon < 1e-08:
            best_epsilon = 1e-08
        if self.verbose:
            print("epsilon:", best_epsilon)
        return best_epsilon
    
def children( m: int):
    if m == 0:
        return print("0 not allowed")
    if m == 1 :
        return [2,3]
    d = 0
    while True:
        d += 1 
        if 2**d <= m <= 2**(d+1)-1:
            hoever = m - 2**d
            childs= [m+2**d + hoever,hoever + m+2**d + 1]
            return childs

def __build_model_gurobi2(self, X, y):
    p = self.n_features_in_
    n = self.n_samples_
    T_B = range(1, (self.T // 2) + 1)
    T_L = range((self.T // 2) + 1, self.T + 1)
    def parent(t):

       return max(self._OptimalRobustTree__ancestors(t)[0],self._OptimalRobustTree__ancestors(t)[1])[0]
    Delta_l = self.Delta_l
    Delta_r = self.Delta_r
    
    model = gp.Model("Optimal_Robust_Tree_Fitting")
    
    a = model.addVars(range(1, p + 1), T_B, vtype=GRB.BINARY, name="a")
    z = model.addVars(
        range(1, n + 1), T_L, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="z"
    )
    s = model.addVars(range(1, n + 1), T_B, range(2), vtype=GRB.BINARY, name="s")
    b = model.addVars(T_B, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="b")
    c = model.addVars(T_L, vtype=GRB.BINARY, name="c")
    e = model.addVars(range(1, n + 1), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="e")
   
    # Let the nodes only split on one feature
    for t in T_B:
        model.addConstr(gp.quicksum(a[j, t] for j in range(1, p + 1)) == 1)
    
    epsilon = self._OptimalRobustTree__determine_epsilon(X)
    self.epsilon_ = epsilon
    M_left = M_right = 2 + epsilon
    

    for i in range(1, n + 1):

        for m in T_B:
# =============================================================================
#             if m <= int((len(T_B)-1)/2) and len(T_B)/2 > 3 :
#                 for indexie,child in enumerate(children(m)):
#                     model.addConstr( s[i,child,0] <= s[i,m,indexie]  )
#                     model.addConstr( s[i,child,1] <= s[i,m,indexie]  )
#                     
# =============================================================================
            if m != 1:
                if m % 2 == 0: # if left child 
                    
                    model.addConstr( ( s[i,parent(m),0] == 1 ) >> (gp.quicksum(a[j, m] * X[i - 1, j - 1] for j in range(1, p + 1))
                        - gp.quicksum(a[j, m] * Delta_l[j - 1] for j in range(1, p + 1))
                        >= epsilon + b[m] - M_left * s[i, m, 0] ))
                    model.addConstr( ( s[i,parent(m),0] == 1 ) >> (gp.quicksum(a[j, m] * X[i - 1, j - 1] for j in range(1, p + 1))
                        + gp.quicksum(a[j, m] * Delta_r[j - 1] for j in range(1, p + 1))
                        <= b[m] + M_right * s[i, m, 1] 
                    ))
                if m % 2 == 1 : # if right child 
                    
                  
                    model.addConstr( ( s[i,parent(m),1] == 1 ) >> (gp.quicksum(a[j, m] * X[i - 1, j - 1] for j in range(1, p + 1))
                        - gp.quicksum(a[j, m] * Delta_l[j - 1] for j in range(1, p + 1))
                        >= epsilon + b[m] - M_left * s[i, m, 0] ))
                    model.addConstr( ( s[i,parent(m),1] == 1 ) >> ( gp.quicksum(a[j, m] * X[i - 1, j - 1] for j in range(1, p + 1))
                        + gp.quicksum(a[j, m] * Delta_r[j - 1] for j in range(1, p + 1))
                        <= b[m] + M_right * s[i, m, 1] 
                    ))
            else:
                model.addConstr(
                        gp.quicksum(a[j, m] * X[i - 1, j - 1] for j in range(1, p + 1))
                        - gp.quicksum(a[j, m] * Delta_l[j - 1] for j in range(1, p + 1))
                        >= epsilon + b[m] - M_left * s[i, m, 0]  
                    )
                model.addConstr(
                        gp.quicksum(a[j, m] * X[i - 1, j - 1] for j in range(1, p + 1))
                        + gp.quicksum(a[j, m] * Delta_r[j - 1] for j in range(1, p + 1))
                        <= b[m] + M_right * s[i, m, 1] 
                    )
    for i in range(1, n + 1):
# =============================================================================
#         if self.max_depth > 1:
#             
#             model.addConstr( (  s[i,1,0] == 0)  >>  (gp.quicksum( z[i,m] for m in T_L if m < int((self.T //2 +1) + len(T_L)/2)  )  == 0 ) )
#             model.addConstr( (  s[i,1,1] == 0)  >>  (gp.quicksum( z[i,m] for m in T_L if m >= int((self.T //2 +1) + len(T_L)/2)  )  == 0) )
#         
# =============================================================================
# =============================================================================
#         if self.max_depth > 2:
#             
# 
#             model.addConstr( (s[i,2,0] == 0) >>  (gp.quicksum( z[i,m] for m in T_L if m < int((self.T //2 +1) + len(T_L)/4)  )  == 0 ) )
#             model.addConstr( (s[i,2,1] == 0) >>  (gp.quicksum( z[i,m] for m in T_L if int((self.T //2 +1) + len(T_L)/2)   >= m >= int((self.T //2 +1) + len(T_L)/4) )   == 0) )
# 
#             model.addConstr( (s[i,3,0] == 0) >>  (gp.quicksum( z[i,m] for m in T_L if int((self.T //2 +1) + len(T_L)/2 + len(T_L)/4)   >= m >= int((self.T //2 +1) + len(T_L)/2)  )  )  == 0 ) 
#             model.addConstr( (s[i,2,1] == 0) >>  (gp.quicksum( z[i,m] for m in T_L if m >= int((self.T //2 +1) + len(T_L)/2 + len(T_L)/4)    == 0) ) )
# =============================================================================
        
        
                     
        for t in T_L:
            A_l, A_r = self._OptimalRobustTree__ancestors(t)
            if self.max_depth == 0:
                model.addConstr(
                    gp.quicksum(s[i, m, 0] for m in A_l)
                    + gp.quicksum(s[i, m, 1] for m in A_r)
                    - len(A_l + A_r)
                    + 1
                    <= z[i, t]
                    )
                model.addConstr(
                    gp.quicksum(s[i, m, 0] for m in A_l)
                    + gp.quicksum(s[i, m, 1] for m in A_r)
                    - len(A_l + A_r)
                    + 1
                    <= z[i, t]
                )
            
            elif t < int((self.T //2 +1) + len(T_L)/2):
                model.addConstr(
                    gp.quicksum(s[i, m, 0] for m in A_l)
                    + gp.quicksum(s[i, m, 1] for m in A_r)
                    - len(A_l + A_r)
                    + s[i,1,0]
                    <= z[i, t]
                )
            else:
                model.addConstr(
                    gp.quicksum(s[i, m, 0] for m in A_l)
                    + gp.quicksum(s[i, m, 1] for m in A_r)
                    - len(A_l + A_r)
                    + s[i,1,1]
                    <= z[i, t]
                    )
                
    
    for i in range(1, n + 1):
        for t in T_L:
            if y[i - 1] == 0:
                model.addConstr(z[i, t] + c[t] - 1 <= e[i])
            else:
                model.addConstr(z[i, t] - c[t] <= e[i])
                
                
    
    # Add constraints stating that close samples with different labels
    # cannot both be classified correctly at once.

# =============================================================================
#     print(f"(equal,diff)  ( {len(equal_in_range)} , {len(in_range)} )")
# =============================================================================
    equal_in_range,diff_in_range = same_class_samples_in_range(X, y, self.Delta_l, self.Delta_r)
    
# =============================================================================
#     v = model.addVars(range(len(equal_in_range)), lb=0, ub=1, vtype=GRB.BINARY,name = "k")
#     
# =============================================================================
    for sample_i, other_sample_i in diff_in_range:
         model.addConstr(e[sample_i + 1] + e[other_sample_i + 1] >= 1)          
  
    
# =============================================================================
#     for count, (sample_i, other_sample_i) in enumerate(equal_in_range):
#         model.addConstr(e[sample_i + 1] <= e[other_sample_i + 1] + M_right*v[count] )
#         model.addConstr(e[sample_i + 1] >= e[other_sample_i + 1] - M_left*v[count] )
# 
# =============================================================================
# =============================================================================
#     for k in range(int(len(T_L)/2)):
#         if self.max_depth > 2 : 
#             
#             first_leaf = (self.T // 2) + 1
#         
#             model.addConstr(c[first_leaf + 2*k] + c[first_leaf + (2*k + 1)] == 1 )
#         
# =============================================================================
    
# =============================================================================
# 
#     model.setObjective(e.sum()+ (1/10)*v.sum(), GRB.MINIMIZE)
#     
# =============================================================================
    model.setObjective(e.sum(),GRB.MINIMIZE)
    
    tolerance = 10 ** (int(math.log10(epsilon)) - 1)
    return model, (a, z, e, s, b, c), tolerance

def __solve_model_gurobi2(self, model, variables, warm_start, tolerance):
   a, z, e, s, b, c = variables
   
   p = self.n_features_in_
   n = self.n_samples_
   T_B = range(1, (self.T // 2) + 1)
   T_L = range((self.T // 2) + 1, self.T + 1)

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

   # If record progress is True then keep track of the lower and upper
   # bounds over time
   if self.record_progress:
       self.lower_bounds_ = []
       self.upper_bounds_ = []
       self.runtimes_ = []
       
       def callback(model, where):
           if where == GRB.Callback.MIP:
               if model.cbGet(GRB.Callback.MIP_OBJBST) > len(e) :

                   self.upper_bounds_.append(len(e)) 

               else:
                    self.upper_bounds_.append(model.cbGet(GRB.Callback.MIP_OBJBST))
                    self.lower_bounds_.append(model.cbGet(GRB.Callback.MIP_OBJBND))
                    self.runtimes_.append(model.cbGet(GRB.Callback.RUNTIME))

       model.optimize(callback)
   else:
       model.optimize()

   self.train_adversarial_accuracy_ = 1 - (
       sum([e[i].X for i in range(1, n + 1)]) / n
   )

   if self.verbose:
       print("Error:", sum([e[i].X for i in range(1, n + 1)]))

   for t in T_B:
       if self.verbose:
           print(b[t].X)

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

def compute_average_runtime_cost(many_runtimes, many_costs):
    all_runtimes = [item for sublist in many_runtimes for item in sublist]
    all_runtimes = np.sort(np.unique(all_runtimes))

    costs_resampled = []
    for runtimes, costs in zip(many_runtimes, many_costs):
        indices = np.searchsorted(runtimes, all_runtimes, side='right') - 1
        costs = np.array(costs)
        costs_resampled.append(costs[indices])

    mean_costs = np.sum(costs_resampled, axis=0) / len(many_runtimes)
    sem_costs = np.std(costs_resampled, axis=0)*len(costs_resampled)/ np.sqrt(len(many_runtimes))

    return all_runtimes, mean_costs, sem_costs

def plot_runtimes_cost(many_runtimes, many_costs, color_index, label, only_avg=False):
    mean_runtimes, mean_costs, sem_costs = compute_average_runtime_cost(
        many_runtimes, many_costs
    )
    if only_avg:
        plt.fill_between(mean_runtimes, mean_costs, mean_costs + sem_costs, color=sns.color_palette()[color_index], alpha=0.05)
        plt.fill_between(mean_runtimes, mean_costs, mean_costs - sem_costs, color=sns.color_palette()[color_index], alpha=0.05)
    else:
        for runtimes, costs in zip(many_runtimes, many_costs):
            plt.plot(
                runtimes, costs, drawstyle="steps-post", c=sns.color_palette()[color_index], alpha=0.2
            )
    plt.plot(mean_runtimes, mean_costs, c=sns.color_palette()[color_index], drawstyle="steps-post", label=label)
    
ROCT_runtimes = []
ROCT_costs = []

ROCT2_runtimes = []
ROCT2_costs = []

acc_list = []
adv_acc_list = []
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
            attack_model = [epsilon]*len(X[1])

            timeout = 1800
            ROCT2  = OptimalRobustTree(max_depth=depth, attack_model=attack_model, time_limit = timeout, record_progress=True)
            ROCT2.fit2(X,y)
            
           
            ROCT  = OptimalRobustTree(max_depth=depth, attack_model=attack_model, time_limit = timeout, record_progress=True)
            ROCT.fit(X,y)
            
            
            ROCT_runtimes.append([0.0] + ROCT.runtimes_)
            ROCT_costs.append([1.0] + [cost / len(X) for cost in ROCT.upper_bounds_])

    
            ROCT2_runtimes.append([0.0] + ROCT2.runtimes_)
            ROCT2_costs.append([1.0] + [cost / (len(X)) for cost in ROCT2.upper_bounds_])
 
            m  = Model.from_groot(ROCT)
            m2 = Model.from_groot(ROCT2)

            adv_acc_list.append([name,epsilon,depth,m.adversarial_accuracy(X_test,y_test,epsilon = epsilon,options = {"disable_progress_bar" : True}),m2.adversarial_accuracy(X_test,y_test,epsilon = epsilon,options = {"disable_progress_bar" : True})])
            acc_list.append([name,epsilon,depth,m.accuracy(X_test,y_test),m2.accuracy(X_test,y_test)])
            
            print("Training acc:",m.accuracy(X,y),m2.accuracy(X,y))
            print("Acc: ROCT, ROCT2 of " + name + str(i))
            print(m.accuracy(X_test,y_test),m2.accuracy(X_test,y_test))
            print("Adversarial acc: ROCT, ROCT2 of " + name + str(i))
            print(m.adversarial_accuracy(X_test,y_test,epsilon = epsilon,options = {"disable_progress_bar" : True}),m2.adversarial_accuracy(X_test,y_test,epsilon = epsilon,options = {"disable_progress_bar" : True}))
            print()


sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.2)

plot_runtimes_cost(ROCT_runtimes, ROCT_costs, 0, "ROCT",only_avg=True)

plot_runtimes_cost(ROCT2_runtimes, ROCT2_costs, 1, "ROCT2",only_avg=True)        

plt.xlim(0.1, timeout)
plt.xlabel("Time (s)")
plt.ylabel("Mean % training error")
plt.ylim(0,1.01)
plt.legend()
plt.tight_layout()
plt.show()

print(np.mean(np.array(acc_list)[:,3].astype(float)) ,np.mean( np.array(acc_list)[:,4].astype(float)))

print(np.mean(np.array(adv_acc_list)[:,3].astype(float)) ,np.mean( np.array(adv_acc_list)[:,4].astype(float)))
