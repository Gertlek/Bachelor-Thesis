import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.preprocessing import MinMaxScaler

import math

from groot.datasets import load_all,load_mnist
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
    Delta_l = self.Delta_l
    Delta_r = self.Delta_r

    p = self.n_features_in_
    n = self.n_samples_
    T_B = range(1, (self.T // 2) + 1)
    T_L = range((self.T // 2) + 1, self.T + 1)
    K = range(len(np.unique(y)))
    
    model = gp.Model("Optimal_Robust_Tree_Fitting")
    
    a = model.addVars(range(1, p + 1), T_B, vtype=GRB.BINARY, name="a")
    z = model.addVars(
        range(1, n + 1), T_L, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="z"
    )
    s = model.addVars(range(1, n + 1), T_B, range(2), vtype=GRB.BINARY, name="s")
    b = model.addVars(T_B, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="b")
    c = model.addVars(T_L,K, vtype=GRB.BINARY, name="c")
    e = model.addVars(range(1, n + 1), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="e")
    
    # The objective is to minimize the sum of errors (weighted by 1 each)
    model.setObjective(e.sum(), GRB.MINIMIZE)
    

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
                - gp.quicksum(a[j, m] * Delta_l[j - 1] for j in range(1, p + 1))
                >= epsilon + b[m] - M_left * s[i, m, 0]
            )
            model.addConstr(
                gp.quicksum(a[j, m] * X[i - 1, j - 1] for j in range(1, p + 1))
                + gp.quicksum(a[j, m] * Delta_r[j - 1] for j in range(1, p + 1))
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
    for t in T_L: 
        model.addConstr( gp.quicksum( c[t,k] for k in K )  == 1)
        
    for i in range(1, n + 1):
        for t in T_L:
            for k in K:
                if y[i - 1] == k:
                    model.addConstr(z[i, t]  - c[t,k]    <= e[i])
    
    
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

   self.model = model

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
       value = []
       for k in range(len(np.unique(y))):
           value.append(round(c[t,k].X))
       value = np.array(value)

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
from sklearn.datasets import load_iris,load_wine

df = load_wine()
X = df['data']
X = X[:,:2] 
y = df['target']
class_names = df['target_names']
feature_names = df['feature_names'][0:2]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
    
epsilon = 0.05
depth   = 5
timeout = 300

attack_model = [epsilon]*len(X[1])

ROCT_with_eps  = OptimalRobustTree(max_depth=depth, attack_model=attack_model, time_limit = timeout, record_progress=True)
ROCT_with_eps.fit2(X,y)

ROCT  = OptimalRobustTree(max_depth=depth, attack_model=None, time_limit = timeout, record_progress=True)
ROCT.fit2(X,y)


def plot_estimator(X, y, mod, ax=None, steps=100, colors=("b", "r")):

    if X.shape[1] != 2:
        raise ValueError("X must be 2D")

    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    x_low = np.min(X[:, 0])
    x_high = np.max(X[:, 0])
    y_low = np.min(X[:, 1])
    y_high = np.max(X[:, 1])

    x_extra = (x_high - x_low) * 0.1
    y_extra = (y_high - y_low) * 0.1

    x_low -= x_extra
    x_high += x_extra
    y_low -= y_extra
    y_high += y_extra

    xx, yy = np.meshgrid(
        np.linspace(x_low, x_high, steps), np.linspace(y_low, y_high, steps)
    )

    Z = mod.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if isinstance(mod, Model) :
        ax.contourf(xx, yy, Z, alpha=0.1, levels=1, colors=colors)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], marker="_", c="b")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], marker="+", c="r")
    else:
        vmin = np.min(y)
        vmax = np.max(y)
        ax.contourf(xx, yy, Z, alpha=0.2, cmap="brg", levels=10, vmin=vmin, vmax=vmax)
        scatter = ax.scatter(X[:, 0], X[:, 1], marker=".", c=y, cmap="brg", vmin=vmin, vmax=vmax)

    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Wine \n class",bbox_to_anchor=(1.1, 1.1))
    ax.add_artist(legend1)
    
    ax.set_title("epsilon = " + str(mod.attack_model[0]))
    
plot_estimator(X,y,ROCT_with_eps)
plot_estimator(X,y,ROCT)
print("ROCT, OCT : ", ROCT_with_eps.score(X,y),ROCT.score(X,y))