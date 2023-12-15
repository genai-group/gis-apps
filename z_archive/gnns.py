#

#%%
import os, numpy as np
import pandas as pd

# Initial connections

#%%

# undirected edges
nodes = [[0,1,0,0,0], 
         [1,0,1,0,0],
         [0,1,0,1,1],
         [0,0,1,0,0],
         [0,0,1,0,0]]

# degree of each node

degrees = [2,2,3,1,1]

#%%
feature_vector = [1,2,3,4,5]

#%%
# updateed feature_vector
feature_vector = np.matmul(nodes, feature_vector)

# %%
