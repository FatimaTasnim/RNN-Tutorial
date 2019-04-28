import numpy as np
from rnn_utils import *
from RNN_UNIT_CELL import *

def rnn_forward(x, a0, parameters):

    caches = []
    
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
  
    a = np.zeros((n_a,m,T_x))
    y_pred = np.zeros((n_y,m, T_x))
    
    a_next = a0
    
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
        
    caches = (caches, x)
    
    return a, y_pred, caches