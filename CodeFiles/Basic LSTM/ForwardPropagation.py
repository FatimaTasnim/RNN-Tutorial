import numpy as np
from rnn_utils import *
from LSTM_UNIT_CELL import *

def LSTM_forward(x, a0, parameters):
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape
    
   
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    a_next = a0
    c_next = np.zeros(a_next.shape)
    
    for t in range(T_x):
        a_next, c_next, yt, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        a[:,:,t] = a_next
        y[:,:,t] = yt
        c[:,:,t]  = c_next
        caches.append(cache)
        
    caches = (caches, x)

    return a, y, c, caches