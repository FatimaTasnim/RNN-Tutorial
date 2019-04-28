from DataGenerator import *
from ForwardPropagation import *

x, a0, da, parameters = DataGenerator()

a, y, caches = LSTM_forward(x, a0, parameters) 

# Forward Propagation Outputs

# Backward Propagation
#gradients = LSTM_backward(da, caches)
