from DataGenerator import *
from ForwardPropagation import *
from BackwardPropagation import *

# Data Generation
x, a0, da, parameters = DataGenerator()

# Forward Propagation
a, y, caches = LSTM_forward(x, a0, parameters) 

# Forward Propagation Outputs

# Backward Propagation
gradients = LSTM_backward(da, caches)
