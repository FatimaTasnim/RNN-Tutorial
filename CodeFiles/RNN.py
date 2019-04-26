from DataGenerator import *
from ForwardPropagation import *
from BackwardPropagation import *

# Data Generation
x, a0, da, parameters = DataGenerator()

# Forward Propagation
a, y, caches = rnn_forward(x, a0, parameters) 

# Forward Propagation Outputs
print("")
print("")
print("Outputs for Forward Propagation")
print("")
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)
print("y[1][3] =", y[1][3])
print("y.shape = ", y.shape)
print("caches[1][1][3] =", caches[1][1][3])
print("len(caches) = ", len(caches))
print("")
print("")

# Backward Propagation
gradients = rnn_backward(da, caches)


### Backward Propagation Outputs
print("Outputs for Backward Propagation")
print("")
print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients["dba"][4])
print("gradients[\"dba\"].shape =", gradients["dba"].shape)
print("")
print("")