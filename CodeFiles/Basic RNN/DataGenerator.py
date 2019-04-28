import numpy as np

def DataGenerator():
    np.random.seed(1)
    x = np.random.randn(3,10,4)
    a0 = np.random.randn(5,10)
    Wax = np.random.randn(5,3)
    Waa = np.random.randn(5,5)
    Wya = np.random.randn(2,5)
    ba = np.random.randn(5,1)
    by = np.random.randn(2,1)
    da = np.random.randn(5, 10, 4)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
    return x, a0, da, parameters