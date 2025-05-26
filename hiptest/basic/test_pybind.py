import numpy as np
import sys
sys.path.append('./build')
import vecadd

a = np.arange(10, dtype=np.float32)
b = np.ones(10, dtype=np.float32)

c = vecadd.vector_add(a, b)
print("a:", a)
print("b:", b)
print("c = a + b:", c)
