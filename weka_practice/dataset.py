import random
import numpy as np
a = -5
b = 5
matrix = np.array([])

for i in range(1,6001):
    x1 = a + (b-a)*random.random()
    x2 = a + (b-a)*random.random()
    class_ = int(np.sign(-2 + x1 + 2 * x2)) ### I think it should be  ( -2*x1 + 2*x2 )
    arr = np.array([x1,x2,class_])
    matrix = np.hstack((matrix, arr))
np.savetxt('dataset.dat', matrix)