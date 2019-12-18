#VECTORIZATION
import numpy as np
import time as time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()
print("vectorized version time : "+ str(1000* (toc-tic))+ "ms")

c = 0
tic = time.time()
for i in range(1000000):
    c+= a[i]*b[i]
   
toc = time.time()
print("non-vectorized version time: "+ str(1000* (toc-tic))+ "ms")

