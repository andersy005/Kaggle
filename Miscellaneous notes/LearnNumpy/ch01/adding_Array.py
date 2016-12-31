
# coding: utf-8

# In[ ]:

import sys
from datetime import datetime
import numpy as np


# In[ ]:

def pythonsum(n):
    a = range(n)
    print a
    b = range(n)
    print b
    c = []
    
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i] + b[i])
    return c


# In[ ]:

pythonsum(5)


# The following is a function that achieves the same with Numpy

# In[ ]:

def numpysum(n):
    a = np.arange(n) ** 2
    print a
    b = np.arange(n) ** 3
    print b
    c = a + b
    return c


# In[ ]:

numpysum(5)


# In[ ]:

# Measuring the elapsed time
size = int(sys.argv[1])

start = datetime.now()
c = pythonsum(size)
delta = datetime.now() - start
print "The last 2 elements of the sum", c[-2:]
print "PythonSum elapsed time in microseconds", delta.microseconds


start = datetime.now()
c = numpysum(size)
delta = datetime.now() - start
print "The last 2 elements of the sum", c[-2:]
print "PythonSum elapsed time in microseconds", delta.microseconds


# In[ ]:



