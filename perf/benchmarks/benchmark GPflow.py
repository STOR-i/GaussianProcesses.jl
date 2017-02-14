
# coding: utf-8

# # Table of Contents
#  <p>

# In[1]:

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")


# In[2]:

import GPflow
import numpy as np

nobsv=3000
X = np.random.normal(size=(nobsv,2))
Y = np.random.normal(size=(nobsv,1))


# In[23]:

se = GPflow.kernels.RBF(2, lengthscales=1.0)
gp = GPflow.gpr.GPR(X, Y, se)
x=gp.get_free_state()
gp._compile()
gp._objective(x)
get_ipython().magic('timeit gp._objective(x)')

