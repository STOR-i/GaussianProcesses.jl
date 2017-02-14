
# coding: utf-8

# # Table of Contents
#  <p>

# In[1]:

# get_ipython().magic('matplotlib inline')
# get_ipython().magic("config InlineBackend.figure_format='retina'")


# In[2]:

import GPy
import numpy as np

nobsv=3000
X = np.random.normal(size=(nobsv,2))
Y = np.random.normal(size=(nobsv,1))
se = GPy.kern.RBF(2, lengthscale=1.0)

gp_se = GPy.models.GPRegression(X, Y, se)


# In[3]:

import gc
gc.collect()
get_ipython().magic('timeit -n2 gp_se.parameters_changed()')


# In[4]:

rq = GPy.kern.RatQuad(2, lengthscale=1.0, power=1.0)
gp_rq = GPy.models.GPRegression(X, Y, rq)


# In[5]:

get_ipython().magic('timeit gp_rq.parameters_changed()')


# In[6]:

gp_sum = GPy.models.GPRegression(X, Y, se+rq)


# In[7]:

get_ipython().magic('timeit gp_sum.parameters_changed()')


# In[8]:

gp_prod = GPy.models.GPRegression(X, Y, se*rq)


# In[9]:

get_ipython().magic('timeit gp_prod.parameters_changed()')


# In[10]:

se2 = GPy.kern.RBF(2, lengthscale=1.0)


# In[11]:

gp_sum = GPy.models.GPRegression(X, Y, se+se2+rq)


# In[12]:

get_ipython().magic('timeit gp_sum.parameters_changed()')


# In[13]:

gp_prod = GPy.models.GPRegression(X, Y, (se+se2)*rq)


# In[14]:

get_ipython().magic('timeit gp_prod.parameters_changed()')


# In[15]:

semask = GPy.kern.RBF(1, lengthscale=1.0, active_dims=[0])
rqmask = GPy.kern.RatQuad(1, lengthscale=1.0, power=1.0, active_dims=[1])


# In[16]:

gp = GPy.models.GPRegression(X, Y, semask)
get_ipython().magic('timeit gp.parameters_changed()')


# In[17]:

gp = GPy.models.GPRegression(X, Y, semask+rqmask)
get_ipython().magic('timeit gp.parameters_changed()')


# In[18]:

sefix = GPy.kern.RBF(2, lengthscale=1.0)
sefix.variance.fix()
gp = GPy.models.GPRegression(X, Y, sefix)
get_ipython().magic('timeit gp.parameters_changed()')


# In[19]:

get_ipython().magic('time gp.parameters_changed()')

