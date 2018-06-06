
# coding: utf-8

# In[8]:


import GPy
import numpy as np

dim = 10

XY_data = np.genfromtxt("simdata.csv", delimiter=",", skip_header=1)
X = XY_data[:, 0:dim]
Y = XY_data[:, dim:dim+1]


# In[9]:


import timeit


# In[10]:


kern = GPy.kern # shorter
common_kwargs = {"lengthscale": 1.0, "variance": 1.0}
kerns = {
    "se": kern.RBF(dim, **common_kwargs),
    "mat12": kern.Exponential(dim, **common_kwargs),
    "rq": kern.RatQuad(dim, power=1.0, **common_kwargs),
    "se+rq":   kern.RBF(dim, **common_kwargs)
             + kern.RatQuad(dim, power=1.0, **common_kwargs),
    "se*rq":   kern.RBF(dim, **common_kwargs)
             * kern.RatQuad(dim, power=1.0, **common_kwargs),
    "se+se2+rq":   kern.RBF(dim, **common_kwargs)
                 + kern.RBF(dim, **common_kwargs)
                 + kern.RatQuad(dim, power=1.0, **common_kwargs),
    "(se+se2)*rq": (kern.RBF(dim, **common_kwargs)
                    +kern.RBF(dim, **common_kwargs)
                   ) * 
                   kern.RatQuad(dim, power=1.0, **common_kwargs),
    "mask(se, [1])": kern.RBF(1, active_dims=[0], **common_kwargs),
    "mask(se, [1])+mask(rq, [2:10])":    kern.RBF(1, active_dims=[0], **common_kwargs)  
                                    + kern.RatQuad(dim-1, power=1.0, active_dims=range(1,dim), **common_kwargs),
}
sefix = kern.RBF(2, **common_kwargs)
sefix.variance.fix()
kerns["fix(se, σ)"] = sefix


# In[11]:


mintimes = {}
for (label, k) in kerns.items():
    import gc
    gc.collect()
    gp = GPy.models.GPRegression(X, Y, k, noise_var=1.0)
    gc.collect()
    times = timeit.repeat("gp.parameters_changed()", setup="from __main__ import gp;gc.collect()", repeat=10, number=1)
    gc.collect()
    mintimes[label] = np.min(times)


# In[12]:


for k in mintimes.keys():
    print("%30s: %4.1f" % (k, mintimes[k]*1000.0))


# In[13]:


with open("bench_results/GPy.csv", "w") as f:
    for k,v in mintimes.items():
        f.write("\"%s\",%f\n" % (k,v*1000))


# In[14]:


# look at results for SE kernel
k = kerns["se"]
gp = GPy.models.GPRegression(X, Y, k, noise_var=1.0)
gp.parameters_changed()


# In[16]:


gp.log_likelihood()


# In[17]:


gp.gradient

