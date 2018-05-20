
# coding: utf-8

# In[7]:


import GPy
import numpy as np

d = 10
nobsv=3000
X = np.random.normal(size=(nobsv,d))
Y = np.random.normal(size=(nobsv,1))


# In[8]:


import timeit


# In[9]:


kern = GPy.kern # shorter
common_kwargs = {"lengthscale": 1.0, "variance": 1.0}
kerns = {
    "se": kern.RBF(d, **common_kwargs),
    "mat12": kern.Exponential(d, **common_kwargs),
    "rq": kern.RatQuad(d, power=1.0, **common_kwargs),
    "se+rq":   kern.RBF(d, **common_kwargs)
             + kern.RatQuad(d, power=1.0, **common_kwargs),
    "se*rq":   kern.RBF(d, **common_kwargs)
             * kern.RatQuad(d, power=1.0, **common_kwargs),
    "se+se2+rq":   kern.RBF(d, **common_kwargs)+kern.RBF(d, **common_kwargs)
                 + kern.RatQuad(d, power=1.0, **common_kwargs),
    "(se+se2)*rq": (kern.RBF(d, **common_kwargs)
                    +kern.RBF(d, **common_kwargs)
                   ) * 
                   kern.RatQuad(d, power=1.0, **common_kwargs),
    "mask(se, [1])": kern.RBF(1, active_dims=[0], **common_kwargs),
    "mask(se, [1])+mask(rq, [2:10])":    kern.RBF(1, active_dims=[0], **common_kwargs)  
                                    + kern.RatQuad(d-1, power=1.0, active_dims=range(1,d), **common_kwargs),
}
sefix = kern.RBF(2, **common_kwargs)
sefix.variance.fix()
kerns["fix(se, σ)"] = sefix


# In[10]:


mintimes = {}
for (label, k) in kerns.items():
    import gc
    gc.collect()
    gp = GPy.models.GPRegression(X, Y, k, noise_var=1.0)
    gc.collect()
    times = timeit.repeat("gp.parameters_changed()", setup="from __main__ import gp;gc.collect()", repeat=10, number=1)
    gc.collect()
    mintimes[label] = np.min(times)


# In[11]:


for k in mintimes.keys():
    print("%30s: %4.1f" % (k, mintimes[k]*1000.0))


# In[12]:


with open("bench_results/GPy.csv", "w") as f:
    for k,v in mintimes.items():
        f.write("\"%s\",%f\n" % (k,v*1000))

