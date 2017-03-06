#This file tests the GP Monte Carlo function against the exact solution from GP regression

using GaussianProcesses

n = 20
X = linspace(-3,3,n)
sigma = 2.0
Y = X + sigma*randn(n)

#build the model
m = MeanZero()
k = Mat(3/2,0.0,0.0)
l = GaussLik(log(2.0))

gp1 = GPE(X', vec(Y), m, k, log(2.0))
gp2 = GPMC(X', vec(Y), m, k, l)

#compare log-likelihoods
abs(gp2.ll - gp1.mLL)>eps()

#compare the gradients as well

optimize!(gp1)
optimize!(gp2)



