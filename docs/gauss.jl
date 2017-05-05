#This file tests the GP Monte Carlo function against the exact solution from GP regression

using GaussianProcesses

n = 20
X = linspace(-3,3,n)
sigma = 2.0
Y = X + sigma*randn(n)

#build the model
m = MeanZero()
k = Matern(3/2,0.0,0.0)
l = GaussLik(log(2.0))

gp1 = GP(X', vec(Y), m, k, log(2.0))
gp2 = GP(X', vec(Y), m, k, l)

#compare log-likelihoods
abs(gp2.target - gp1.target)>eps()

#compare the gradients as well

optimize!(gp1)
optimize!(gp2)



