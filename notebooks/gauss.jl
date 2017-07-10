#This file tests the GP Monte Carlo function against the exact solution from GP regression

using Gadfly, GaussianProcesses

n = 20
X = linspace(-3,3,n)
sigma = 2.0
Y = sin(X) + sigma*randn(n)

#build the model
m = MeanZero()
k = SE(0.0,0.0)
l = GaussLik(log(sigma))

gp1 = GP(X', vec(Y), m, k, log(sigma))
gp2 = GP(X', vec(Y), m, k, l)


#maximise the parameters wrt the target
optimize!(gp1)
optimize!(gp2)

#compare log-likelihoods
abs(gp2.target - gp1.target)>eps()

#MCMC
samples1 = mcmc(gp1)
samples2 = mcmc(gp2)

xtest = linspace(minimum(gp1.X),maximum(gp1.X),50);
y1 = predict_y(gp1,xtest)[1]
y2 = predict_y(gp2,xtest)[1]

plot(layer(x=X,y=Y,Geom.point),
     layer(x=xtest,y=y1,Geom.line),
     layer(x=xtest,y=y2,Geom.line))


#likelihood parameters for gp2 are not converging to the correct answer, but are close. Seems to be an issue with the optimizer. I think the stopping condition needs to be depend on the magnitude of the gradient.

init_params = GaussianProcesses.get_params(gp1)

function ff(hyp)
    GaussianProcesses.set_params!(gp1, hyp)
    GaussianProcesses.update_target!(gp1)
    return gp1.target
end

GaussianProcesses.update_target_and_dtarget!(gp1)
theor_grad = copy(gp1.dtarget)
numer_grad = Calculus.gradient(ff, init_params)


init_params = GaussianProcesses.get_params(gp2)

function fg(hyp)
    GaussianProcesses.set_params!(gp2, hyp)
    GaussianProcesses.update_target!(gp2)
    return gp2.target
end

GaussianProcesses.update_target_and_dtarget!(gp2)
theor_grad = copy(gp2.dtarget)
numer_grad = Calculus.gradient(fg, init_params)

