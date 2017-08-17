using GaussianProcesses
using GaussianProcesses: get_params, get_param_names, num_params

d, n = 2, 20
ll = rand(d)
x = 2π * rand(d, n)
y = randn(n) + 0.5

""" Not much of a test really... just checks that it doesn't crash
"""
function test_mcmc(kern::Kernel,lik::Likelihood,X::Matrix{Float64}, y::Vector{Float64})
	gp = GP(X,y,MeanZero(), kern,lik)
        set_priors!(gp.k,[Distributions.Normal(-1.0,1.0) for i in 1:num_params(gp.k)])
	mcmc(gp)
end


rq = RQ(1.0, 1.0, 1.0)
lik = GaussLik(-1.0)
test_mcmc(rq, lik, x, y)
