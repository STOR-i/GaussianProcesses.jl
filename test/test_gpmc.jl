#This file checks that each of the likelihoods works

using GaussianProcesses
import GaussianProcesses: update_target!, update_target_and_dtarget!, set_params!, get_params, Likelihood

d, n = 3, 20
ll = rand(d)
X = 2π * rand(d, n)
y = Float64[sum(sin.(X[:,i])) for i in 1:n]/d

#mean and kernel function
mZero = MeanZero()
kern = SE(0.0,0.0)

#used for numerical gradients
function objective(para::Vector{Float64})
    temp_para = copy(para)
    set_params!(gp,para)
    update_target!(gp)
    return gp.target
end    

#Test that the GPMC constructor fits to the likelihood and check the target derivatives
function test_gpmc(y::Vector,lik::Likelihood)
    gp = GP(X,y,mZero,kern,lik)

    params = randn(length(get_params(gp)))
    num_grad = Calculus.gradient(θ->objective(θ),params)

    set_params!(gp,params)
    theor_grad = update_target_and_dtarget!(gp)
    
    @test num_grad ≈ theor_grad
end


#Bernoulli likelihood
Y = convert(Vector{Bool},[rand(Distributions.Bernoulli(abs(y[i]))) for i in 1:n])
lik = BernLik()
    
test_gpmc(Y,lik)


#Binomial likelihood
Y = [rand(Distributions.Binomial(n,exp(y[i])/(1.0+exp(y[i])))) for i in 1:n]
lik = BinLik(n)
test_gpmc(Y,lik)

#Exponential likelihood
Y = [rand(Distributions.Exponential(y[i]^2)) for i in 1:n]
lik = ExpLik()
test_gpmc(Y,lik)

#Gaussin likelihood     
lik = GaussLik(-1.0)
test_gpmc(y,lik)

#Poisson likelihood
Y = [rand(Distributions.Poisson(exp(y[i]))) for i in 1:n]
lik = PoisLik()
test_gpmc(Y,lik)

#Student-T likelihood
df = 3
Y = y .+ rand(Distributions.TDist(df),n)
lik = StuTLik(df,0.1)
test_gpmc(Y,lik)






