"""
# Description
Constructor for the Bernoulli likelihood
    
    p(y=k|f) = θᵏ(1-θ)¹⁻ᵏ, for k=0, 1

where θ = Φ(f), and f is the latent Gaussian process
"""
type BernLik <: Likelihood
    BernLik() = new()
end

function log_dens(bernoulli::BernLik, f::Vector{Float64}, y::Vector{Bool})
    return Float64[yi? log(Φ(fi)) : log(1.0 - Φ(fi)) for (fi,yi) in zip(f,y)]
end

function dlog_dens_df(bernoulli::BernLik, f::Vector{Float64}, y::Vector{Bool})
    return Float64[yi? φ(fi)/Φ(fi) : -φ(fi)/(1.0 - Φ(fi)) for (fi,yi) in zip(f,y)]
end                   

#mean and variance under the likelihood
mean_lik(bernoulli::BernLik, f::Vector{Float64}) = Float64[Φ(fi) for fi in f]
var_lik(bernoulli::BernLik, f::Vector{Float64}) = Float64[Φ(fi)*(1-Φ(fi)) for fi in f]

get_params(bernoulli::BernLik) = []
num_params(bernoulli::BernLik) = 0


#Computes the predictive mean and variance 
function predict_obs(bernoulli::BernLik, fmean::Vector{Float64}, fvar::Vector{Float64})
    p = Float64[Φ(fm./sqrt(1+fv)) for (fm,fv) in zip(fmean,fvar)]
    return p, p-p.*p
end
