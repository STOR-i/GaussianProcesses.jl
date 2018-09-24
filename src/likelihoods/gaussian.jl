"""
    GaussLik <: Likelihood

Gaussian, a.k.a. Normal, likelihood
```math
p(y | f, σ) = 1 / √(2πσ²) \\exp(-(x - f)²/(2σ²)),
```
where standard deviation ``σ`` is a non-fixed hyperparameter and ``f`` is the latent
Gaussian process.
"""
mutable struct GaussLik <: Likelihood
    "Standard deviation"
    σ::Float64
    "Priors for likelihood parameters"
    priors::Array

    """
        GaussLik(lσ::Float64)

    Create `GaussLik` with standard deviation `exp(lσ)`.
    """
    GaussLik(lσ::Float64) = new(exp(lσ), [])
end

# log of probability density
function log_dens(gauss::GaussLik, f::VecF64, y::VecF64)
    return (-0.5 * log(2 * pi) - log(gauss.σ)) .- 0.5 * ((y - f) / gauss.σ).^2
end

# derivative of log pdf wrt latent function
function dlog_dens_df(gauss::GaussLik, f::VecF64, y::VecF64)
    return [(yi-fi)/gauss.σ^2 for (fi,yi) in zip(f,y)]
end

# derivative of log pdf wrt to parameters
function dlog_dens_dθ(gauss::GaussLik, f::VecF64, y::VecF64)
    return gauss.σ*[-1/gauss.σ + 1/gauss.σ^3*(yi-fi).^2 for (fi,yi) in zip(f,y)]
end

#mean and variance under likelihood
mean_lik(gauss::GaussLik, f::VecF64) = f
var_lik(gauss::GaussLik, f::VecF64) = ones(length(f))*gauss.σ^2


function set_params!(gauss::GaussLik, hyp::VecF64)
    length(hyp) == 1 || throw(ArgumentError("Gaussian/Normal likelihood has only one free parameter"))
    gauss.σ = exp(hyp[])
end

get_params(gauss::GaussLik) = Float64[log(gauss.σ)]
num_params(gauss::GaussLik) = 1


#Computes the predictive mean and variance
function predict_obs(gauss::GaussLik, fmean::VecF64, fvar::VecF64)
    return fmean, fvar + gauss.σ^2
end
