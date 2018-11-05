"""
    StuTLik <: Likelihood

Student-t likelihood (a.k.a. non-standardized [Student's
t-distribution](https://en.wikipedia.org/wiki/Student's_t-distribution))
```math
p(y | f, σ) = Γ((ν + 1)/2)/[Γ(ν/2)√(πν)σ](1 + 1/ν((y - f)/σ)²)^{-(ν + 1)/2}
```
with degrees of freedom ``ν ∈ N₀``, where scale ``σ`` is a non-fixed hyperparameter and
``f`` is the latent Gaussian process.
"""
mutable struct StuTLik <: Likelihood
    "Degrees of freedom"
    ν::Int
    "Scale"
    σ::Float64
    "Priors for likelihood parameters"
    priors::Array

    """
        StuTLik(ν::Int, lσ::Float64)

    Create a `StuTLik` with degrees of freedom `ν` and scale `exp(lσ)`.
    """
    StuTLik(ν::Int, lσ::Float64) = new(ν, exp(lσ), [])
end

#log of probability density
function log_dens(studentT::StuTLik, f::AbstractVector, y::AbstractVector)
    ν = studentT.ν
    σ = studentT.σ
    c = lgamma(0.5*(ν+1)) - lgamma(0.5*ν) - 0.5*log(pi*ν) - log(σ)
    return [c - (0.5*(ν+1))*log(1+(1/ν)*((yi-fi)/σ)^2) for (fi,yi) in zip(f,y)]
end

#derivative of log pdf wrt latent function
function dlog_dens_df(studentT::StuTLik, f::AbstractVector, y::AbstractVector)
    ν = studentT.ν
    σ = studentT.σ
    return [(ν+1)*(yi-fi)/(ν*σ^2 + (yi-fi)^2) for (fi,yi) in zip(f,y)]
end

#derivative of log pdf wrt to parameters
function dlog_dens_dθ(studentT::StuTLik, f::AbstractVector, y::AbstractVector)
    ν = studentT.ν
    σ = studentT.σ
    return σ*[-1/σ+(ν+1)*(yi-fi)^2/(ν*σ^3 + σ*(yi-fi)^2) for (fi,yi) in zip(f,y)]
end

#mean and variance under likelihood
mean_lik(studentT::StuTLik, f::AbstractVector) = f
var_lik(studentT::StuTLik, f::AbstractVector) = studentT.σ^2*(studentT.ν/(studentT.ν-2.0))*ones(length(f))

function set_params!(studentT::StuTLik, hyp::AbstractVector)
    length(hyp) == 1 || throw(ArgumentError("Student-t likelihood has only one free parameter"))
    studentT.σ = exp(hyp[])
end

get_params(studentT::StuTLik) = Float64[log(studentT.σ)]
num_params(studentT::StuTLik) = 1




