"""
# Description
Constructor for the Gaussian, aka Normal, likelihood

p(y|μ,σ²) = 1/(√(2πσ²))*exp(-(x-μ)²/2σ²)
# Arguments:
* `μ::Float64`: Mean
* `lσ::Float64`: Log of the signal standard deviation σ
"""
type Gaussian <: Likelihood
    μ::Float64      # mean
    σ2::Float64     # variance
    Gaussian(μ::Float64, lσ::Float64) = new(μ, exp(2*lσ))
end

function loglik(gaussian::Gaussian, f::Vector{Float64}, y::Vector{Float64})
    return Distributions.logpdf(Distributions.MvNormal(f,lik.σ2*eye(length(f))),y)
end
