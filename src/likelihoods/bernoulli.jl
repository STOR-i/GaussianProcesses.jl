"""
# Description
Constructor for the Bernoulli likelihood

p(y=k|p) = pᵏ(1-p)¹⁻ᵏ, for k=0,1
# Arguments:
* `p::Float64`: probability of a success
"""
type Bernoulli <: Likelihood
    Bernoulli() = new()
end

function loglik(bernoulli::Bernoulli, f::Vector{Float64}, y::Vector{Bool})
    return Float64[yi? log(Φ(fi)) : 1.0 - log(Φ(fi)) for (fi,yi) in zip(f,y)]
end

get_params(bernoulli::Bernoulli) = []
num_params(bernoulli::Bernoulli) = 0
