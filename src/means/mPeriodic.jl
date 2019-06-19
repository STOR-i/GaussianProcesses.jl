# Periodic mean function

"""
    MeanPeriodic <: Mean

Periodic mean function
```math
m(x) = a'cos(2πx/p) + b'sin(2πx/p)
```
with polynomial coefficients ``βᵢⱼ`` of shape ``d × D`` where ``d`` is the dimension of
observations and ``D`` is the degree of the polynomial.
"""
mutable struct MeanPeriodic <: Mean
    "Cosine and sine coefficients"
    a::Vector{Float64}
    b::Vector{Float64}
    "Period"
    p::Vector{Float64}
    "Priors for mean parameters"
    priors::Array

    """
        MeanPeriodic(a::Vector{Float64}, b::Vector{Float64}, p::Vector{Float64})

    Create `MeanPeriodic` with cosine and sine coefficients `a` and `b`
    and period `p`.
    """
    MeanPeriodic(a::Vector{Float64}, b::Vector{Float64}, lp::Vector{Float64}) = new(a, b, exp.(lp), [])
end
MeanPeriodic(a::Real, b::Real, lp::Real) = MeanPeriodic([a], [b], [lp])

function mean(mPer::MeanPeriodic, x::AbstractVector)
    return dot(mPer.a, cos.(2π .* x ./ mPer.p)) + 
           dot(mPer.b, sin.(2π .* x ./ mPer.p))
end

get_params(mPer::MeanPeriodic) = [mPer.a; mPer.b; log.(mPer.p)]
get_param_names(mPer::MeanPeriodic) = [get_param_names(mPer.a, :a);
                                       get_param_names(mPer.b, :b);
                                       get_param_names(mPer.p, :lp)]
num_params(mPer::MeanPeriodic) = length(mPer.a)+length(mPer.b)+length(mPer.p)

function set_params!(mPer::MeanPeriodic, hyp::AbstractVector)
    length(hyp) == num_params(mPer) || throw(ArgumentError("MeanPeriodic mean function has $(num_params(mPer)) parameters"))
    dim = length(mPer.a)
    copyto!(mPer.a, hyp[1:dim])
    copyto!(mPer.b, hyp[dim+1:2dim])
    copyto!(mPer.p, exp.(hyp[2dim+1:3dim]))
end

function grad_mean(mPer::MeanPeriodic, x::AbstractVector)
    dim = length(x)
    costerm = cos.(2π .* x ./ mPer.p)
    sinterm = sin.(2π .* x ./ mPer.p)
    dM_da = costerm
    dM_db = sinterm
    dM_dlp =  (mPer.a .* sinterm .- mPer.b .* costerm) .*
                       (2π .* x ./ mPer.p)
    return [dM_da; dM_db; dM_dlp]
end
