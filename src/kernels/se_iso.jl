@doc """
# Description
Constructor for the isotropic Squared Exponential kernel (covariance)

k(x,x') = σ²exp(-(x-x')ᵀ(x-x')/2ℓ²)
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type SEIso <: Stationary
    ℓ2::Float64      # Length scale
    σ2::Float64      # Signal std
    SEIso(ll::Float64, lσ::Float64) = new(exp(2*ll), exp(2*lσ))
end

function set_params!(se::SEIso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Squared exponential only has two parameters"))
    se.ℓ2, se.σ2 = exp(2.0*hyp)
end

get_params(se::SEIso) = Float64[log(se.ℓ2)/2.0, log(se.σ2)/2.0]
get_param_names(::SEIso) = [:ll, :lσ]
num_params(se::SEIso) = 2

metric(se::SEIso) = SqEuclidean()
kern(se::SEIso, r::Float64) = se.σ2*exp(-0.5*r/se.ℓ2)

function grad_kern(se::SEIso, x::Vector{Float64}, y::Vector{Float64})
    r = distance(se, x, y)
    exp_r = exp(-0.5*r/se.ℓ2)
    g1 = se.σ2*r/se.ℓ2*exp_r # dK_d(log ℓ)
    g2 = 2.0*se.σ2*exp_r     # dK_d(log σ)
    return [g1,g2]
end

function grad_stack!(stack::AbstractArray, X::Matrix{Float64}, se::SEIso)
    nobsv = size(X, 2)
    R = distance(se, X)
    exp_R = exp(-0.5*R/se.ℓ2)
    for i in 1:nobsv, j in 1:i
        @inbounds stack[i,j,1] = se.σ2*R[i,j]/se.ℓ2*exp_R[i,j] # dK_dℓ
        @inbounds stack[j,i,1] = stack[i,j,1]
        @inbounds stack[i,j,2] = 2.0*se.σ2*exp_R[i,j]
        @inbounds stack[j,i,2] = stack[i,j,2]
    end

    return stack
end


