# Squared Exponential Function with ARD

@doc """
# Description
Constructor for the ARD Squared Exponential kernel (covariance)

k(x,x') = σ²exp(-(x-x')ᵀL⁻²(x-x')/2), where L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type SEArd <: StationaryARD
    ℓ2::Vector{Float64}      # Log of Length scale
    σ2::Float64              # Log of Signal std
    SEArd(ll::Vector{Float64}, lσ::Float64) = new(exp(2.0*ll),exp(2.0*lσ))
end

function set_params!(se::SEArd, hyp::Vector{Float64})
    length(hyp) == num_params(se) || throw(ArgumentError("SEArd only has $(num_params(se)) parameters"))
    d = length(se.ℓ2)
    se.ℓ2 = exp(2.0*hyp[1:d])
    se.σ2 = exp(2.0*hyp[d+1])
end

get_params(se::SEArd) = [log(se.ℓ2)/2.0; log(se.σ2)/2.0]
get_param_names(k::SEArd) = [get_param_names(k.ℓ2, :ll); :lσ]
num_params(se::SEArd) = length(se.ℓ2) + 1

metric(se::SEArd) = WeightedSqEuclidean(1.0./(se.ℓ2))
cov(se::SEArd, r::Float64) = se.σ2*exp(-0.5*r)

function grad_kern(se::SEArd, x::Vector{Float64}, y::Vector{Float64})
    r = distance(se, x, y)
    exp_r = exp(-0.5*r)
    wdiff = ((x-y).^2)./se.ℓ2
    
    g1   = se.σ2.*wdiff*exp_r   #dK_d(log ℓ)
    g2 = 2.0*se.σ2*exp_r        #dK_d(log σ)
    
    return [g1; g2]
end

function grad_stack!(stack::AbstractArray, se::SEArd, X::Matrix{Float64}, data::StationaryARDData)
    d = size(X,1)
    ck = cov(se, X, data)
    broadcast!(*, view(stack, :, :, 1:d), data.dist_stack, reshape(1.0./se.ℓ2, (1,1,d)), ck)
    stack[:,:, d+1] = 2.0 * ck
    return stack
end
