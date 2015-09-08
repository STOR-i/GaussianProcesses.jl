# Squared Exponential Function with ARD

@doc """
# Description
Constructor for the ARD Squared Exponential kernel (covariance)

k(x,x') = σ²exp(-(x-x')ᵀL⁻²(x-x')/2), where L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type SEArd <: Stationary
    ℓ2::Vector{Float64}      # Log of Length scale
    σ2::Float64              # Log of Signal std
    dim::Int                 # Number of hyperparameters
    SEArd(ll::Vector{Float64}, lσ::Float64) = new(exp(2.0*ll),exp(2.0*lσ), size(ll,1)+1)
end

function set_params!(se::SEArd, hyp::Vector{Float64})
    length(hyp) == se.dim || throw(ArgumentError("Squared exponential ARD only has $(se.dim) parameters"))
    se.ℓ2 = exp(2.0*hyp[1:(se.dim-1)])
    se.σ2 = exp(2.0*hyp[se.dim])
end

get_params(se::SEArd) = [log(se.ℓ2)/2.0, log(se.σ2)/2.0]
num_params(se::SEArd) = se.dim

metric(se::SEArd) = WeightedSqEuclidean(1.0./(se.ℓ2))
kern(se::SEArd, r::Float64) = se.σ2*exp(-0.5*r)

function grad_kern(se::SEArd, x::Vector{Float64}, y::Vector{Float64})
    r = distance(se, x, y)
    exp_r = exp(-0.5*r)
    wdiff = ((x-y).^2)./se.ℓ2
    
    g1   = se.σ2.*wdiff*exp_r   #dK_d(log ℓ)
    g2 = 2.0*se.σ2*exp_r        #dK_d(log σ)
    
    return [g1, g2]
end
