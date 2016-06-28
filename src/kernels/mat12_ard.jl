# Matern 1/2 ARD covariance Function

@doc """
# Description
Constructor for the ARD Matern 1/2 kernel (covariance)

k(x,x') = σ²exp(-d/L^2), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat12Ard <: StationaryARD
    ℓ2::Vector{Float64}      # Log of length scale
    σ2::Float64              # Log of signal std
    dim::Int                 # Number of hyperparameters
    Mat12Ard(ll::Vector{Float64}, lσ::Float64) = new(2.0*exp(ll),exp(2.0*lσ), size(ll,1)+1)
end

function set_params!(mat::Mat12Ard, hyp::Vector{Float64})
    length(hyp) == mat.dim || throw(ArgumentError("Matern 1/2 covariance function only has $(mat.dim) parameters"))
    mat.ℓ2  = exp(2.0*hyp[1:(mat.dim-1)])
    mat.σ2 = exp(2.0*hyp[mat.dim])
end

get_params(mat::Mat12Ard) = [log(mat.ℓ2)/2.0; log(mat.σ2)/2.0]
get_param_names(mat::Mat12Ard) = [get_param_names(mat.ℓ2, :ll); :lσ]
num_params(mat::Mat12Ard) = mat.dim

metric(mat::Mat12Ard) = WeightedEuclidean(1.0./(mat.ℓ2))
cov(mat::Mat12Ard, r::Float64) = mat.σ2*exp(-r)

function grad_kern(mat::Mat12Ard, x::Vector{Float64}, y::Vector{Float64})
    r = distance(mat, x, y)
    exp_r = exp(-r)
    wdiff = ((x-y).^2)./mat.ℓ2
    
    g1 = (mat.σ2*wdiff*exp_r)/r
    g2 = 2.0*mat.σ2*exp_r
    
    return [g1; g2]
end

function grad_stack!(stack::AbstractArray, mat::Mat12Ard, X::Matrix{Float64}, data::StationaryARDData)
    d = size(X,1)
    R = distance(mat, X, data)
    stack[:,:,d+1] = mat.σ2 * exp(-R)
    ck = view(stack, :, :, d+1)
    broadcast!(*, view(stack, :, :, 1:d), data.dist_stack, reshape(1.0./mat.ℓ2, (1,1,d)), ck)
    broadcast!(/, view(stack, :, :, 1:d), view(stack, :, :, 1:d), R)
    stack[:,:, d+1] = 2.0 * ck
    return stack
end
