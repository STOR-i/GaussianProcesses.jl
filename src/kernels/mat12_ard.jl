# Matern 1/2 ARD covariance Function

@doc """
# Description
Constructor for the ARD Matern 1/2 kernel (covariance)

k(x,x') = σ²exp(-d/L), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat12Ard <: Stationary
    ℓ::Vector{Float64}      # Log of length scale
    σ2::Float64             # Log of signal std
    dim::Int                # Number of hyperparameters
    Mat12Ard(ll::Vector{Float64}, lσ::Float64) = new(exp(ll),exp(2.0*lσ), size(ll,1)+1)
end

function set_params!(mat::Mat12Ard, hyp::Vector{Float64})
    length(hyp) == mat.dim || throw(ArgumentError("Matern 1/2 covariance function only has $(mat.dim) parameters"))
    mat.ℓ  = exp(hyp[1:(mat.dim-1)])
    mat.σ2 = exp(2.0*hyp[mat.dim])
end

get_params(mat::Mat12Ard) = [log(mat.ℓ); log(mat.σ2)/2.0]
get_param_names(mat::Mat12Ard) = [get_param_names(mat.ℓ, :ll); :lσ]
num_params(mat::Mat12Ard) = mat.dim

metric(mat::Mat12Ard) = WeightedEuclidean(1.0./(mat.ℓ))
kern(mat::Mat12Ard, r::Float64) = mat.σ2*exp(-r)

function grad_kern(mat::Mat12Ard, x::Vector{Float64}, y::Vector{Float64})
    r = distance(mat, x, y)
    exp_r = exp(-r)
    wdiff = ((x-y)./mat.ℓ).^2
    
    g1 = (mat.σ2*wdiff*exp_r)/r
    g2 = 2.0*mat.σ2*exp_r
    
    return [g1; g2]
end

function grad_stack!(stack::AbstractArray, X::Matrix{Float64}, mat::Mat12Ard)
    d = size(X,1)
    R = distance(mat,X)
    stack[:,:,d+1] = crossKern(X, mat)
    ck = view(stack, :, :, d+1)
    for i in 1:d
        grad_ls = view(stack, :, :, i)
        pairwise!(grad_ls, WeightedSqEuclidean([1.0/mat.ℓ[i]^2]), view(X, i, :))
        map!(*, grad_ls, grad_ls, ck./R)
    end
    stack[:,:, d+1] = 2.0 * ck
    return stack
end
