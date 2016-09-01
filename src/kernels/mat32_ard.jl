# Matern 3/2 ARD covariance function

@doc """
# Description
Constructor for the ARD Matern 3/2 kernel (covariance)

k(x,x') = σ²(1+√3*d/L^2)exp(-√3*d/L^2), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat32Ard <: StationaryARD
    ℓ2::Vector{Float64}     # Log of Length scale 
    σ2::Float64            # Log of Signal std
    Mat32Ard(ll::Vector{Float64}, lσ::Float64) = new(exp(2.0*ll), exp(2.0*lσ))
end

function set_params!(mat::Mat32Ard, hyp::Vector{Float64})
    length(hyp) == num_params(mat) || throw(ArgumentError("Mat32 kernel only has $(num_params(mat)) parameters"))
    d=length(mat.ℓ2)
    mat.ℓ2 = exp(2.0*hyp[1:d])
    mat.σ2 = exp(2.0*hyp[d+1])
end

get_params(mat::Mat32Ard) = [log(mat.ℓ2)/2.0; log(mat.σ2)/2.0]
get_param_names(mat::Mat32Ard) = [get_param_names(mat.ℓ2, :ll); :lσ]
num_params(mat::Mat32Ard) = length(mat.ℓ2) + 1

metric(mat::Mat32Ard) = WeightedEuclidean(1.0./(mat.ℓ2))
cov(mat::Mat32Ard, r::Float64) = mat.σ2*(1+sqrt(3)*r)*exp(-sqrt(3)*r)

function grad_kern(mat::Mat32Ard, x::Vector{Float64}, y::Vector{Float64})
    r = distance(mat, x, y)
    exp_r = exp(-sqrt(3)*r)
    wdiff = ((x-y).^2)./mat.ℓ2
    
    g1 = 3*mat.σ2*wdiff*exp_r
    g2 = 2.0*mat.σ2*(1+sqrt(3)*r)*exp_r
    
    return [g1; g2]
end

function grad_stack!(stack::AbstractArray, mat::Mat32Ard, X::Matrix{Float64}, data::StationaryARDData)
    d = size(X,1)
    R = distance(mat, X, data)
    exp_R = exp(-sqrt(3)*R)
    broadcast!(*, view(stack, :, :, 1:d), mat.σ2*3, data.dist_stack, reshape(1.0./mat.ℓ2, (1,1,d)), exp_R)
    stack[:,:, d+1] = 2.0 * mat.σ2 * (1 + sqrt(3) * R) .* exp_R
    return stack
end
