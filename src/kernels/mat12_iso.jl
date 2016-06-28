# Matern 1/2 isotropic covariance Function

@doc """
# Description
Constructor for the isotropic Matern 1/2 kernel (covariance)

k(x,x') = σ²exp(-d/ℓ), where d=|x-x'|
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat12Iso <: Isotropic
    ℓ::Float64     # Length scale
    σ2::Float64    # Signal std
    Mat12Iso(ll::Float64, lσ::Float64) = new(exp(ll),exp(2*lσ))
end

function set_params!(mat::Mat12Iso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern 1/2 covariance function only has two parameters"))
    mat.ℓ, mat.σ2 = exp(hyp[1]), exp(2.0*hyp[2])
end

get_params(mat::Mat12Iso) = Float64[log(mat.ℓ), log(mat.σ2)/2.0]
get_param_names(mat::Mat12Iso) = [:ll, :lσ]
num_params(mat::Mat12Iso) = 2

metric(mat::Mat12Iso) = Euclidean()
cov(mat::Mat12Iso, r::Float64) = mat.σ2*exp(-r/mat.ℓ)

function grad_kern(mat::Mat12Iso, x::Vector{Float64}, y::Vector{Float64})
    r = distance(mat,x,y)
    exp_r = exp(-r/mat.ℓ)
    
    g1 = mat.σ2*r/mat.ℓ*exp_r  #dK_d(log ℓ)
    g2 = 2.0*mat.σ2*exp_r      #dK_d(log σ)
    return [g1,g2]
end


function grad_stack!(stack::AbstractArray, mat::Mat12Iso, X::Matrix{Float64}, data::IsotropicData)
    nobsv = size(X,2)
    R = distance(mat, X, data)
    exp_R = exp(-R/mat.ℓ)

    for i in 1:nobsv, j in 1:i
        @inbounds stack[i,j,1] = mat.σ2*R[i,j]/mat.ℓ*exp_R[i,j] # dK_dℓ
        @inbounds stack[j,i,1] = stack[i,j,1] 
        @inbounds stack[i,j,2] = 2.0 * mat.σ2 * exp_R[i,j]    # dK_dσ
        @inbounds stack[j,i,2] = stack[i,j,2] 
    end
    
    return stack
end
