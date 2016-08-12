# Matern 3/2 isotropic covariance function

@doc """
# Description
Constructor for the isotropic Matern 3/2 kernel (covariance)

k(x,x') = σ²(1+√3*d/ℓ)exp(-√3*d/ℓ), where d = |x-x'|
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat32Iso <: Isotropic
    ℓ::Float64       # Length scale 
    σ2::Float64      # Signal std
    Mat32Iso(ll::Float64, lσ::Float64) = new(exp(ll),exp(2*lσ))
end

function set_params!(mat::Mat32Iso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern 3/2 only has two parameters"))
    mat.ℓ, mat.σ2 = exp(hyp[1]), exp(2.0*hyp[2])
end

get_params(mat::Mat32Iso) = Float64[log(mat.ℓ), log(mat.σ2)/2.0]
get_param_names(mat::Mat32Iso) = [:ll, :lσ]
num_params(mat::Mat32Iso) = 2

metric(mat::Mat32Iso) = Euclidean()
cov(mat::Mat32Iso, r::Float64) = mat.σ2*(1+sqrt(3)*r/mat.ℓ)*exp(-sqrt(3)*r/mat.ℓ)

function grad_kern(mat::Mat32Iso, x::Vector{Float64}, y::Vector{Float64})
    r = distance(mat,x,y)
    exp_r = exp(-sqrt(3)*r/mat.ℓ)

    g1= mat.σ2*(sqrt(3)*r/mat.ℓ)^2*exp_r       #dK_d(log ℓ)
    g2 = 2.0*mat.σ2*(1+sqrt(3)*r/mat.ℓ)*exp_r  #dK_d(log σ)
    return [g1,g2]
end    


function grad_stack!(stack::AbstractArray, mat::Mat32Iso, X::Matrix{Float64}, data::IsotropicData)
    nobsv = size(X,2)
    R = distance(mat, X, data)
    exp_R = exp(-sqrt(3)*R/mat.ℓ)

    for i in 1:nobsv, j in 1:i
        @inbounds stack[i,j,1] = mat.σ2*(sqrt(3)*R[i,j]/mat.ℓ)^2*exp_R[i,j]       # dK_dℓ
        @inbounds stack[j,i,1] = stack[i,j,1] 
        @inbounds stack[i,j,2] = 2.0 *mat.σ2*(1+sqrt(3)*R[i,j]/mat.ℓ)*exp_R[i,j]  # dK_dσ
        @inbounds stack[j,i,2] = stack[i,j,2] 
    end
    
    return stack
end
