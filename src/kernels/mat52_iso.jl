# Matern 5/2 isotropic covariance function

@doc """
# Description
Constructor for the isotropic Matern 5/2 kernel (covariance)

k(x,x') = σ²(1+√5*d/ℓ + 5d²/3ℓ²)exp(-√5*d/ℓ), where d = |x-x'|
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat52Iso <: Stationary
    ℓ::Float64      # Length scale 
    σ2::Float64     # Signal std
    Mat52Iso(ll::Float64, lσ::Float64) = new(exp(ll), exp(2*lσ))
end

function set_params!(mat::Mat52Iso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern 5/2 only has two parameters"))
    mat.ℓ, mat.σ2 = exp(hyp[1]), exp(2.0*hyp[2])
end
get_params(mat::Mat52Iso) = Float64[log(mat.ℓ), log(mat.σ2)/2.0]
get_param_names(mat::Mat52Iso) = [:ll, :lσ]
num_params(mat::Mat52Iso) = 2

metric(mat::Mat52Iso) = Euclidean()
cov(mat::Mat52Iso, r::Float64) = mat.σ2*(1+sqrt(5)*r/mat.ℓ+5*r^2/(3*mat.ℓ^2))*exp(-sqrt(5)*r/mat.ℓ)

function grad_kern(mat::Mat52Iso, x::Vector{Float64}, y::Vector{Float64})
    r = distance(mat,x,y)
    exp_r = exp(-sqrt(5)*r/mat.ℓ)

    g1 = mat.σ2*(5*r^2/mat.ℓ^2)*((1+sqrt(5)*r/mat.ℓ)/3)*exp_r   #dK_d(log ℓ)
    g2 = 2.0*mat.σ2*(1+sqrt(5)*r/mat.ℓ+5*r^2/(3*mat.ℓ^2))*exp_r #dK_d(log σ)
    return [g1,g2]
end

function grad_stack!(stack::AbstractArray, mat::Mat52Iso, X::Matrix{Float64}, data::IsotropicData)
    nobsv = size(X,2)
    R = distance(data)
    exp_R = exp(-sqrt(5)*R/mat.ℓ)

    for i in 1:nobsv, j in 1:i
        @inbounds stack[i,j,1] = mat.σ2*(5*R[i,j]^2/mat.ℓ^2)*((1+sqrt(5)*R[i,j]/mat.ℓ)/3)*exp_R[i,j]      # dK_dℓ
        @inbounds stack[j,i,1] = stack[i,j,1] 
        @inbounds stack[i,j,2] = 2.0*mat.σ2*(1+sqrt(5)*R[i,j]/mat.ℓ+5*R[i,j]^2/(3*mat.ℓ^2))*exp_R[i,j]   # dK_dσ
        @inbounds stack[j,i,2] = stack[i,j,2] 
    end
    
    return stack
end
