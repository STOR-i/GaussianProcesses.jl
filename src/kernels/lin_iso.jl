# Linear Isotropic Covariance Function

@doc """
# Description
Constructor for the isotropic linear kernel (covariance)

k(x,x') = xᵀx'/ℓ²
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
""" ->
type LinIso <: Kernel
    ll::Float64      # Log of Length scale 
    LinIso(ll::Float64) = new(ll)
end

function cov(lin::LinIso, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(lin.ll)
    K = dot(x,y)/ell^2
    return K
end

function cov(lin::LinIso, X::Matrix{Float64}, data::EmptyData)
    ell = exp(lin.ll)
    return ((1/ell^2).*X') * X
end

get_params(lin::LinIso) = Float64[lin.ll]
get_param_names(lin::LinIso) = [:ll]
num_params(lin::LinIso) = 1

function set_params!(lin::LinIso, hyp::Vector{Float64})
    length(hyp) == 1 || throw(ArgumentError("Linear isotropic kernel only has one parameter"))
    lin.ll = hyp[1]
end

function grad_kern(lin::LinIso, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(lin.ll)
    
    dK_ell = -2.0*dot(x,y)/ell^2
    dK_theta = [dK_ell]
    return dK_theta
end

function grad_stack!(stack::AbstractArray, lin::LinIso, X::Matrix{Float64}, data::EmptyData)
    ell = exp(lin.ll)
    stack[:,:,1] = ((-2.0/ell^2).*X') * X
end
