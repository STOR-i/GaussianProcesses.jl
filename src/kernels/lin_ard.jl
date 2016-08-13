#Linear ARD Covariance Function


@doc """
# Description
Constructor for the ARD linear kernel (covariance)

k(x,x') = xᵀL⁻²x', where L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
""" ->
type LinArd <: Kernel
    ll::Vector{Float64}      # Log of Length scale
    dim::Int                 # Number of hyperparameters
    LinArd(ll::Vector{Float64}) = new(ll,size(ll,1))
end

function cov(lin::LinArd, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(lin.ll)
    K = dot(x./ell,y./ell)
    return K
end

function cov(lin::LinArd, X::Matrix{Float64}, data::EmptyData)
    ell = exp(lin.ll)
    return (X./ell)' * (X./ell)
end

get_params(lin::LinArd) = lin.ll
get_param_names(lin::LinArd) = get_param_names(lin.ll, :ll)
num_params(lin::LinArd) = lin.dim

function set_params!(lin::LinArd, hyp::Vector{Float64})
    length(hyp) == lin.dim || throw(ArgumentError("Linear ARD kernel has $(lin.dim) parameters"))
    lin.ll = hyp
end

function grad_kern(lin::LinArd, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(lin.ll)
    dK_ell = -2.0*(x./ell).*(y./ell)
    return dK_ell
end

function grad_stack!(stack::AbstractArray, lin::LinArd, X::Matrix{Float64}, data::EmptyData)
    ell = exp(lin.ll)
    d, nobsv = size(X)
    for j in 1:d
        stack[:,:,j] = ((-2.0/ell[j]^2).*vec(X[j,:])) * vec(X[j,:])'
    end
end
