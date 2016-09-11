#Linear ARD Covariance Function


@doc """
# Description
Constructor for the ARD linear kernel (covariance)

k(x,x') = xᵀL⁻²x', where L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
""" ->
type LinArd <: Kernel
    ℓ::Vector{Float64}      # Length Scale
    dim::Int                 # Number of hyperparameters
    LinArd(ll::Vector{Float64}) = new(exp(ll),size(ll,1))
end

function cov(lin::LinArd, x::Vector{Float64}, y::Vector{Float64})
    K = dot(x./lin.ℓ,y./lin.ℓ)
    return K
end

type LinArdData <: KernelData
    XtX_d::Array{Float64,3}
end

function KernelData(k::LinArd, X::Matrix{Float64})
    dim,n=size(X)
    XtX_d = Array(Float64,n,n,dim)
    for d in 1:dim
        XtX_d[:,:,d] = view(X,d,:) * view(X,d,:)'
        Base.LinAlg.copytri!(view(XtX_d,:,:,d), 'U')
    end
    LinArdData(XtX_d)
end
kernel_data_key(k::LinArd, X::Matrix{Float64}) = :LinArdData
function cov(lin::LinArd, X::Matrix{Float64})
    K = (X./lin.ℓ)' * (X./lin.ℓ)
    Base.LinAlg.copytri!(K, 'U')
    return K
end
function cov!(cK::AbstractMatrix, lin::LinArd, X::Matrix{Float64}, data::LinArdData)
    dim,n=size(X)
    cK[:,:] = 0.0
    for d in 1:dim
        Base.LinAlg.axpy!(1/lin.ℓ[d]^2, view(data.XtX_d,:,:,d), cK)
    end
    return cK
end
function cov(lin::LinArd, X::Matrix{Float64}, data::LinArdData)
    nobsv=size(X,2)
    K = zeros(Float64,nobsv,nobsv)
    cov!(K,lin,X,data)
    return K
end

get_params(lin::LinArd) = log(lin.ℓ)
get_param_names(lin::LinArd) = get_param_names(lin.ℓ, :ll)
num_params(lin::LinArd) = lin.dim

function set_params!(lin::LinArd, hyp::Vector{Float64})
    length(hyp) == lin.dim || throw(ArgumentError("Linear ARD kernel has $(lin.dim) parameters"))
    lin.ℓ = exp(hyp)
end

@inline dk_dll(lin::LinArd, xy::Float64, d::Int) = -2.0*xy/lin.ℓ[d]^2
@inline function dKij_dθp(lin::LinArd, X::Matrix{Float64}, i::Int, j::Int, p::Int, dim::Int)
    if p<=dim
        return dk_dll(lin, dotijp(X,i,j,p), p)
    else
        return NaN
    end
end
@inline function dKij_dθp(lin::LinArd, X::Matrix{Float64}, data::LinArdData, i::Int, j::Int, p::Int, dim::Int)
    if p<=dim
        return dk_dll(lin, data.XtX_d[i,j,p],p)
    else
        return NaN
    end
end
