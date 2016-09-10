# Linear Isotropic Covariance Function

@doc """
# Description
Constructor for the isotropic linear kernel (covariance)

k(x,x') = xᵀx'/ℓ²
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
""" ->
type LinIso <: Kernel
    ℓ2::Float64      # Log of Length scale 
    LinIso(ll::Float64) = new(exp(2.0*ll))
end

type LinIsoData <: KernelData
    XtX::Matrix{Float64}
end

function KernelData(k::LinIso, X::Matrix{Float64})
    XtX=X'*X
    Base.LinAlg.copytri!(XtX, 'U') # make sure it's symmetric
    LinIsoData(XtX)
end
kernel_data_key(k::LinIso, X::Matrix{Float64}) = :LinIsoData

_cov(lin::LinIso, xTy) = xTy ./ lin.ℓ2
function cov(lin::LinIso, x::Vector{Float64}, y::Vector{Float64})
    K = _cov(lin, dot(x,y))
    return K
end

function cov(lin::LinIso, X::Matrix{Float64}, data::LinIsoData)
    K = _cov(lin, data.XtX)
    return K
end
function cov!(cK::AbstractMatrix, lin::LinIso, X::Matrix{Float64}, data::LinIsoData)
    copy!(cK, data.XtX)
    cK[:,:] ./= lin.ℓ2
    return cK
end

get_params(lin::LinIso) = Float64[log(lin.ℓ2)/2.0]
get_param_names(lin::LinIso) = [:ll]
num_params(lin::LinIso) = 1

function set_params!(lin::LinIso, hyp::Vector{Float64})
    length(hyp) == 1 || throw(ArgumentError("Linear isotropic kernel only has one parameter"))
    lin.ℓ2 = exp(2.0*hyp[1])
end

@inline dk_dll(lin::LinIso, xTy::Float64) = -2.0*_cov(lin,xTy)
@inline function dKij_dθp(lin::LinIso, X::Matrix{Float64}, i::Int, j::Int, p::Int, dim::Int)
    if p==1
        return dk_dll(lin, dotij(X,i,j,dim))
    else
        return NaN
    end
end
@inline function dKij_dθp(lin::LinIso, X::Matrix{Float64}, data::LinIsoData, i::Int, j::Int, p::Int, dim::Int)
    if p==1
        return dk_dll(lin, data.XtX[i,j])
    else
        return NaN
    end
end
