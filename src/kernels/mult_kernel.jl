type MultKernel{K1<:Kernel, K2<:Kernel} <: PairKernel{K1,K2}
    kleft::K1
    kright::K2
end
leftkern(multkern::MultKernel) = multkern.kleft
rightkern(multkern::MultKernel) = multkern.kright

subkernels(multkern::MultKernel) = [multkern.kleft, multkern.kright]
get_param_names(multkern::MultKernel) = composite_param_names(subkernels(multkern), :ak)

function cov(sk::MultKernel{K1, K2}, x::V1, y::V2) where {V1<:VecF64, V2<:VecF64, K1, K2}
    cov(sk.kleft, x, y) * cov(sk.kright, x, y)
end

function multcov!{M<:MatF64}(s::MatF64, multkern::MultKernel, X::M, data::PairData)
    multcov!(s, multkern.kleft, X, data.data1)
    multcov!(s, multkern.kright, X, data.data2)
    return s
end
function cov!{M<:MatF64}(s::MatF64, multkern::MultKernel, X::M, data::PairData)
    cov!(s, multkern.kleft, X, data.data1)
    multcov!(s, multkern.kright, X, data.data2)
end
function cov{M<:MatF64}(multkern::MultKernel, X::M, data::PairData)
    d, nobsv = size(X)
    s = zeros(nobsv, nobsv)
    cov!(s, multkern, X, data)
end
    
@inline function dKij_dθp(multkern::MultKernel{K1, K2}, X::M, i::Int, j::Int, p::Int, dim::Int) where {M<:MatF64, K1, K2}
    np = num_params(multkern.kleft)
    if p<=np
        cK_other = cov(multkern.kright, @view(X[:,i]), @view(X[:,j]))
        return dKij_dθp(multkern.kleft, X, i,j,p,dim)*cK_other
    else
        cK_other = cov(multkern.kleft, @view(X[:,i]), @view(X[:,j]))
        return dKij_dθp(multkern.kright, X, i,j,p-np,dim)*cK_other
    end
end
@inline function dKij_dθp{M<:MatF64}(multkern::MultKernel, X::M, data::PairData, i::Int, j::Int, p::Int, dim::Int)
    np = num_params(multkern.kleft)
    if p<=np
        cK_other = cov(multkern.kright, @view(X[:,i]), @view(X[:,j]))
        dKij_sub = dKij_dθp(multkern.kleft, X, data.data1,i,j,p,dim)
        return dKij_sub * cK_other
    else
        cK_other = cov(multkern.kleft, @view(X[:,i]), @view(X[:,j]))
        dKij_sub = dKij_dθp(multkern.kright, X, data.data2,i,j,p-np,dim)
        return dKij_sub * cK_other
    end
end

function grad_slice!{M<:MatF64}(dK::MatF64, multkern::MultKernel, X::M, data::PairData, p::Int)
    np = num_params(multkern.kleft)
    if p<=np
        grad_slice!(dK, multkern.kleft, X, data.data1, p)
        multcov!(dK, multkern.kright, X, data.data2)
        return dK
    else
        grad_slice!(dK, multkern.kright, X, data.data2, p-np)
        multcov!(dK, multkern.kleft, X, data.data1)
        return dK
    end
end

# Multiplication operators
*(kleft::Kernel, kright::Kernel) = MultKernel(kleft,kright)
