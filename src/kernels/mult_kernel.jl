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

@inline cov_ij(k::K, X::M, i::Int, j::Int, dim::Int) where {K<:MultKernel, M<:MatF64} = cov_ij(k.kleft, X, i, j, dim) * cov_ij(k.kright, X, i, j, dim)
@inline cov_ij(k::K, X::M, data::PairData, i::Int, j::Int, dim::Int) where {K<:MultKernel, M<:MatF64} = cov_ij(k.kleft, X, data.data1, i, j, dim) * cov_ij(k.kright, X, data.data2, i, j, dim)

@inline function dKij_dθp(multkern::MultKernel{K1, K2}, X::M, i::Int, j::Int, p::Int, dim::Int) where {M<:MatF64, K1, K2}
    np = num_params(multkern.kleft)
    if p<=np
        cK_other = cov_ij(multkern.kright, X, i, j, dim)
        return dKij_dθp(multkern.kleft, X, i,j,p,dim)*cK_other
    else
        cK_other = cov_ij(multkern.kleft, X, i, j, dim)
        return dKij_dθp(multkern.kright, X, i,j,p-np,dim)*cK_other
    end
end
@inline function dKij_dθp{M<:MatF64}(multkern::MultKernel, X::M, data::PairData, i::Int, j::Int, p::Int, dim::Int)
    np = num_params(multkern.kleft)
    if p<=np
        cK_other = cov_ij(multkern.kright, X, i, j, dim)
        dKij_sub = dKij_dθp(multkern.kleft, X, data.data1,i,j,p,dim)
        return dKij_sub * cK_other
    else
        cK_other = cov_ij(multkern.kleft, X, i, j, dim)
        dKij_sub = dKij_dθp(multkern.kright, X, data.data2,i,j,p-np,dim)
        return dKij_sub * cK_other
    end
end

# Multiplication operators
*(kleft::Kernel, kright::Kernel) = MultKernel(kleft,kright)
