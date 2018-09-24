type SumKernel{K1<:Kernel, K2<:Kernel} <: PairKernel{K1, K2}
    kleft::K1
    kright::K2
end
leftkern(sumkern::SumKernel) = sumkern.kleft
rightkern(sumkern::SumKernel) = sumkern.kright

subkernels(sumkern::SumKernel) = [sumkern.kleft, sumkern.kright]
get_param_names(sumkern::SumKernel) = composite_param_names(subkernels(sumkern), :ak)

function cov(sk::SumKernel, x::VecF64, y::VecF64)
    cov(sk.kleft, x, y) + cov(sk.kright, x, y)
end

@inline cov_ij(k::K, X::MatF64, i::Int, j::Int, dim::Int) where {K<:SumKernel} = cov_ij(k.kleft, X, i, j, dim) + cov_ij(k.kright, X, i, j, dim)
@inline cov_ij(k::K, X::MatF64, data::PairData, i::Int, j::Int, dim::Int) where {K<:SumKernel} = cov_ij(k.kleft, X, data.data1, i, j, dim) + cov_ij(k.kright, X, data.data2, i, j, dim)
    
@inline function dKij_dθp(sumkern::SumKernel, X::MatF64, i::Int, j::Int, p::Int, dim::Int)
    np = num_params(sumkern.kleft)
    if p<=np
        return dKij_dθp(sumkern.kleft, X, i, j, p, dim)
    else
        return dKij_dθp(sumkern.kright, X, i, j, p-np, dim)
    end
end
@inline function dKij_dθp(sumkern::SumKernel, X::MatF64, data::PairData, i::Int, j::Int, p::Int, dim::Int)
    np = num_params(sumkern.kleft)
    if p<=np
        return dKij_dθp(sumkern.kleft, X, data.data1, i, j, p, dim)
    else
        return dKij_dθp(sumkern.kright, X, data.data2, i, j, p-np, dim)
    end
end
@inline @inbounds function dKij_dθ!(dK::VecF64, sumkern::SumKernel, X::MatF64, data::PairData, i::Int, j::Int, dim::Int, npars::Int)
    npright = num_params(sumkern.kright) 
    npleft = num_params(sumkern.kleft)
    dKij_dθ!(dK, sumkern.kright, X, data.data2, i, j, dim, npars-npleft)
    for ipar in npright:-1:1
        dK[npleft+ipar] = dK[ipar]
    end
    dKij_dθ!(dK,  sumkern.kleft,  X, data.data1, i, j, dim, npleft)
end

function grad_slice!(dK::MatF64, sumkern::SumKernel, X::MatF64, data::PairData, p::Int)
    np = num_params(sumkern.kleft)
    if p<=np
        return grad_slice!(dK, sumkern.kleft, X, data.data1, p)
    else
        return grad_slice!(dK, sumkern.kright, X, data.data2, p-np)
    end
end

# Addition operators
Base.+(kleft::Kernel, kright::Kernel) = SumKernel(kleft,kright)
