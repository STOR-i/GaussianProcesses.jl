struct ProdKernel{K1<:Kernel, K2<:Kernel} <: PairKernel{K1,K2}
    kleft::K1
    kright::K2
end
leftkern(prodkern::ProdKernel) = prodkern.kleft
rightkern(prodkern::ProdKernel) = prodkern.kright

get_param_names(prodkern::ProdKernel) = composite_param_names(components(prodkern), :pk)

function cov(sk::ProdKernel, x::AbstractVector, y::AbstractVector)
    cov(sk.kleft, x, y) * cov(sk.kright, x, y)
end

@inline cov_ij(k::ProdKernel, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int) = cov_ij(k.kleft, X1, X2, i, j, dim) * cov_ij(k.kright, X1, X2, i, j, dim)
@inline cov_ij(k::ProdKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::PairData, i::Int, j::Int, dim::Int) = cov_ij(k.kleft, X1, X2, data.data1, i, j, dim) * cov_ij(k.kright, X1, X2, data.data2, i, j, dim)

@inline function dKij_dθp(prodkern::ProdKernel, X::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int)
    np = num_params(prodkern.kleft)
    if p<=np
        cK_other = cov_ij(prodkern.kright, X, X, i, j, dim)
        return dKij_dθp(prodkern.kleft, X, i,j,p,dim)*cK_other
    else
        cK_other = cov_ij(prodkern.kleft, X, X, i, j, dim)
        return dKij_dθp(prodkern.kright, X, i,j,p-np,dim)*cK_other
    end
end
@inline function dKij_dθp(prodkern::ProdKernel, X::AbstractMatrix, data::PairData, i::Int, j::Int, p::Int, dim::Int)
    np = num_params(prodkern.kleft)
    if p<=np
        cK_other = cov_ij(prodkern.kright, X, X, i, j, dim)
        dKij_sub = dKij_dθp(prodkern.kleft, X, data.data1,i,j,p,dim)
        return dKij_sub * cK_other
    else
        cK_other = cov_ij(prodkern.kleft, X, X, i, j, dim)
        dKij_sub = dKij_dθp(prodkern.kright, X, data.data2,i,j,p-np,dim)
        return dKij_sub * cK_other
    end
end
@inline @inbounds function dKij_dθ!(dK::AbstractVector, prodkern::ProdKernel, X::AbstractMatrix,
                                    i::Int, j::Int, dim::Int, npars::Int)
    cov_left  = cov_ij(prodkern.kleft,  X, X, i, j, dim)
    cov_right = cov_ij(prodkern.kright, X, X, i, j, dim)
    npright = num_params(prodkern.kright)
    npleft = num_params(prodkern.kleft)
    dKij_dθ!(dK, prodkern.kright, X, i, j, dim, npars-npleft)
    for ipar in npright:-1:1
        dK[npleft+ipar] = dK[ipar]*cov_left
    end
    dKij_dθ!(dK,  prodkern.kleft,  X, i, j, dim, npleft)
    for ipar in 1:npleft
        dK[ipar] *= cov_right
    end
end
@inline @inbounds function dKij_dθ!(dK::AbstractVector, prodkern::ProdKernel, X::AbstractMatrix, data::PairData,
                                    i::Int, j::Int, dim::Int, npars::Int)
    cov_left  = cov_ij(prodkern.kleft,  X, X, data.data1, i, j, dim)
    cov_right = cov_ij(prodkern.kright, X, X, data.data2, i, j, dim)
    npright = num_params(prodkern.kright)
    npleft = num_params(prodkern.kleft)
    dKij_dθ!(dK, prodkern.kright, X, data.data2, i, j, dim, npars-npleft)
    for ipar in npright:-1:1
        dK[npleft+ipar] = dK[ipar]*cov_left
    end
    dKij_dθ!(dK,  prodkern.kleft,  X, data.data1, i, j, dim, npleft)
    for ipar in 1:npleft
        dK[ipar] *= cov_right
    end
end

# Multiplication operators
Base.:*(kleft::Kernel, kright::Kernel) = ProdKernel(kleft,kright)
