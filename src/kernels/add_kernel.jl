type AddKernel{K1<:Kernel, K2<:Kernel} <: PairKernel{K1, K2}
    kleft::K1
    kright::K2
end
leftkern(addkern::AddKernel) = addkern.kleft
rightkern(addkern::AddKernel) = addkern.kright


subkernels(addkern::AddKernel) = [addkern.kleft, addkern.kright]
get_param_names(addkern::AddKernel) = composite_param_names(subkernels(addkern), :ak)

function cov(sk::AddKernel{K1, K2}, x::V1, y::V2) where {V1<:VecF64, V2<:VecF64, K1, K2}
    cov(sk.kleft, x, y) + cov(sk.kright, x, y)
end

function addcov!{M<:MatF64}(s::MatF64, addkern::AddKernel, X::M, data::PairData)
    addcov!(s, addkern.kleft, X, data.data1)
    addcov!(s, addkern.kright, X, data.data2)
    return s
end
function cov!{M<:MatF64}(s::MatF64, addkern::AddKernel, X::M, data::PairData)
    cov!(s, addkern.kleft, X, data.data1)
    addcov!(s, addkern.kright, X, data.data2)
end
function cov{M<:MatF64}(addkern::AddKernel, X::M, data::PairData)
    d, nobsv = size(X)
    s = zeros(nobsv, nobsv)
    cov!(s, addkern, X, data)
end
    
@inline function dKij_dθp(addkern::AddKernel{K1, K2}, X::M, i::Int, j::Int, p::Int, dim::Int) where {M<:MatF64, K1, K2}
    np = num_params(addkern.kleft)
    if p<=np
        return dKij_dθp(addkern.kleft, X, i, j, p, dim)
    else
        return dKij_dθp(addkern.kright, X, i, j, p-np, dim)
    end
end
@inline function dKij_dθp{M<:MatF64}(addkern::AddKernel, X::M, data::PairData, i::Int, j::Int, p::Int, dim::Int)
    np = num_params(addkern.kleft)
    if p<=np
        return dKij_dθp(addkern.kleft, X, data.data1, i, j, p, dim)
    else
        return dKij_dθp(addkern.kright, X, data.data2, i, j, p-np, dim)
    end
end

function grad_slice!{M<:MatF64}(dK::MatF64, addkern::AddKernel, X::M, data::PairData, p::Int)
    np = num_params(addkern.kleft)
    if p<=np
        return grad_slice!(dK, addkern.kleft, X, data.data1, p)
    else
        return grad_slice!(dK, addkern.kright, X, data.data2, p-np)
    end
end
        
# Addition operators
+(kleft::Kernel, kright::Kernel) = AddKernel(kleft,kright)
