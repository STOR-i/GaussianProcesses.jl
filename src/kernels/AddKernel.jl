type AddKernel{K1<:Kernel, K2<:Kernel} <: CompositeKernel
    k1::K1
    k2::K2
end

type PairData{KD1 <: KernelData, KD2 <: KernelData} <: KernelData
    data1::KD1
    data2::KD2
end

subkernels(addkern::AddKernel) = [addkern.k1, addkern.k2]
get_param_names(addkern::AddKernel) = composite_param_names(subkernels(addkern), :ak)

function cov(sk::AddKernel{K1, K2}, x::V1, y::V2) where {V1<:VecF64, V2<:VecF64, K1, K2}
    cov(sk.k1, x, y) + cov(sk.k2, x, y)
end

function addcov!{M<:MatF64}(s::MatF64, addkern::AddKernel, X::M, data::PairData)
    addcov!(s, addkern.k1, X, data.data1)
    addcov!(s, addkern.k2, X, data.data2)
    return s
end
function cov!{M<:MatF64}(s::MatF64, addkern::AddKernel, X::M, data::PairData)
    cov!(s, addkern.k1, X, data.data1)
    addcov!(s, addkern.k2, X, data.data2)
end
function cov{M<:MatF64}(addkern::AddKernel, X::M, data::PairData)
    d, nobsv = size(X)
    s = zeros(nobsv, nobsv)
    cov!(s, addkern, X, data)
end
    
function grad_kern{V1<:VecF64,V2<:VecF64}(addkern::AddKernel, x::V1, y::V2)
    dk = Array{Float64}(0)
    for k in addkern.kerns
        append!(dk,grad_kern(k, x, y))
    end
    dk
end

@inline function dKij_dθp(addkern::AddKernel{K1, K2}, X::M, i::Int, j::Int, p::Int, dim::Int) where {M<:MatF64, K1, K2}
    s=0
    for k in (addkern.k1, addkern.k2)
        np = num_params(k)
        if p<=np+s
            return dKij_dθp(k, X, i,j,p-s,dim)
        end
        s += np
    end
end
@inline function dKij_dθp{M<:MatF64}(addkern::AddKernel, X::M, data::PairData, i::Int, j::Int, p::Int, dim::Int)
    s=0
    np = num_params(addkern.k1)
    if p<=np+s
        return dKij_dθp(addkern.k1, X, data.data1,i,j,p-s,dim)
    else
        s += np
        return dKij_dθp(addkern.k2, X, data.data2,i,j,p-s,dim)
    end
end

function grad_slice!{M<:MatF64}(dK::MatF64, addkern::AddKernel, X::M, data::PairData, p::Int)
    s=0
    np = num_params(addkern.k1)
    if p<=np+s
        return grad_slice!(dK, addkern.k1, X, data.data1, p-s)
    else
        s += np
        return grad_slice!(dK, addkern.k2, X, data.data2, p-s)
    end
end
        
function KernelData(addkern::AddKernel, X::M) where {M<:MatF64}
    # this is a bit broken:
    if kernel_data_key(addkern.k1, X) == kernel_data_key(addkern.k2, X)
        kdata = KernelData(addkern.k1, X)
        return PairData(kdata, kdata)
    else
        return PairData(
                KernelData(addkern.k1, X),
                KernelData(addkern.k2, X)
               )
    end
end

function kernel_data_key{M<:MatF64}(compkern::CompositeKernel, X::M)
    @sprintf("AddData:%s+%s", kernel_data_key(addkern.k1, X), kernel_data_key(addkern.k2, X))
end
# Addition operators
+(k1::Kernel, k2::Kernel) = AddKernel(k1,k2)
