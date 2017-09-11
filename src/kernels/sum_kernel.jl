type SumKernel <: CompositeKernel
    kerns::Vector{Kernel}
    SumKernel(args::Vararg{Kernel}) = new(collect(args))
end

subkernels(sumkern::SumKernel) = sumkern.kerns
get_param_names(sumkern::SumKernel) = composite_param_names(sumkern.kerns, :pk)

cov{V1<:VecF64,V2<:VecF64}(sk::SumKernel, x::V1, y::V2) = sum(cov(k, x, y) for k in subkernels(sk))


function addcov!{M<:MatF64}(s::MatF64, sumkern::SumKernel, X::M, data::CompositeData)
    for (ikern,kern) in enumerate(sumkern.kerns)
        addcov!(s, kern, X, data.datadict[data.keys[ikern]])
    end
    return s
end
function cov!{M<:MatF64}(s::MatF64, sumkern::SumKernel, X::M, data::CompositeData)
    s[:,:] = 0.0
    addcov!(s, sumkern, X, data)
end
function cov{M<:MatF64}(sumkern::SumKernel, X::M, data::CompositeData)
    d, nobsv = size(X)
    s = zeros(nobsv, nobsv)
    cov!(s, sumkern, X, data)
end
    
@inline function dKij_dθp{M<:MatF64}(sumkern::SumKernel, X::M, i::Int, j::Int, p::Int, dim::Int)
    s=0
    for k in sumkern.kerns
        np = num_params(k)
        if p<=np+s
            return dKij_dθp(k, X, i,j,p-s,dim)
        end
        s += np
    end
end
@inline function dKij_dθp{M<:MatF64}(sumkern::SumKernel, X::M, data::CompositeData, i::Int, j::Int, p::Int, dim::Int)
    s=0
    for (ikern,kern) in enumerate(sumkern.kerns)
        np = num_params(kern)
        if p<=np+s
            return dKij_dθp(kern, X, data.datadict[data.keys[ikern]],i,j,p-s,dim)
        end
        s += np
    end
end

function grad_slice!{M<:MatF64}(dK::MatF64, sumkern::SumKernel, X::M, data::CompositeData, p::Int)
    s=0
    for (ikern,kern) in enumerate(sumkern.kerns)
        np = num_params(kern)
        if p<=np+s
            return grad_slice!(dK, kern, X, data.datadict[data.keys[ikern]],p-s)
        end
        s += np
    end
    return dK
end
        
# Addition operators
+(k1::SumKernel, k2::Kernel) = SumKernel(k1.kerns..., k2)
+(k1::SumKernel, k2::SumKernel) = SumKernel(k1.kerns..., k2.kerns...)
+(k1::Kernel, k2::Kernel) = SumKernel(k1,k2)
+(k1::Kernel, k2::SumKernel) = SumKernel(k1, k2.kerns...)
