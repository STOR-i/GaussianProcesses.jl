struct SumKernel <: CompositeKernel
    kerns::Vector{Kernel}
    SumKernel(args::Vararg{Kernel}) = new(collect(args))
end

subkernels(sumkern::SumKernel) = sumkern.kerns
get_param_names(sumkern::SumKernel) = composite_param_names(sumkern.kerns, :pk)

Statistics.cov(sk::SumKernel, x::VecF64, y::VecF64) = sum(cov(k, x, y) for k in subkernels(sk))


function addcov!(s::MatF64, sumkern::SumKernel, X::MatF64, data::CompositeData)
    for (ikern,kern) in enumerate(sumkern.kerns)
        addcov!(s, kern, X, data.datadict[data.keys[ikern]])
    end
    return s
end
function cov!(s::MatF64, sumkern::SumKernel, X::MatF64, data::CompositeData)
    fill!(s, 0)
    addcov!(s, sumkern, X, data)
end
function Statistics.cov(sumkern::SumKernel, X::MatF64, data::CompositeData)
    d, nobsv = size(X)
    s = zeros(nobsv, nobsv)
    cov!(s, sumkern, X, data)
end

function grad_kern(sumkern::SumKernel, x::VecF64, y::VecF64)
    dk = Array{Float64}(undef, 0)
    for k in sumkern.kerns
        append!(dk, grad_kern(k, x, y))
    end
    dk
end

@inline function dKij_dθp(sumkern::SumKernel, X::MatF64, i::Int, j::Int, p::Int, dim::Int)
    s=0
    for k in sumkern.kerns
        np = num_params(k)
        if p<=np+s
            return dKij_dθp(k, X, i,j,p-s,dim)
        end
        s += np
    end
end
@inline function dKij_dθp(sumkern::SumKernel, X::MatF64, data::CompositeData, i::Int, j::Int, p::Int, dim::Int)
    s=0
    for (ikern,kern) in enumerate(sumkern.kerns)
        np = num_params(kern)
        if p<=np+s
            return dKij_dθp(kern, X, data.datadict[data.keys[ikern]],i,j,p-s,dim)
        end
        s += np
    end
end

function grad_slice!(dK::MatF64, sumkern::SumKernel, X::MatF64, data::CompositeData, p::Int)
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
Base.:+(k1::SumKernel, k2::Kernel) = SumKernel(k1.kerns..., k2)
Base.:+(k1::SumKernel, k2::SumKernel) = SumKernel(k1.kerns..., k2.kerns...)
Base.:+(k1::Kernel, k2::Kernel) = SumKernel(k1,k2)
Base.:+(k1::Kernel, k2::SumKernel) = SumKernel(k1, k2.kerns...)
