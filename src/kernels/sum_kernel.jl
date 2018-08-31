struct SumKernel{T<:NTuple{N,Kernel} where N} <: CompositeKernel
    kernels::T
end

SumKernel(kernels::Kernel...) = SumKernel(kernels)

get_param_names(sumkern::SumKernel) = composite_param_names(components(sumkern), :sk)

Statistics.cov(sk::SumKernel, x::VecF64, y::VecF64) = sum(cov(k, x, y) for k in components(sk))

function addcov!(s::MatF64, sumkern::SumKernel, X::MatF64, data::CompositeData)
    for (ikern,kern) in enumerate(components(sumkern))
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
    for k in components(sumkern)
        append!(dk, grad_kern(k, x, y))
    end
    dk
end

@inline function dKij_dθp(sumkern::SumKernel, X::MatF64, i::Int, j::Int, p::Int, dim::Int)
    s = 0
    for k in components(sumkern)
        t = s + num_params(k)
        if p <= t
            return dKij_dθp(k, X, i,j,p-s,dim)
        end
        s = t
    end
end
@inline function dKij_dθp(sumkern::SumKernel, X::MatF64, data::CompositeData, i::Int, j::Int, p::Int, dim::Int)
    s = 0
    for (ikern,kern) in enumerate(components(sumkern))
        t = s + num_params(kern)
        if p <= t
            return dKij_dθp(kern, X, data.datadict[data.keys[ikern]],i,j,p-s,dim)
        end
        s = t
    end
end

function grad_slice!(dK::MatF64, sumkern::SumKernel, X::MatF64, data::CompositeData, p::Int)
    s = 0
    for (ikern,kern) in enumerate(components(sumkern))
        t = s + num_params(kern)
        if p <= t
            return grad_slice!(dK, kern, X, data.datadict[data.keys[ikern]],p-s)
        end
        s = t
    end
    return dK
end

# Addition operators
Base.:+(k1::SumKernel, k2::Kernel) = SumKernel(k1.kernels..., k2)
Base.:+(k1::SumKernel, k2::SumKernel) = SumKernel(k1.kernels..., k2.kernels...)
Base.:+(k1::Kernel, k2::Kernel) = SumKernel(k1, k2)
Base.:+(k1::Kernel, k2::SumKernel) = SumKernel(k1, k2.kernels...)
