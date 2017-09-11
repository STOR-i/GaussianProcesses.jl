type ProdKernel <: CompositeKernel
    kerns::Vector{Kernel}
    ProdKernel(args::Vararg{Kernel}) = new(collect(args))
end

subkernels(pk::ProdKernel) = pk.kerns
get_param_names(pk::ProdKernel) = composite_param_names(pk.kerns, :pk)

cov{V1<:VecF64,V2<:VecF64}(pk::ProdKernel, x::V1, y::V2) = prod(cov(k, x, y) for k in subkernels(pk))

function cov{M<:MatF64}(prodkern::ProdKernel, X::M)
    d, nobsv = size(X)
    p = ones(nobsv, nobsv)
    for k in prodkern.kerns
        p[:,:] .*= cov(k, X)
    end
    return p
end

function cov!{M<:MatF64}(s::MatF64, prodkern::ProdKernel, X::M, data::CompositeData)
    s[:,:] = 1.0
    for (ikern,kern) in enumerate(prodkern.kerns)
        multcov!(s, kern, X, data.datadict[data.keys[ikern]])
    end
    return s
end

function cov{M<:MatF64}(prodkern::ProdKernel, X::M, data::CompositeData)
    d, nobsv = size(X)
    s = Array{Float64}( nobsv, nobsv)
    cov!(s, prodkern, X, data)
end

@inline function dKij_dθp{M<:MatF64}(prodkern::ProdKernel, X::M, i::Int, j::Int, p::Int, dim::Int)
    cKij = cov(prodkern, X[:,i], X[:,j])
    s=0
    for k in prodkern.kerns
        np = num_params(k)
        if p<=np+s
            cKk = cov(k, X[:,i], X[:,j])
            return dKij_dθp(k,X,i,j,p-s,dim)*cKij/cKk
        end
        s += np
    end
end
@inline function dKij_dθp{M<:MatF64}(prodkern::ProdKernel, X::M, data::CompositeData, i::Int, j::Int, p::Int, dim::Int)
    cKij = cov(prodkern, X[:,i], X[:,j])
    s=0
    for (ikern,kern) in enumerate(prodkern.kerns)
        np = num_params(kern)
        if p<=np+s
            cKk = cov(kern, X[:,i], X[:,j])
            return dKij_dθp(kern, X, data.datadict[data.keys[ikern]],i,j,p-s,dim)*cKij/cKk
        end
        s += np
    end
end
function grad_slice!{M1<:MatF64, M2<:MatF64}(
    dK::M1, prodkern::ProdKernel, X::M2, data::CompositeData, iparam::Int)
    istart=0
    for (ikern,kern) in enumerate(prodkern.kerns)
        np = num_params(kern)
        if istart<iparam<=np+istart
            grad_slice!(dK, kern, X, data.datadict[data.keys[ikern]],iparam-istart)
            break
        end
        istart += np
    end
    istart=0
    for (ikern,kern) in enumerate(prodkern.kerns)
        np = num_params(kern)
        if !(istart<iparam<=np+istart)
            multcov!(dK, kern, X, data.datadict[data.keys[ikern]])
        end
        istart += np
    end

    return dK
end

# Multiplication operators
*(k1::ProdKernel, k2::Kernel) = ProdKernel(k1.kerns..., k2)
*(k1::ProdKernel, k2::ProdKernel) = ProdKernel(k1.kerns..., k2.kerns...)
*(k1::Kernel, k2::Kernel) = ProdKernel(k1,k2)
*(k1::Kernel, k2::ProdKernel) = ProdKernel(k1, k2.kerns...)
