struct ProdKernel <: CompositeKernel
    kerns::Vector{Kernel}
    ProdKernel(args::Vararg{Kernel}) = new(collect(args))
end

subkernels(pk::ProdKernel) = pk.kerns
get_param_names(pk::ProdKernel) = composite_param_names(pk.kerns, :pk)

Statistics.cov(pk::ProdKernel, x::VecF64, y::VecF64) = prod(cov(k, x, y) for k in subkernels(pk))

function Statistics.cov(prodkern::ProdKernel, X::MatF64)
    d, nobsv = size(X)
    p = ones(nobsv, nobsv)
    for k in prodkern.kerns
        p .*= cov(k, X)
    end
    return p
end

function cov!(s::MatF64, prodkern::ProdKernel, X::MatF64, data::CompositeData)
    fill!(s, 1)
    for (ikern,kern) in enumerate(prodkern.kerns)
        multcov!(s, kern, X, data.datadict[data.keys[ikern]])
    end
    return s
end

function Statistics.cov(prodkern::ProdKernel, X::MatF64, data::CompositeData)
    d, nobsv = size(X)
    s = Array{Float64}(undef, nobsv, nobsv)
    cov!(s, prodkern, X, data)
end

#=# This function is extremely inefficient=#
#=function grad_kern(prodkern::ProdKernel, x::Vector{Float64}, y::Vector{Float64})=#
#=     dk = Array{Float64}( 0)=#
#=      for k in prodkern.kerns=#
#=          p = 1.0=#
#=          for j in prodkern.kerns[find(k.!=prodkern.kerns)]=#
#=              p = p.*cov(j, x, y)=#
#=          end=#
#=        append!(dk,grad_kern(k, x, y).*p) =#
#=      end=#
#=    dk=#
#=end=#

@inline function dKij_dθp(prodkern::ProdKernel, X::MatF64, i::Int, j::Int, p::Int, dim::Int)
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
@inline function dKij_dθp(prodkern::ProdKernel, X::MatF64, data::CompositeData, i::Int, j::Int, p::Int, dim::Int)
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
function grad_slice!(dK::MatF64, prodkern::ProdKernel, X::MatF64, data::CompositeData, iparam::Int)
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
Base.:*(k1::ProdKernel, k2::Kernel) = ProdKernel(k1.kerns..., k2)
Base.:*(k1::ProdKernel, k2::ProdKernel) = ProdKernel(k1.kerns..., k2.kerns...)
Base.:*(k1::Kernel, k2::Kernel) = ProdKernel(k1,k2)
Base.:*(k1::Kernel, k2::ProdKernel) = ProdKernel(k1, k2.kerns...)
