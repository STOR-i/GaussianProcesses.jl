type ProdKernel <: CompositeKernel
    kerns::Vector{Kernel}
    function ProdKernel(args...)
        kerns = Array(Kernel, length(args))
        for (i,k) in enumerate(args)
            isa(k, Kernel) || throw(ArgumentError("All arguments of ProdKernel must be Kernel objects"))
            kerns[i] = k
        end
        return new(kerns)
    end
end

subkernels(prodkern::ProdKernel) = prodkern.kerns
get_param_names(prodkern::ProdKernel) = composite_param_names(prodkern.kerns, :pk)

function cov{V1<:VecF64,V2<:VecF64}(prodkern::ProdKernel, x::V1, y::V2)
    p = 1.0
    for k in prodkern.kerns
        p *= cov(k, x, y)
    end
    return p
end

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
    s = Array(Float64, nobsv, nobsv)
    cov!(s, prodkern, X, data)
end

#=# This function is extremely inefficient=#
#=function grad_kern(prodkern::ProdKernel, x::Vector{Float64}, y::Vector{Float64})=#
#=     dk = Array(Float64, 0)=#
#=      for k in prodkern.kerns=#
#=          p = 1.0=#
#=          for j in prodkern.kerns[find(k.!=prodkern.kerns)]=#
#=              p = p.*cov(j, x, y)=#
#=          end=#
#=        append!(dk,grad_kern(k, x, y).*p) =#
#=      end=#
#=    dk=#
#=end=#

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
function *(k1::ProdKernel, k2::Kernel)
    kerns = [k1.kerns; k2]
    ProdKernel(kerns...)
end
function *(k1::ProdKernel, k2::ProdKernel)
    kerns = [k1.kerns; k2.kerns]
    ProdKernel(kerns...)
end
*(k1::Kernel, k2::Kernel) = ProdKernel(k1,k2)
*(k1::Kernel, k2::ProdKernel) = *(k2,k1)
