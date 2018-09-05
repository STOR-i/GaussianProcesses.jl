struct ProdKernel{T<:NTuple{N,Kernel} where N} <: CompositeKernel
    kernels::T
end

ProdKernel(kernels::Kernel...) = ProdKernel(kernels)

get_param_names(pk::ProdKernel) = composite_param_names(components(pk), :pk)

Statistics.cov(pk::ProdKernel, x::VecF64, y::VecF64) = prod(cov(k, x, y) for k in components(pk))

function Statistics.cov(prodkern::ProdKernel, X::MatF64)
    d, nobsv = size(X)
    p = ones(nobsv, nobsv)
    for k in components(prodkern)
        p .*= cov(k, X)
    end
    return p
end

function cov!(s::MatF64, prodkern::ProdKernel, X::MatF64, data::CompositeData)
    fill!(s, 1)
    for (ikern,kern) in enumerate(components(prodkern))
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
#=     dk = Array{Float64}(undef, 0)=#
#=      for k in components(prodkern)=#
#=          p = 1.0=#
#=          for j in components(prodkern)[find(k.!=components(prodkern))]=#
#=              p = p.*cov(j, x, y)=#
#=          end=#
#=        append!(dk,grad_kern(k, x, y).*p) =#
#=      end=#
#=    dk=#
#=end=#

@inline function dKij_dθp(prodkern::ProdKernel, X::MatF64, i::Int, j::Int, p::Int, dim::Int)
    Xi, Xj = view(X, :, i), view(X, :, j)
    cKij = cov(prodkern, Xi, Xj)
    s=0
    for k in components(prodkern)
        t = s + num_params(k)
        if p <= t
            cKk = cov(k, Xi, Xj)
            return dKij_dθp(k,X,i,j,p-s,dim)*cKij/cKk
        end
        s = t
    end
end
@inline function dKij_dθp(prodkern::ProdKernel, X::MatF64, data::CompositeData, i::Int, j::Int, p::Int, dim::Int)
    Xi, Xj = view(X, :, i), view(X, :, j)
    cKij = cov(prodkern, Xi, Xj)
    s=0
    for (ikern,kern) in enumerate(components(kernels))
        t = s + num_params(kern)
        if p <= t
            cKk = cov(kern, Xi, Xj)
            return dKij_dθp(kern, X, data.datadict[data.keys[ikern]],i,j,p-s,dim)*cKij/cKk
        end
        s = t
    end
end
function grad_slice!(dK::MatF64, prodkern::ProdKernel, X::MatF64, data::CompositeData, iparam::Int)
    istart=0
    for (ikern,kern) in enumerate(components(prodkern))
        istop = istart + num_params(kern)
        if istart < iparam <= istop
            grad_slice!(dK, kern, X, data.datadict[data.keys[ikern]],iparam-istart)
            break
        end
        istart = istop
    end
    istart=0
    for (ikern,kern) in enumerate(components(prodkern))
        istop = istart + num_params(kern)
        if iparam <= istart || istop < iparam
            multcov!(dK, kern, X, data.datadict[data.keys[ikern]])
        end
        istart = istop
    end

    return dK
end

# Multiplication operators
Base.:*(k1::ProdKernel, k2::Kernel) = ProdKernel(k1.kernels..., k2)
Base.:*(k1::ProdKernel, k2::ProdKernel) = ProdKernel(k1.kernels..., k2.kernels...)
Base.:*(k1::Kernel, k2::Kernel) = ProdKernel(k1, k2)
Base.:*(k1::Kernel, k2::ProdKernel) = ProdKernel(k1, k2.kernels...)
