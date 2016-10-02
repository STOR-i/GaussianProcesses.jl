type ProdKernel <: Kernel
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
function KernelData{M<:MatF64}(prodkern::ProdKernel, X::M)
    datadict = Dict{Symbol, KernelData}()
    datakeys = Symbol[]
    for k in prodkern.kerns
        data_type = kernel_data_key(k, X)
        if !haskey(datadict, data_type)
            datadict[data_type] = KernelData(k, X)
        end
        push!(datakeys, data_type)
    end
    SumData(datadict, datakeys)
end
kernel_data_key{M<:MatF64}(prodkern::ProdKernel, X::M) = :SumData

function show(io::IO, prodkern::ProdKernel, depth::Int = 0)
    pad = repeat(" ", 2 * depth)
    println(io, "$(pad)Type: $(typeof(prodkern))")
    for k in prodkern.kerns
        show(io, k, depth+1)
    end
end

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

function cov!{M<:MatF64}(s::MatF64, prodkern::ProdKernel, X::M, data::SumData)
    s[:,:] = 1.0
    for (ikern,kern) in enumerate(prodkern.kerns)
        multcov!(s, kern, X, data.datadict[data.keys[ikern]])
    end
    return s
end
function cov{M<:MatF64}(prodkern::ProdKernel, X::M, data::SumData)
    d, nobsv = size(X)
    s = Array(Float64, nobsv, nobsv)
    cov!(s, prodkern, X, data)
end

function get_params(prodkern::ProdKernel)
    p = Array(Float64, 0)
    for k in prodkern.kerns
        append!(p, get_params(k))
    end
    p
end

get_param_names(prodkern::ProdKernel) = composite_param_names(prodkern.kerns, :pk)

function num_params(prodkern::ProdKernel)
    n = 0
    for k in prodkern.kerns
        n += num_params(k)
    end
    n
end

function set_params!(prodkern::ProdKernel, hyp::Vector{Float64})
    i, n = 1, num_params(prodkern)
    length(hyp) == num_params(prodkern) || throw(ArgumentError("ProdKernel object requires $(n) hyperparameters"))
    for k in prodkern.kerns
        np = num_params(k)
        set_params!(k, hyp[i:(i+np-1)])
        i += np
    end
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
@inline function dKij_dθp{M<:MatF64}(prodkern::ProdKernel, X::M, data::SumData, i::Int, j::Int, p::Int, dim::Int)
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
