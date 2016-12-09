type SumKernel <: Kernel
    kerns::Vector{Kernel}
    function SumKernel(args...)
        kerns = Array(Kernel, length(args))
        for (i,k) in enumerate(args)
            isa(k, Kernel) || throw(ArgumentError("All arguments of SumKernel must be Kernel objects"))
            kerns[i] = k
        end
        return new(kerns)
    end
end

type SumData <: KernelData
    datadict::Dict{Symbol, KernelData}
    keys::Vector{Symbol}
end

function KernelData{M<:MatF64}(sumkern::SumKernel, X::M)
    datadict = Dict{Symbol, KernelData}()
    datakeys = Symbol[]
    for k in sumkern.kerns
        data_type = kernel_data_key(k, X)
        if !haskey(datadict, data_type)
            datadict[data_type] = KernelData(k, X)
        end
        push!(datakeys, data_type)
    end
    SumData(datadict, datakeys)
end
kernel_data_key{M<:MatF64}(sumkern::SumKernel, X::M) = :SumData

function show(io::IO, sumkern::SumKernel, depth::Int = 0)
    pad = repeat(" ", 2 * depth)
    println(io, "$(pad)Type: $(typeof(sumkern))")
    for k in sumkern.kerns
        show(io, k, depth+1)
    end
end

function cov{V1<:VecF64,V2<:VecF64}(sumkern::SumKernel, x::V1, y::V2)
    s = 0.0
    for k in sumkern.kerns
        s += cov(k, x, y)
    end
    return s
end

function addcov!{M<:MatF64}(s::MatF64, sumkern::SumKernel, X::M, data::SumData)
    for (ikern,kern) in enumerate(sumkern.kerns)
        addcov!(s, kern, X, data.datadict[data.keys[ikern]])
    end
    return s
end
function cov!{M<:MatF64}(s::MatF64, sumkern::SumKernel, X::M, data::SumData)
    s[:,:] = 0.0
    addcov!(s, sumkern, X, data)
end
function cov{M<:MatF64}(sumkern::SumKernel, X::M, data::SumData)
    d, nobsv = size(X)
    s = zeros(nobsv, nobsv)
    cov!(s, sumkern, X, data)
end
    
function get_params(sumkern::SumKernel)
    p = Array(Float64, 0)
    for k in sumkern.kerns
        append!(p, get_params(k))
    end
    p
end

get_param_names(sumkern::SumKernel) = composite_param_names(sumkern.kerns, :sk)

function num_params(sumkern::SumKernel)
    n = 0
    for k in sumkern.kerns
        n += num_params(k)
    end
    n
end

function set_params!(sumkern::SumKernel, hyp::Vector{Float64})
    i, n = 1, num_params(sumkern)
    length(hyp) == num_params(sumkern) || throw(ArgumentError("SumKernel object requires $(n) hyperparameters"))
    for k in sumkern.kerns
        np = num_params(k)
        set_params!(k, hyp[i:(i+np-1)])
        i += np
    end
end

function grad_kern{V1<:VecF64,V2<:VecF64}(sumkern::SumKernel, x::V1, y::V2)
     dk = Array(Float64, 0)
      for k in sumkern.kerns
        append!(dk,grad_kern(k, x, y))
      end
    dk
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
@inline function dKij_dθp{M<:MatF64}(sumkern::SumKernel, X::M, data::SumData, i::Int, j::Int, p::Int, dim::Int)
    s=0
    for (ikern,kern) in enumerate(sumkern.kerns)
        np = num_params(kern)
        if p<=np+s
            return dKij_dθp(kern, X, data.datadict[data.keys[ikern]],i,j,p-s,dim)
        end
        s += np
    end
end
function grad_slice!{M<:MatF64}(dK::MatF64, sumkern::SumKernel, X::M, data::SumData, p::Int)
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
function +(k1::SumKernel, k2::Kernel)
    kerns = [k1.kerns; k2]
    SumKernel(kerns...)
end
function +(k1::SumKernel, k2::SumKernel)
    kerns = [k1.kerns; k2.kerns]
    SumKernel(kerns...)
end
+(k1::Kernel, k2::Kernel) = SumKernel(k1,k2)
+(k1::Kernel, k2::SumKernel) = +(k2,k1)
