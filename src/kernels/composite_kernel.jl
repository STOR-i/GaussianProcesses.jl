abstract type CompositeKernel <: Kernel end

subkernels(k::CompositeKernel) = throw(MethodError(subkernels, (k,)))

function Base.show(io::IO, compkern::CompositeKernel, depth::Int = 0)
    pad = repeat(" ", 2 * depth)
    println(io, "$(pad)Type: $(typeof(compkern))")
    for k in subkernels(compkern)
        show(io, k, depth+1)
    end
end

function get_params(compkern::CompositeKernel)
    p = Array{Float64}(undef, 0)
    for k in subkernels(compkern)
        append!(p, get_params(k))
    end
    p
end

get_param_names(compkern::CompositeKernel) = composite_param_names(subkernels(compkern), :sk)

num_params(ck::CompositeKernel) = sum(num_params(k) for k in subkernels(ck))


function set_params!(compkern::CompositeKernel, hyp::VecF64)
    i, n = 1, num_params(compkern)
    length(hyp) == num_params(compkern) || throw(ArgumentError("CompositeKernel object requires $(n) hyperparameters"))
    for k in subkernels(compkern)
        np = num_params(k)
        set_params!(k, hyp[i:(i+np-1)])
        i += np
    end
end

##########
# Priors #
##########

function set_priors!(compkern::CompositeKernel, priors::Array)
    i, n = 1, num_params(compkern)
    length(priors) == num_params(compkern) || throw(ArgumentError("CompositeKernel object requires $(n) priors"))
    for k in subkernels(compkern)
        np = num_params(k)
        set_priors!(k, priors[i:(i+np-1)])
        i += np
    end
end

function get_priors(compkern::CompositeKernel)
    p = []
    for k in subkernels(compkern)
        append!(p, get_priors(k))
    end
    p
end

#################
# CompositeData #
#################

struct CompositeData <: KernelData
    datadict::Dict{String, KernelData}
    keys::Vector{String}
end

function KernelData(compkern::CompositeKernel, X::MatF64)
    datadict = Dict{String, KernelData}()
    datakeys = String[]
    for k in subkernels(compkern)
        data_type = kernel_data_key(k, X)
        if !haskey(datadict, data_type)
            datadict[data_type] = KernelData(k, X)
        end
        push!(datakeys, data_type)
    end
    CompositeData(datadict, datakeys)
end

function kernel_data_key(compkern::CompositeKernel, X::MatF64)
    join(["CompositeData" ; sort(unique(kernel_data_key(k, X) for k in subkernels(compkern)))])
end

include("sum_kernel.jl")        # Sum of kernels
include("prod_kernel.jl")       # Product of kernels
