abstract type CompositeKernel <: Kernel end

components(k::CompositeKernel) = k.kernels

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
    for k in components(compkern)
        data_type = kernel_data_key(k, X)
        if !haskey(datadict, data_type)
            datadict[data_type] = KernelData(k, X)
        end
        push!(datakeys, data_type)
    end
    CompositeData(datadict, datakeys)
end

function kernel_data_key(compkern::CompositeKernel, X::MatF64)
    join(["CompositeData" ; sort(unique(kernel_data_key(k, X) for k in components(compkern)))])
end

include("sum_kernel.jl")        # Sum of kernels
include("prod_kernel.jl")       # Product of kernels
