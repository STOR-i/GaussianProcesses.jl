abstract type CompositeKernel <: Kernel end

components(k::CompositeKernel) = k.kernels

@deprecate subkernels components

#################
# CompositeData #
#################

struct CompositeData <: KernelData
    datadict::Dict{String, KernelData}
    keys::Vector{String}
end

function KernelData(compkern::CompositeKernel, X1::AbstractMatrix, X2::AbstractMatrix)
    datadict = Dict{String, KernelData}()
    datakeys = String[]
    for k in components(compkern)
        data_type = kernel_data_key(k, X1, X2)
        if !haskey(datadict, data_type)
            datadict[data_type] = KernelData(k, X1, X2)
        end
        push!(datakeys, data_type)
    end
    CompositeData(datadict, datakeys)
end

function kernel_data_key(compkern::CompositeKernel, X1::AbstractMatrix, X2::AbstractMatrix)
    join(["CompositeData" ; sort(unique(kernel_data_key(k, X1, X2) for k in components(compkern)))])
end
