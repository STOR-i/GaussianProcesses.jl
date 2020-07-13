struct SparseKernelData{K<:KernelData} <: KernelData
    Kuu::K
    Kux1::K
    Kux2::K
end

function SparseKernelData(k::Kernel, Xu::AbstractMatrix, X1::AbstractMatrix, X2::AbstractMatrix)
    Kuu = KernelData(k, Xu, Xu)
    Kux1 = KernelData(k, Xu, X1)
    if X1 == X2
        return SparseKernelData(Kuu, Kux1, Kux1)
    else
        Kux2 = KernelData(k, Xu, X2)
        return SparseKernelData(Kuu, Kux1, Kux2)
    end
end

function KernelData(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, covstrat::SparseStrategy)
    Xu = covstrat.inducing
    return SparseKernelData(k, Xu, X1, X2)
end
