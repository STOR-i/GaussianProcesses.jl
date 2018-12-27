abstract type SparsePDMat{T} <: AbstractPDMat{T} end
abstract type SparseStrategy <: CovarianceStrategy end

include("subsetofregressors.jl")

