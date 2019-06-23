import LinearAlgebra: tr, logdet

abstract type SparsePDMat{T} <: AbstractPDMat{T} end
abstract type SparseStrategy <: CovarianceStrategy end

function get_ααinvcKI!(ααinvcKI::AbstractMatrix, cK::SparsePDMat, α::Vector)
    nobs = length(α)
    size(ααinvcKI) == (nobs, nobs) || throw(ArgumentError(
                @sprintf("Buffer for ααinvcKI should be a %dx%d matrix, not %dx%d",
                         nobs, nobs,
                         size(ααinvcKI,1), size(ααinvcKI,2))))
    ααinvcKI[:,:] = cK \ (-I)
    BLAS.ger!(1.0, α, α, ααinvcKI)
end

include("sparsekerneldata.jl")
include("subsetofregressors.jl")
include("determ_train_conditional.jl")
include("fully_indep_train_conditional.jl")
include("full_scale_approximation.jl")
