abstract type SparsePDMat{T} <: AbstractPDMat{T} end
abstract type SparseStrategy <: CovarianceStrategy end

function get_ααinvcKI!(ααinvcKI::AbstractMatrix, cK::SparsePDMat, α::Vector)
    nobs = length(α)
    size(ααinvcKI) == (nobs, nobs) || throw(ArgumentError(
                @sprintf("Buffer for ααinvcKI should be a %dx%d matrix, not %dx%d",
                         nobs, nobs,
                         size(ααinvcKI,1), size(ααinvcKI,2))))
    # fill!(ααinvcKI, 0)
    # @inbounds for i in 1:nobs
        # ααinvcKI[i,i] = -1.0
    # end
    ααinvcKI = cK \ (-I)
    BLAS.ger!(1.0, α, α, ααinvcKI)
end

include("subsetofregressors.jl")

