# Main GaussianProcess type

abstract type GPBase end

"""
    The abstract CovarianceStrategy type is for types that control how
    the covariance matrices and their positive definite representation
    are obtained or approximated. See SparseStrategy for examples.
"""
abstract type CovarianceStrategy end
struct FullCovariance <: CovarianceStrategy end
KernelData(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, covstrat::CovarianceStrategy) = EmptyData()
KernelData(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, covstrat::FullCovariance) = KernelData(k, X1, X2)
function alloc_cK(covstrat::CovarianceStrategy, nobs)
    # create placeholder PDMat
    m = Matrix{Float64}(undef, nobs, nobs)
    chol = Matrix{Float64}(undef, nobs, nobs)
    cK = PDMats.PDMat(m, Cholesky(chol, 'U', 0))
    return cK
end

#===============================
  Predictions
================================#
function predictMVN!(Kxx, Kff, Kfx, mx, αf)
    mu = mx + Kfx' * αf
    Lck = whiten!(Kff, Kfx)
    subtract_Lck!(Kxx, Lck)
    return mu, Kxx
end

"""
        predictMVN(xpred::AbstractMatrix, xtrain::AbstractMatrix, ytrain::AbstractVector,
                   kernel::Kernel, meanf::Mean, alpha::AbstractVector,
                   covstrat::CovarianceStrategy, Ktrain::AbstractPDMat)
        
Compute predictions using the standard multivariate normal conditional distribution formulae.
"""
function predictMVN(xpred::AbstractMatrix, xtrain::AbstractMatrix, ytrain::AbstractVector,
                   kernel::Kernel, meanf::Mean, alpha::AbstractVector,
                   covstrat::CovarianceStrategy, Ktrain::AbstractPDMat)
    crossdata = KernelData(kernel, xtrain, xpred)
    priordata = KernelData(kernel, xpred, xpred)
    Kcross = cov(kernel, xtrain, xpred, crossdata)
    Kpred = cov(kernel, xpred, xpred, priordata)
    mx = mean(meanf, xpred)
    mu, Sigma_raw = predictMVN!(Kpred, Ktrain, Kcross, mx, alpha)
    return mu, Sigma_raw
end

@inline function subtract_Lck!(Sigma_raw::AbstractArray{<:AbstractFloat}, Lck::AbstractArray{<:AbstractFloat})
    LinearAlgebra.BLAS.syrk!('U', 'T', -1.0, Lck, 1.0, Sigma_raw)
    LinearAlgebra.copytri!(Sigma_raw, 'U')
end
@inline subtract_Lck!(Sigma_raw, Lck) = Sigma_raw .-= Lck'Lck

"""
    predict_f(gp::GPBase, X::Matrix{Float64}[]; full_cov::Bool = false)

Return posterior mean and variance of the Gaussian Process `gp` at specfic points which are
given as columns of matrix `X`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""
function predict_f(gp::GPBase, x::AbstractMatrix; full_cov::Bool=false)
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
    if full_cov
        return predict_full(gp, x)
    else
        ## Calculate prediction for each point independently
        μ = Array{eltype(x)}(undef, size(x,2))
        σ2 = similar(μ)
        for k in 1:size(x,2)
            m, sig = predict_full(gp, x[:,k:k])
            μ[k] = m[1]
            σ2[k] = max(diag(sig)[1], 0.0)
        end
        return μ, σ2
    end
end

# 1D Case for prediction of process
predict_f(gp::GPBase, x::AbstractVector, args...; kwargs...) = predict_f(gp, x', args...; kwargs...)
# 1D Case for prediction of observations
predict_y(gp::GPBase, x::AbstractVector, args...; kwargs...) = predict_y(gp, x', args...; kwargs...)
@deprecate predict predict_y

wrap_cK(cK::PDMat, Σbuffer, chol::Cholesky) = PDMat(Σbuffer, chol)
mat(cK::PDMat) = cK.mat
cholfactors(cK::PDMat) = cK.chol.factors
"""
    make_posdef!(m::Matrix{Float64}, chol_factors::Matrix{Float64})

Try to encode covariance matrix `m` as a positive definite matrix.
The `chol_factors` matrix is recycled to store the cholesky decomposition,
so as to reduce the number of memory allocations.

Sometimes covariance matrices of Gaussian processes are positive definite mathematically
but have negative eigenvalues numerically. To resolve this issue, small weights are added
to the diagonal (and hereby all eigenvalues are raised by that amount mathematically)
until all eigenvalues are positive numerically.
"""
function make_posdef!(m::AbstractMatrix, chol_factors::AbstractMatrix)
    n = size(m, 1)
    size(m, 2) == n || throw(ArgumentError("Covariance matrix must be square"))
    for _ in 1:10 # 10 chances
        try
            # return m, cholesky(m)
            copyto!(chol_factors, m)
            chol = cholesky!(Symmetric(chol_factors, :U))
            return m, chol
        catch err
            if typeof(err)!=LinearAlgebra.PosDefException
                throw(err)
            end
            # that wasn't (numerically) positive definite,
            # so let's add some weight to the diagonal
            ϵ = 1e-6 * tr(m) / n
            @inbounds for i in 1:n
                m[i, i] += ϵ
            end
        end
    end
    copyto!(chol_factors, m)
    chol = cholesky!(Symmetric(chol_factors, :U))
    return m, chol
end
function make_posdef!(m::AbstractMatrix)
    chol_buffer = similar(m)
    return make_posdef!(m, chol_buffer)
end

#———————————————————————————————————————————————————————————
# Sample random draws from the GP
function Random.rand!(gp::GPBase, x::AbstractMatrix, A::DenseMatrix)
    nobs = size(x,2)
    n_sample = size(A,2)

    if gp.nobs == 0
        # Prior mean and covariance
        μ = mean(gp.mean, x);
        Σraw = cov(gp.kernel, x, x);
        Σraw, chol = make_posdef!(Σraw)
        Σ = PDMat(Σraw, chol)
    else
        # Posterior mean and covariance
        μ, Σraw = predict_f(gp, x; full_cov=true)
        Σraw, chol = make_posdef!(Σraw)
        Σ = PDMat(Σraw, chol)
    end
    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobs, n_sample)))
end

Random.rand(gp::GPBase, x::AbstractMatrix, n::Int) = rand!(gp, x, Array{Float64}(undef, size(x, 2), n))

# Sample from 1D GPBase
Random.rand(gp::GPBase, x::AbstractVector, n::Int) = rand(gp, x', n)

# Generate only one sample from the GPBase and returns a vector
Random.rand(gp::GPBase, x::AbstractMatrix) = vec(rand(gp,x,1))
Random.rand(gp::GPBase, x::AbstractVector) = vec(rand(gp,x',1))
