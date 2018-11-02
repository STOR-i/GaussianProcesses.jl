# Main GaussianProcess type

abstract type GPBase end

"""
    predict_f(gp::GPBase, X::Matrix{Float64}[; full_cov::Bool = false])

Return posterior mean and variance of the Gaussian Process `gp` at specfic points which are
given as columns of matrix `X`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""
function predict_f(gp::GPBase, x::AbstractMatrix; full_cov::Bool=false)
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
    if full_cov
        return _predict(gp, x)
    else
        ## Calculate prediction for each point independently
            μ = Array{eltype(x)}(undef, size(x,2))
            σ2 = similar(μ)
        for k in 1:size(x,2)
            m, sig = _predict(gp, x[:,k:k])
            μ[k] = m[1]
            σ2[k] = max(diag(sig)[1], 0.0)
        end
        return μ, σ2
    end
end

# 1D Case for prediction
predict_f(gp::GPBase, x::AbstractVector; full_cov::Bool=false) = predict_f(gp, x'; full_cov=full_cov)

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

