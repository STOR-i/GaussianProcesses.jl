import Base.show
# Main GaussianProcess type

@compat abstract type GPBase end

@doc """
# Description
Calculates the posterior mean and variance of the Gaussian Process function at specified points

# Arguments:
* `gp::GP`: Gaussian Process object
* `X::Matrix{Float64}`:  matrix of points for which one would would like to predict the value of the process.
                       (each column of the matrix is a point)

# Keyword Arguments
* `full_cov::Bool`: indicates whether full covariance matrix should be returned instead of only variances (default is false)

# Returns:
* `(μ, σ²)::(Vector{Float64}, Vector{Float64})`: respectively the posterior mean  and variances of the posterior
                                                    process at the specified points
""" ->
function predict_f{M<:MatF64}(gp::GPBase, x::M; full_cov::Bool=false)
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
    if full_cov
        return _predict(gp, x)
    else
        ## Calculate prediction for each point independently
            μ = Array{Float64}( size(x,2))
            σ2 = similar(μ)
        for k in 1:size(x,2)
            m, sig = _predict(gp, x[:,k:k])
            μ[k] = m[1]
            σ2[k] = max(full(sig)[1,1], 0.0)
        end
        return μ, σ2
    end
end

# 1D Case for prediction
predict_f{V<:VecF64}(gp::GPBase, x::V; full_cov::Bool=false) = predict_f(gp, x'; full_cov=full_cov)

@doc """
# Description
Sometimes with Gaussian processes once gets covariance matrices that are 
mathematically positive definite, but numerically have negative eigenvalues.
To get around this, we add a little bit of weight to the diagonal (which
raises all eigenvalues by that amount) until we get a positive definite matrix.

# Arguments:
* Sigma_raw::AbstractMatrix{Float64}: a covariance matrix
"""
function tolerant_PDMat{M<:MatF64}(Sigma_raw::M)
    n = size(Sigma_raw, 1)
    size(Sigma_raw, 2) == n || throw(ArgumentError("Covariance matrix must be square"))
    for _ in 1:10 # 10 chances
        try
            Sigma = PDMat(Sigma_raw)
            return Sigma
        catch
            # that wasn't (numerically) positive definite,
            # so let's add some weight to the diagonal
            ϵ = 1e-6*trace(Sigma_raw)/n
            @inbounds for i in 1:n
                Sigma_raw[i, i] += ϵ
            end
        end
    end
    # last chance
    Sigma = PDMat(Sigma_raw)
    return Sigma
end
