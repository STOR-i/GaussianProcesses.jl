import Base.show
# Main GaussianProcess type

abstract GPBase

@doc """
# Description
Calculates the posterior mean and variance of the Gaussian Process function at specified points

# Arguments:
* `gp::GPE`: Gaussian Process object
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
            μ = Array(Float64, size(x,2))
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
