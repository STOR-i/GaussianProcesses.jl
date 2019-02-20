using Distributions: Normal, logpdf
using LinearAlgebra: ldiv!, diag, inv

########################
# Leave-one-out        #
########################

function predict_LOO(Σ::AbstractPDMat, alpha::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    invΣ = inv(Σ)
    σi2 = 1 ./ diag(invΣ)
    μi = -alpha .* σi2 .+ y
    return μi, σi2
end
"""
    predict_LOO(gp::GPE)

Leave-one-out cross-validated predictions. 
Returns predictions of yᵢ given all other observations y₋ᵢ,
as a vector of means and standard deviations.
Using the notation from Rasmussen & Williams, see e.g. equation 5.12:
    σᵢ = 𝕍 (yᵢ | y₋ᵢ)^(1/2)
    μᵢ = 𝔼 (yᵢ | y₋ᵢ)
"""
function predict_LOO(gp::GPE)
    # extract relevant bits from GPE object
    Σ = gp.cK
    alpha = gp.alpha
    y = gp.y
    return predict_LOO(Σ, alpha, y)
end

"""
    logp_LOO(gp::GPE)

Leave-one-out log probability CV criterion.
"""
function logp_LOO(gp::GPE)
    y = gp.y
    μ, σ2 = predict_LOO(gp)
    return sum(logpdf(Normal(μi,√σi2), yi) 
               for (μi,σi2,yi)
               in zip(μ,σ2,y)
               )
end

"""
    dlogpdθ_LOO(gp::GPE)

Derivative of leave-one-out CV criterion with respect to the kernel hyperparameters.
See Rasmussen & Williams equations 5.13.

TODO: mean and noise parameters also.
"""
function dlogpdθ_LOO(gp::GPE)
    Σ = gp.cK
    x, y = gp.x, gp.y
    data = gp.data
    nobs = gp.nobs
    k = gp.kernel
    dim = num_params(k)
    alpha = gp.alpha

    invΣ = inv(Σ)
    σi2 = 1 ./ diag(invΣ)
    μi = -alpha .* σi2 .+ y

    # Note: if useful, the derivatives of μ and σ could be moved to a separate function.
    # ∂μ∂θ = Matrix{Float64}(undef, nobs, dim)
    # ∂σ∂θ = Matrix{Float64}(undef, nobs, dim)
    ∂logp∂θ = Vector{Float64}(undef, dim)
    Zj = Matrix{Float64}(undef, nobs, nobs)
    for j in 1:dim
        grad_slice!(Zj, k, x, data, j)
        Zj = Σ \ Zj
        # ldiv!(Σ, Zj)

        ZjΣinv = diag(Zj*Matrix(invΣ))
        ∂σ2∂θj = ZjΣinv.*(σi2.^2)
        ∂μ∂θj = (Zj*alpha).*σi2 .- alpha .* ∂σ2∂θj
        # ∂μ∂θ[:,j] = ∂μ∂θj
        # ∂σ∂θ[:,j] = ∂σ∂θj

        ∂logp∂θj = 0.0
        for i in 1:nobs
            # exponentiated quadratic component:
            ∂logp∂θj -= 2*(y[i]-μi[i]) / σi2[i] * ∂μ∂θj[i]
            ∂logp∂θj -= (y[i]-μi[i])^2 * ZjΣinv[i]
            # log determinant component:
            @assert ZjΣinv[i] * σi2[i] ≈ ∂σ2∂θj[i] / σi2[i]
            ∂logp∂θj += ZjΣinv[i] * σi2[i]
        end
        ∂logp∂θ[j] = ∂logp∂θj
    end
    return -∂logp∂θ ./ 2
end

########################
# Arbitrary fold       #
########################

const Folds = AbstractVector{<:AbstractVector{Int}}

function predict_CVfold(Σ::AbstractPDMat, alpha::AbstractVector{<:Real}, y::AbstractVector{<:Real}, folds::Folds)
    invΣ = inv(Σ)
    μ = Vector{Float64}[]
    Σ = Matrix{Float64}[]
    for V in folds
        ΣVT = inv(Matrix(invΣ)[V,V])
        μVT = y[V]-ΣVT*alpha[V]
        push!(μ, μVT)
        push!(Σ, ΣVT)
    end
    return μ, Σ
end
"""
    predict_CVfold(gp::GPE)

Leave-one-out cross-validated predictions. 
Returns predictions of yᵢ given all other observations y₋ᵢ,
as a vector of means and standard deviations.
Using the notation from Rasmussen & Williams, see e.g. equation 5.12:
    σᵢ = 𝕍 (yᵢ | y₋ᵢ)^(1/2)
    μᵢ = 𝔼 (yᵢ | y₋ᵢ)
"""
function predict_CVfold(gp::GPE, folds::Folds)
    # extract relevant bits from GPE object
    Σ = gp.cK
    alpha = gp.alpha
    y = gp.y
    return predict_CVfold(Σ, alpha, y, folds)
end

"""
    logp_CVfold(gp::GPE)

Leave-one-out log probability CV criterion.
"""
function logp_CVfold(gp::GPE, folds::Folds)
    y = gp.y
    μ, Σ = predict_CVfold(gp, folds)
    CV = 0.0
    for (μVT,ΣVT,V) in zip(μ,Σ,folds)
        chol = similar(ΣVT)
        ΣVT, chol = make_posdef!(ΣVT, chol)
        ΣPD = PDMat(ΣVT, chol)
        CV += logpdf(MvNormal(μVT, ΣPD), y[V])
    end
    return CV
end

"""
    dlogpdθ_CVfold(gp::GPE)

Derivative of leave-one-out CV criterion with respect to the kernel hyperparameters.
See Rasmussen & Williams equations 5.13.

TODO: mean and noise parameters also.
"""
function dlogpdθ_CVfold(gp::GPE, folds::Folds)
    Σ = gp.cK
    x, y = gp.x, gp.y
    data = gp.data
    nobs = gp.nobs
    k = gp.kernel
    dim = num_params(k)
    alpha = gp.alpha

    invΣ = inv(Σ)

    ∂logp∂θ = Vector{Float64}(undef, dim)
    buffer1 = Matrix{Float64}(undef, nobs, nobs)
    buffer2 = Matrix{Float64}(undef, nobs, nobs)
    for j in 1:dim
        grad_slice!(buffer2, k, x, data, j)
        mul!(buffer1, invΣ.mat, buffer2)
        Zj = buffer1
        # ldiv!(Σ, Zj)
        Zjα = Zj*alpha

        mul!(buffer2, Zj, invΣ.mat)
        ZjΣinv = buffer2

        ∂logp∂θj = 0.0
        for V in folds
            ΣVT = inv(@view(invΣ.mat[V,V]))
            μVT = y[V]-ΣVT*alpha[V]
            # exponentiated quadratic component:
            resid = y[V]-μVT
            ZjΣinvVV = ZjΣinv[V,V]
            ∂logp∂θj -= 2*dot(resid, Zjα[V] .- ZjΣinvVV*ΣVT*alpha[V])
            ∂logp∂θj -= dot(resid, ZjΣinvVV*resid)
            # log determinant component:
            ∂logp∂θj += dot(ZjΣinvVV,ΣVT)
        end
        ∂logp∂θ[j] = ∂logp∂θj
    end
    return -∂logp∂θ ./ 2
end
