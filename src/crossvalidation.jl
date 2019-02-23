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
    dlogpdθ_LOO_kern(gp::GPE)

Derivative of leave-one-out CV criterion with respect to the kernel hyperparameters.
See Rasmussen & Williams equations 5.13.

TODO: mean and noise parameters also.
"""
function dlogpdθ_LOO_kern!(∂logp∂θ::AbstractVector{<:Real}, invΣ::PDMat, kernel::Kernel, x::AbstractMatrix, y::AbstractVector, data::KernelData, alpha::AbstractVector)
    dim = num_params(kernel)
    nobs = length(y)
    @assert length(∂logp∂θ) == dim

    σi2 = 1 ./ diag(invΣ)
    μi = -alpha .* σi2 .+ y

    # Note: if useful, the derivatives of μ and σ could be moved to a separate function.
    # ∂μ∂θ = Matrix{Float64}(undef, nobs, dim)
    # ∂σ∂θ = Matrix{Float64}(undef, nobs, dim)
    Zj = Matrix{Float64}(undef, nobs, nobs)
    for j in 1:dim
        grad_slice!(Zj, kernel, x, data, j)
        Zj = invΣ.mat * Zj
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
    ∂logp∂θ .*= -1/2
    return ∂logp∂θ
end

function dlogpdσ2_LOO(invΣ::PDMat, x::AbstractMatrix, y::AbstractVector, data::KernelData, alpha::AbstractVector)
    nobs = length(y)

    σi2 = 1 ./ diag(invΣ)
    μi = -alpha .* σi2 .+ y

    Zj = invΣ.mat
    ZjΣinv = diag(Zj^2)
    ∂σ2∂σ2 = ZjΣinv.*(σi2.^2)
    ∂μ∂σ2 = (Zj*alpha).*σi2 .- alpha .* ∂σ2∂σ2

    ∂logp∂σ2 = 0.0
    for i in 1:nobs
        # exponentiated quadratic component:
        ∂logp∂σ2 -= 2*(y[i]-μi[i]) / σi2[i] * ∂μ∂σ2[i]
        ∂logp∂σ2 -= (y[i]-μi[i])^2 * ZjΣinv[i]
        # log determinant component:
        @assert ZjΣinv[i] * σi2[i] ≈ ∂σ2∂σ2[i] / σi2[i]
        ∂logp∂σ2 += ZjΣinv[i] * σi2[i]
    end
    return -∂logp∂σ2 ./ 2
end

function dlogpdθ_LOO(gp::GPE; noise::Bool, domean::Bool, kern::Bool)
    Σ = gp.cK
    x, y = gp.x, gp.y
    data = gp.data
    kernel = gp.kernel
    alpha = gp.alpha

    invΣ = inv(Σ)

    n_mean_params = num_params(gp.mean)
    n_kern_params = num_params(gp.kernel)
    ∂logp∂θ = Vector{Float64}(undef, noise + domean*n_mean_params + kern*n_kern_params)
    i = 1
    if noise
        ∂logp∂θ[i] = dlogpdσ2_LOO(invΣ, x, y, data, alpha)*2*exp(2 * gp.logNoise)
        i += 1
    end
    if domean && n_mean_params>0
        throw("I don't know how to do means yet")
        Mgrads = grad_stack(gp.mean, gp.x)
        for j in 1:n_mean_params
            gp.dmll[i] = dot(Mgrads[:,j], gp.alpha)
            i += 1
        end
    end
    if kern
        ∂logp∂θ_k = @view(∂logp∂θ[i:end])
        dlogpdθ_LOO_kern!(∂logp∂θ_k, invΣ, kernel, x, y, data, alpha)
    end
    return ∂logp∂θ
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
    dlogpdθ_CVfold_kern!(∂logp∂θ::AbstractVector{<:Real}, gp::GPE, folds::Folds)

Derivative of leave-one-out CV criterion with respect to the kernel hyperparameters.
See Rasmussen & Williams equations 5.13.

TODO: mean and noise parameters also.
"""
function dlogpdθ_CVfold_kern!(∂logp∂θ::AbstractVector{<:Real}, invΣ::PDMat, kernel::Kernel, x::AbstractMatrix, y::AbstractVector, data::KernelData, alpha::AbstractVector, folds::Folds)
    nobs = length(y)
    dim = num_params(kernel)

    @assert length(∂logp∂θ) == dim
    buffer1 = Matrix{Float64}(undef, nobs, nobs)
    buffer2 = Matrix{Float64}(undef, nobs, nobs)
    for j in 1:dim
        grad_slice!(buffer2, kernel, x, data, j)
        mul!(buffer1, invΣ.mat, buffer2)
        Zj = buffer1
        # ldiv!(Σ, Zj)
        Zjα = Zj*alpha

        mul!(buffer2, Zj, invΣ.mat)
        ZjΣinv = buffer2

        ∂logp∂θj = 0.0
        for V in folds
            ΣVT = inv(invΣ.mat[V,V])
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
    ∂logp∂θ .*= -1/2
    return ∂logp∂θ
end

function dlogpdσ2_CVfold(invΣ::PDMat, x::AbstractMatrix, y::AbstractVector, data::KernelData, alpha::AbstractVector, folds::Folds)
    nobs = length(y)

    Zj = invΣ.mat
    Zjα = Zj*alpha
    ZjΣinv = invΣ.mat^2

    ∂logp∂σ2 = 0.0
    for V in folds
        ΣVT = inv(invΣ.mat[V,V])
        μVT = y[V]-ΣVT*alpha[V]
        # exponentiated quadratic component:
        resid = y[V]-μVT
        ZjΣinvVV = ZjΣinv[V,V]
        ∂logp∂σ2 -= 2*dot(resid, Zjα[V] .- ZjΣinvVV*ΣVT*alpha[V])
        ∂logp∂σ2 -= dot(resid, ZjΣinvVV*resid)
        # log determinant component:
        ∂logp∂σ2 += dot(ZjΣinvVV,ΣVT)
    end
    return -∂logp∂σ2 / 2
end

function dlogpdθ_CVfold(gp::GPE, folds::Folds; noise::Bool, domean::Bool, kern::Bool)
    Σ = gp.cK
    x, y = gp.x, gp.y
    data = gp.data
    kernel = gp.kernel
    alpha = gp.alpha

    invΣ = inv(Σ)

    n_mean_params = num_params(gp.mean)
    n_kern_params = num_params(gp.kernel)
    ∂logp∂θ = Vector{Float64}(undef, noise + domean*n_mean_params + kern*n_kern_params)
    i = 1
    if noise
        ∂logp∂θ[i] = dlogpdσ2_CVfold(invΣ, x, y, data, alpha, folds)*2*exp(2 * gp.logNoise)
        i += 1
    end
    if domean && n_mean_params>0
        throw("I don't know how to do means yet")
        Mgrads = grad_stack(gp.mean, gp.x)
        for j in 1:n_mean_params
            gp.dmll[i] = dot(Mgrads[:,j], gp.alpha)
            i += 1
        end
    end
    if kern
        ∂logp∂θ_k = @view(∂logp∂θ[i:end])
        dlogpdθ_CVfold_kern!(∂logp∂θ_k, invΣ, kernel, x, y, data, alpha, folds)
    end
    return ∂logp∂θ
end
