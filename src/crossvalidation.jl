using Distributions: Normal, logpdf
using LinearAlgebra: ldiv!, diag, inv, tr

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
as a vector of means and variances.
Using the notation from Rasmussen & Williams, see e.g. equation 5.12:
    σᵢ = 𝕍 (yᵢ | y₋ᵢ)^(1/2)
    μᵢ = 𝔼 (yᵢ | y₋ᵢ)

The output is the same as fitting the GP on x₋ᵢ,y₋ᵢ,  and calling `predict_f`
on xᵢ to obtain the LOO predictive mean and variance, repeated for each observation i.
With GPs, this can thankfully be done analytically with a bit of linear algebra,
which is what this function implements.

See also: [`predict_CVfold`](@ref), [`logp_LOO`](@ref)
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
This is implemented by  summing the
normal log-pdf of each observation 
with predictive LOO mean and variance parameters
obtained by the `predict_LOO` function.

See also: [`logp_CVfold`](@ref), [`update_mll!`](@ref)
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
    dlogpdθ_LOO_kern!(gp::GPE)

Gradient of leave-one-out CV criterion with respect to the kernel hyperparameters.
See Rasmussen & Williams equations 5.13.

See also: [`logp_LOO`](@ref), [`dlogpdσ2_LOO`](@ref), [`dlogpdθ_LOO`](@ref)
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
        grad_slice!(Zj, kernel, x, x, data, j)
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

"""
    dlogpdσ2_LOO(invΣ::PDMat, x::AbstractMatrix, y::AbstractVector, data::KernelData, alpha::AbstractVector)

Derivative of leave-one-out CV criterion with respect to the logNoise parameter.

See also: [`logp_LOO`](@ref), [`dlogpdθ_LOO_kern!`](@ref), [`dlogpdθ_LOO`](@ref)
"""
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

"""
    dlogpdθ_LOO(gp::GPE; noise::Bool, domean::Bool, kern::Bool)

Derivatives of the leave-one-out CV criterion.

See also: [`logp_LOO`](@ref), [`dlogpdθ_LOO_kern!`](@ref), [`dlogpdσ2_LOO`](@ref)
"""
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
        ∂logp∂θ[i] = dlogpdσ2_LOO(invΣ, x, y, data, alpha)*2*exp(2 * gp.logNoise.value)
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
    predict_CVfold(gp::GPE, folds::Folds)

Cross-validated predictions for arbitrary folds. 
A fold is a set of indices of the validation set.
Returns predictions of y_V (validation set) given all other observations y_T (training set),
as a vector of means and covariances.
Using the notation from Rasmussen & Williams, see e.g. equation 5.12:
    σᵢ = 𝕍 (yᵢ | y₋ᵢ)^(1/2)
    μᵢ = 𝔼 (yᵢ | y₋ᵢ)

The output is the same as fitting the GP on x_T,y_T,  and calling `predict_f`
on x_V to obtain the LOO predictive mean and covariance, repeated for each fold V.
With GPs, this can thankfully be done analytically with a bit of linear algebra,
which is what this function implements.

See also: [`predict_LOO`](@ref)
"""
function predict_CVfold(gp::GPE, folds::Folds)
    # extract relevant bits from GPE object
    Σ = gp.cK
    alpha = gp.alpha
    y = gp.y
    return predict_CVfold(Σ, alpha, y, folds)
end

"""
    logp_CVfold(gp::GPE, folds::Folds)

CV criterion for arbitrary fold.  A fold is a set of indices of the validation set.

See also [`predict_CVfold`](@ref), [`logp_LOO`](@ref)
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
    gradient_fold(invΣ, alpha, ZjΣinv, Zjα, V::AbstractVector{Int})

Gradient with respect to the kernel hyperparameters of the CV criterion
component for one validation fold:
```math
    \\nabla_{\\theta}\\left(\\log p (Y_V \\mid Y_T, \\theta) \\right)
```
where `Y_V` is the validation set and `Y_T` is the training set (all other observations)
for this fold.
"""
@inline function gradient_fold(invΣ, alpha, ZjΣinv, Zjα, V::AbstractVector{Int})
    ∂logp∂θj = 0.0
    ΣVTinv = PDMats.PDMat(mat(invΣ)[V,V])
    ΣVTα = ΣVTinv\alpha[V]
    # exponentiated quadratic component:
    ZjΣinvVV = ZjΣinv[V,V]
    ∂logp∂θj -= 2*dot(ΣVTα,Zjα[V])
    ∂logp∂θj += dot(ΣVTα, ZjΣinvVV*ΣVTα)
    # log determinant component:
    ∂logp∂θj += tr(ΣVTinv\ZjΣinvVV)
    return ∂logp∂θj
end

function dlogpdθ_CVfold_kern!(∂logp∂θ::AbstractVector{<:Real}, invΣ::PDMat, kernel::Kernel, x::AbstractMatrix, y::AbstractVector, data::KernelData, alpha::AbstractVector, folds::Folds)
    nobs = length(y)
    dim = num_params(kernel)

    @assert length(∂logp∂θ) == dim
    buffer1 = Matrix{Float64}(undef, nobs, nobs)
    buffer2 = Matrix{Float64}(undef, nobs, nobs)
    for j in 1:dim
        grad_slice!(buffer2, kernel, x, x, data, j)
        mul!(buffer1, mat(invΣ), buffer2)
        Zj = buffer1
        Zjα = Zj*alpha

        mul!(buffer2, Zj, mat(invΣ))
        ZjΣinv = buffer2

        ∂logp∂θj = 0.0
        for V in folds
            ∂logp∂θj += gradient_fold(invΣ, alpha, ZjΣinv, Zjα, V)
        end
        ∂logp∂θ[j] = ∂logp∂θj
    end
    ∂logp∂θ .*= -1/2
    return ∂logp∂θ
end


function dlogpdσ2_CVfold(invΣ::PDMat, x::AbstractMatrix, y::AbstractVector, data::KernelData, alpha::AbstractVector, folds::Folds)
    nobs = length(y)

    Zj = mat(invΣ)
    Zjα = Zj*alpha
    ZjΣinv = Zj^2

    ∂logp∂σ2 = 0.0
    for V in folds
        ∂logp∂σ2 += gradient_fold(invΣ, alpha, ZjΣinv, Zjα, V)
    end
    return -∂logp∂σ2 / 2
end

"""
    dlogpdθ_CVfold_kern!(∂logp∂θ::AbstractVector{<:Real}, gp::GPE, folds::Folds; 
                         noise::Bool, domean::Bool, kern::Bool)

Derivative of leave-one-out CV criterion with respect to the noise, mean and kernel hyperparameters.
See Rasmussen & Williams equations 5.13.
"""
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
        ∂logp∂θ[i] = dlogpdσ2_CVfold(invΣ, x, y, data, alpha, folds)*2*exp(2 * gp.logNoise.value)
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
