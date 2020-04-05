#==========================================================
 Sparse Positive Definite Matrix for Subset of Regressors
===========================================================#
"""
    Subset of Regressors sparse positive definite matrix.
"""
mutable struct SubsetOfRegsPDMat{T,M<:AbstractMatrix,PD<:AbstractPDMat{T},M2<:AbstractMatrix{T}} <: SparsePDMat{T}
    inducing::M
    ΣQR_PD::PD
    Kuu::PD
    Kuf::M2
    logNoise::Float64
end
function getQab(cK::SparsePDMat, kernel::Kernel, Xa::AbstractMatrix, Xb::AbstractMatrix, kerneldata::SparseKernelData)
    Kuu = cK.Kuu
    inducing = cK.inducing
    Kua = cov(kernel, inducing, Xa, kerneldata.Kux1)
    Kub = cov(kernel, inducing, Xb, kerneldata.Kux2)
    Lua = whiten!(Kuu, Kua)
    Lub = whiten!(Kuu, Kub)
    return Lua'Lub
end
function getQab(cK::SparsePDMat, kernel::Kernel, Xa::AbstractMatrix, Xb::AbstractMatrix)
    inducing = cK.inducing
    kdata = SparseKernelData(kernel, inducing, Xa, Xb)
    return getQab(cK, kernel, Xa, Xb, kdata)
end
function getQaa(cK::SparsePDMat, kernel::Kernel, Xa::AbstractMatrix, kerneldata::SparseKernelData)
    Kuu = cK.Kuu
    inducing = cK.inducing
    Kua = cov(kernel, inducing, Xa, kerneldata.Kux1)
    Qaa = PDMats.Xt_invA_X(Kuu, Kua) # Kau Kuu⁻¹ Kua
    return Qaa
end
function getQaa(cK::SparsePDMat, kernel::Kernel, Xa::AbstractMatrix)
    inducing = cK.inducing
    kdata = SparseKernelData(kernel, inducing, Xa, Xa)
    return getQaa(cK, kernel, Xa, kdata)
end
size(a::SubsetOfRegsPDMat) = (size(a.Kuf,2), size(a.Kuf,2))
size(a::SubsetOfRegsPDMat, d::Int) = size(a.Kuf,2)
"""
    We have
        Σ ≈ Kuf' Kuu⁻¹ Kuf + σ²I
    By Woodbury
        Σ⁻¹ = σ⁻²I - σ⁻⁴ Kuf'(Kuu + σ⁻² Kuf Kuf')⁻¹ Kuf
            = σ⁻²I - σ⁻⁴ Kuf'(       ΣQR        )⁻¹ Kuf
"""
function \(a::SubsetOfRegsPDMat, x)
    return exp(-2*a.logNoise)*x - exp(-4*a.logNoise)*a.Kuf'*(a.ΣQR_PD \ (a.Kuf * x))
end
logdet(a::SubsetOfRegsPDMat) = logdet(a.ΣQR_PD) - logdet(a.Kuu) + 2*a.logNoise*size(a,1)

function wrap_cK(cK::SubsetOfRegsPDMat, inducing, ΣQR_PD, Kuu, Kuf, logNoise::Scalar)
    wrap_cK(cK, inducing, ΣQR_PD, Kuu, Kuf, logNoise.value)
end
function wrap_cK(cK::SubsetOfRegsPDMat, inducing, ΣQR_PD, Kuu, Kuf, logNoise)
    SubsetOfRegsPDMat(inducing, ΣQR_PD, Kuu, Kuf, logNoise)
end
function LinearAlgebra.tr(a::SubsetOfRegsPDMat)
    exp(2*a.logNoise)*size(a.Kuf,2) + dot(a.Kuf, a.Kuu \ a.Kuf) # TODO: there may be a shortcut here
end
function Base.Matrix(a::SubsetOfRegsPDMat)
    Lk = whiten(a.Kuu, a.Kuf)
    Σ = Lk'Lk
    nobs = size(Σ,1)
    for i in 1:nobs
        Σ[i,i] += exp(2*a.logNoise)
    end
    return Σ
end

#========================================
 Subset of Regressors strategy
=========================================#

struct SubsetOfRegsStrategy{M<:AbstractMatrix} <: SparseStrategy
    inducing::M
end

function alloc_cK(covstrat::SubsetOfRegsStrategy, nobs)
    inducing = covstrat.inducing
    ninducing = size(inducing, 2)
    Kuu  = Matrix{Float64}(undef, ninducing, ninducing)
    chol_uu = Matrix{Float64}(undef, ninducing, ninducing)
    Kuu_PD = PDMats.PDMat(Kuu, Cholesky(chol_uu, 'U', 0))
    Kuf  = Matrix{Float64}(undef, ninducing, nobs)
    ΣQR  = Matrix{Float64}(undef, ninducing, ninducing)
    chol = Matrix{Float64}(undef, ninducing, ninducing)
    cK = SubsetOfRegsPDMat(inducing,
                            PDMats.PDMat(ΣQR, Cholesky(chol, 'U', 0)), # ΣQR_PD
                            Kuu_PD, Kuf, 42.0)
    return cK
end
function update_cK!(cK::SubsetOfRegsPDMat, x::AbstractMatrix, kernel::Kernel,
                    logNoise::Real, kerneldata::SparseKernelData, covstrat::SubsetOfRegsStrategy)
    inducing = covstrat.inducing
    Kuu = cK.Kuu
    Kuubuffer = mat(Kuu)
    cov!(Kuubuffer, kernel, inducing, kerneldata.Kuu)
    Kuubuffer, chol = make_posdef!(Kuubuffer, cholfactors(cK.Kuu))
    Kuu_PD = wrap_cK(cK.Kuu, Kuubuffer, chol)
    Kuf = cov!(cK.Kuf, kernel, inducing, x, kerneldata.Kux1)
    Kfu = Kuf'

    ΣQR = exp(-2*logNoise) * Kuf * Kfu + Kuu
    LinearAlgebra.copytri!(ΣQR, 'U')

    ΣQR, chol = make_posdef!(ΣQR, cholfactors(cK.ΣQR_PD))
    ΣQR_PD = wrap_cK(cK.ΣQR_PD, ΣQR, chol)
    return wrap_cK(cK, inducing, ΣQR_PD, Kuu_PD, Kuf, logNoise)
end

#==========================================
  Log-likelihood gradients
===========================================#
struct SoRPrecompute <: AbstractGradientPrecompute
    Kuu⁻¹Kuf::Matrix{Float64}
    Kuu⁻¹KufΣ⁻¹y::Vector{Float64}
    Σ⁻¹Kfu::Matrix{Float64}
    ∂Kuu::Matrix{Float64} # buffer
    ∂Kuf::Matrix{Float64} # buffer
end
function SoRPrecompute(nobs::Int, ninducing::Int)
    Kuu⁻¹Kuf = Matrix{Float64}(undef, ninducing, nobs)
    Kuu⁻¹KufΣ⁻¹y =  Vector{Float64}(undef, ninducing)
    Σ⁻¹Kfu = Matrix{Float64}(undef, nobs, ninducing)
    ∂Kuu = Matrix{Float64}(undef, ninducing, ninducing)
    ∂Kuf = Matrix{Float64}(undef, ninducing, nobs)
    return SoRPrecompute(Kuu⁻¹Kuf, Kuu⁻¹KufΣ⁻¹y, Σ⁻¹Kfu, ∂Kuu, ∂Kuf)
end

function init_precompute(covstrat::SubsetOfRegsStrategy, X, y, kernel)
    nobs = size(X, 2)
    ninducing = size(covstrat.inducing, 2)
    SoRPrecompute(nobs, ninducing)
end

function precompute!(precomp::SoRPrecompute, gp::GPBase)
    cK = gp.cK
    alpha = gp.alpha
    Kuf = cK.Kuf
    Kuu = cK.Kuu

    precomp.Kuu⁻¹Kuf[:,:] = Kuu \ Kuf # Kuu⁻¹Kuf
    precomp.Kuu⁻¹KufΣ⁻¹y[:] = vec(Kuu \ (Kuf * alpha)) # Kuu⁻¹Kuf Σ⁻1 y appears repeatedly, so pre-compute
    precomp.Σ⁻¹Kfu[:,:] = cK \ (Kuf') # TODO: reduce memory allocations
    return precomp
end
function dmll_kern!(dmll::AbstractVector, gp::GPBase, precomp::SoRPrecompute, covstrat::SubsetOfRegsStrategy)
    return dmll_kern!(dmll, gp.kernel, gp.x, gp.cK, gp.data, gp.alpha,
                      gp.cK.Kuu, gp.cK.Kuf,
                      precomp.Kuu⁻¹Kuf, precomp.Kuu⁻¹KufΣ⁻¹y, precomp.Σ⁻¹Kfu,
                      precomp.∂Kuu, precomp.∂Kuf,
                      covstrat)
end
function dmll_noise(logNoise::Real, cK::SubsetOfRegsPDMat, alpha::AbstractVector)
    nobs = length(alpha)
    Lk = whiten(cK.ΣQR_PD, cK.Kuf)
    return exp(2*logNoise) * (
        dot(alpha, alpha)
        - exp(-2*logNoise) * nobs
        + exp(-4*logNoise)  * dot(Lk, Lk)
        )
end
"""
    dmll_noise(gp::GPE, precomp::SoRPrecompute)

∂logp(Y|θ) = 1/2 y' Σ⁻¹ ∂Σ Σ⁻¹ y - 1/2 tr(Σ⁻¹ ∂Σ)

∂Σ = I for derivative wrt σ², so
∂logp(Y|θ) = 1/2 y' Σ⁻¹ Σ⁻¹ y - 1/2 tr(Σ⁻¹)
            = 1/2[ dot(α,α) - tr(Σ⁻¹) ]

Σ⁻¹ = σ⁻²I - σ⁻⁴ Kuf'(Kuu + σ⁻² Kuf Kuf')⁻¹ Kuf
    = σ⁻²I - σ⁻⁴ Kuf'(       ΣQR        )⁻¹ Kuf
"""
function dmll_noise(gp::GPE, precomp::SoRPrecompute, covstrat::SubsetOfRegsStrategy)
    dmll_noise(get_value(gp.logNoise), gp.cK, gp.alpha)
end

"""
    dmll_kern!(dmll::AbstractVector, kernel::Kernel, X::AbstractMatrix, cK::SubsetOfRegsPDMat, kerneldata::KernelData, ααinvcKI::AbstractMatrix, covstrat::SubsetOfRegsStrategy)

Derivative of the log likelihood under the Subset of Regressors (SoR) approximation.

Helpful reference: Vanhatalo, Jarno, and Aki Vehtari.
                   "Sparse log Gaussian processes via MCMC for spatial epidemiology."
                   In Gaussian processes in practice, pp. 73-89. 2007.

Generally, for a multivariate normal with zero mean
    ∂logp(Y|θ) = 1/2 y' Σ⁻¹ ∂Σ Σ⁻¹ y - 1/2 tr(Σ⁻¹ ∂Σ)
                    ╰───────────────╯     ╰──────────╯
                           `V`                 `T`

where Σ = Kff + σ²I.

Notation: `f` is the observations, `u` is the inducing points.
          ∂X stands for ∂X/∂θ, where θ is the kernel hyperparameters.

In the SoR approximation, we replace Kff with Qff = Kfu Kuu⁻¹ Kuf

∂Σ = ∂(Qff) = ∂(Kfu Kuu⁻¹ Kuf)
            = ∂(Kfu) Kuu⁻¹ Kuf + Kfu ∂(Kuu⁻¹) Kuf + Kfu Kuu⁻¹ ∂(Kuf)

∂(Kuu⁻¹) = -Kuu⁻¹ ∂(Kuu) Kuu⁻¹  --------^

Also have pre-computed α = Σ⁻¹ y, so `V` can now be computed
efficiency (O(nm²) I think…) by careful ordering of the matrix multiplication steps.

For `T`, we use the identity tr(AB) = dot(A',B):
tr(Σ⁻¹ ∂Σ) = 2 tr(Σ⁻¹ ∂(Kfu) Kuu⁻¹ Kuf) + tr(Σ⁻¹ Kfu ∂(Kuu⁻¹) Kuf)
           = 2 dot((Σ⁻¹ ∂(Kfu))', Kuu⁻¹ Kuf) - tr(Σ⁻¹ Kfu Kuu⁻¹ ∂Kuu Kuu⁻¹ Kuf)
           = 2 dot((Σ⁻¹ ∂(Kfu))', Kuu⁻¹ Kuf) - dot((Σ⁻¹ Kfu)', Kuu⁻¹ ∂Kuu Kuu⁻¹ Kuf)
which again is computed in O(nm²).
"""
function dmll_kern!(dmll::AbstractVector, kernel::Kernel, X::AbstractMatrix, cK::AbstractPDMat, kerneldata::SparseKernelData,
                    alpha::AbstractVector, Kuu, Kuf, Kuu⁻¹Kuf, Kuu⁻¹KufΣ⁻¹y, Σ⁻¹Kfu, ∂Kuu, ∂Kuf,
                    covstrat::SubsetOfRegsStrategy)
    dim, nobs = size(X)
    inducing = covstrat.inducing
    ninducing = size(inducing, 2)
    nparams = num_params(kernel)
    @assert nparams == length(dmll)
    dK_buffer = Vector{Float64}(undef, nparams)
    dmll[:] .= 0.0
    for iparam in 1:nparams
        grad_slice!(∂Kuu, kernel, inducing, inducing, kerneldata.Kuu, iparam)
        grad_slice!(∂Kuf, kernel, inducing, X       , kerneldata.Kux1, iparam)
        V =  2 * dot(alpha, ∂Kuf' * (Kuu⁻¹KufΣ⁻¹y))    # = 2 y' Σ⁻¹ ∂Kuf Kuu⁻¹ Kuf Σ⁻¹y
        V -= dot(Kuu⁻¹KufΣ⁻¹y, ∂Kuu * (Kuu⁻¹KufΣ⁻¹y)) # = y' Σ⁻¹ Kfu ∂(Kuu⁻¹) Kuf Σ⁻¹ y

        T = 2 * dot(cK \ ∂Kuf', Kuu⁻¹Kuf')              # = 2 tr(Kuu⁻¹ Kuf Σ⁻¹ ∂Kuf')
        T -=    dot(Σ⁻¹Kfu',  (Kuu \ ∂Kuu) * Kuu⁻¹Kuf) # = tr(Kuu⁻¹ Kuf Σ⁻¹ Kfu Kuu⁻¹ ∂Kuu)

        # # BELOW FOR DEBUG ONLY
        # ∂Σ = ∂Kuf' * Kuu⁻¹Kuf
        # ∂Σ += ∂Σ'
        # ∂Σ -= Kuu⁻¹Kuf' * ∂Kuu * Kuu⁻¹Kuf
        # Valt = alpha'*∂Σ*alpha
        # Talt = tr(cK \ ∂Σ)
        # @show V, Valt
        # @show T, Talt
        # # dmll_alt = dot(ααinvcKI, ∂Σ)/2
        # # @show dmll_alt, dmll[iparam]
        # # ABOVE FOR DEBUG ONLY

        dmll[iparam] = (V-T)/2
    end
    return dmll
end

"""
    alpha_u(Ktrain::SubsetOfRegsPDMat, xtrain::AbstractMatrix, ytrain::AbstractVector, m::Mean)

    ΣQR⁻¹ Kuf Λ⁻¹ (y-μ)
"""
function get_alpha_u(Ktrain::SubsetOfRegsPDMat, xtrain::AbstractMatrix, ytrain::AbstractVector, meanf::Mean)
    ΣQR_PD = Ktrain.ΣQR_PD
    Kuf = Ktrain.Kuf
    meantrain = mean(meanf, xtrain)
    logNoise = Ktrain.logNoise
    Λ = exp(2*logNoise)*I
    alpha_u = ΣQR_PD \ (Kuf * (Λ \ (ytrain-meantrain)) )
    return alpha_u
end
"""
    See Quiñonero-Candela and Rasmussen 2005, equations 16b.
    Some derivations can be found below that are not spelled out in the paper.

    Notation: Qab = Kau Kuu⁻¹ Kub
              ΣQR = Kuu + σ⁻² Kuf Kuf'

              x: prediction (test) locations
              f: training (observed) locations
              u: inducing point locations

    We have
        Σ ≈ Kuf' Kuu⁻¹ Kuf + σ²I
    By Woodbury
        Σ⁻¹ = σ⁻²I - σ⁻⁴ Kuf'(Kuu + σ⁻² Kuf Kuf')⁻¹ Kuf
            = σ⁻²I - σ⁻⁴ Kuf'(       ΣQR        )⁻¹ Kuf

    The predictive mean can be derived (assuming zero mean function for simplicity)
    μ = Qxf (Qff + σ²I)⁻¹ y
      = Kxu Kuu⁻¹ Kuf [σ⁻²I - σ⁻⁴ Kuf' ΣQR⁻¹ Kuf] y   # see Woodbury formula above.
      = σ⁻² Kxu Kuu⁻¹ [ΣQR - σ⁻² Kuf Kfu] ΣQR⁻¹ Kuf y # factoring out common terms
      = σ⁻² Kxu Kuu⁻¹ [Kuu] ΣQR⁻¹ Kuf y               # using definition of ΣQR
      = σ⁻² Kxu ΣQR⁻¹ Kuf y                           # matches equation 16b

    Similarly for the posterior predictive covariance:
    Σ = Qxx - Qxf (Qff + σ²I)⁻¹ Qxf'
      = Qxx - σ⁻² Kxu ΣQR⁻¹ Kuf Qxf'                # substituting result from μ
      = Qxx - σ⁻² Kxu ΣQR⁻¹  Kuf Kfu    Kuu⁻¹ Kux   # definition of Qxf
      = Qxx -     Kxu ΣQR⁻¹ (ΣQR - Kuu) Kuu⁻¹ Kux   # using definition of ΣQR
      = Qxx - Kxu Kuu⁻¹ Kux + Kxu ΣQR⁻¹ Kux         # expanding
      = Qxx - Qxx           + Kxu ΣQR⁻¹ Kux         # definition of Qxx
      = Kxu ΣQR⁻¹ Kux                               # simplifying
"""
function predictMVN(xpred::AbstractMatrix,
                    xtrain::AbstractMatrix, ytrain::AbstractVector,
                    kernel::Kernel, meanf::Mean,
                    alpha::AbstractVector,
                    covstrat::SubsetOfRegsStrategy, Ktrain::AbstractPDMat)
    ΣQR_PD = Ktrain.ΣQR_PD
    inducing = covstrat.inducing
    Kuxdata = KernelData(kernel, inducing, xpred)

    Kux = cov(kernel, inducing, xpred, Kuxdata)

    meanx = mean(meanf, xpred)
    alpha_u = get_alpha_u(Ktrain, xtrain, ytrain, meanf)
    mupred = meanx + (Kux' * alpha_u)

    Lck = PDMats.whiten(ΣQR_PD, Kux) # ΣQR^(-1/2) Kux
    Σpred = Lck'Lck # Kux' * (ΣQR_PD \ Kux)
    LinearAlgebra.copytri!(Σpred, 'U')
    return mupred, Σpred
end


function SoR(x::AbstractMatrix, inducing::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Real)
    covstrat = SubsetOfRegsStrategy(inducing)
    GPE(x, y, mean, kernel, logNoise, covstrat)
end

