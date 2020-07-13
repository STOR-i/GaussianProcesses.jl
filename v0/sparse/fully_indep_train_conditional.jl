#=======================================,2)
 Fully Independent Training Conditional
=========================================#

"""
    Positive Definite Matrix for Fully Independent Training Conditional approximation.
"""
mutable struct FullyIndepPDMat{T,M<:AbstractMatrix,PD<:AbstractPDMat{T},M2<:AbstractMatrix{T},DIAG<:Diagonal} <: SparsePDMat{T}
    inducing::M
    ΣQR_PD::PD
    Kuu::PD
    Kuf::M2
    Λ::DIAG
end
size(a::FullyIndepPDMat) = (size(a.Kuf,2), size(a.Kuf,2))
size(a::FullyIndepPDMat, d::Int) = size(a.Kuf,2)
"""
    We have
        Σ ≈ Kuf' Kuu⁻¹ Kuf + Λ
    where Λ is a diagonal matrix (here given by σ²I + diag(Kff - Qff))
    The Woodbury matrix identity gives
        (A+UCV)⁻¹ = A⁻¹ - A⁻¹ U(C⁻¹ + V A⁻¹ U)⁻¹ V A⁻¹
    which we use here with
        A ← Λ
        U = Kuf'
        V = Kuf
        C = Kuu⁻¹
    which gives
        Σ⁻¹ = Λ⁻¹ - Λ⁻² Kuf'(Kuu + Kuf Λ⁻¹ Kuf')⁻¹ Kuf
            = Λ⁻¹ - Λ⁻² Kuf'(        ΣQR       )⁻¹ Kuf
"""
function \(a::FullyIndepPDMat, x)
    Lk = whiten(a.ΣQR_PD, a.Kuf)
    return a.Λ \ (x .- Lk' * (Lk * (a.Λ \ x)))
end

"""
    trinvAB(A::AbstractPDMat, B)

    Computes tr(A⁻¹ B).
"""
function trinvAB(A::AbstractPDMat, B)
    tr(A \ B)
end
"""
    trinvAB(A::FullyIndepPDMat, B::Diagonal)

    Computes tr(A⁻¹ B) efficiently under the FITC approximation:
        Σ ≈ Kuf' Kuu⁻¹ Kuf + Λ

    Derivation:
        tr(Σ⁻¹ B) = tr[ (Λ⁻¹ - Λ⁻² Kuf'(Kuu + Kuf Λ⁻¹ Kuf')⁻¹ Kuf) B ]
                  = tr(Λ⁻¹ B) - tr( Λ⁻¹ Kuf' ΣQR⁻¹ Kuf Λ⁻¹ B )
                  = tr(Λ⁻¹ B) - tr( Λ⁻¹ Kuf' ΣQR^(-1/2) ΣQR^(-1/2) Kuf Λ⁻¹ B )
                                                        ╰────────────────╯
                                                            ≡ L
                  = tr(Λ⁻¹ B) - tr(L'*L*B)
                  = tr(Λ⁻¹ B) - dot(L, L*B)

    See also: [`\`](@ref).
"""
function trinvAB(A::FullyIndepPDMat, B::Diagonal)
    Λ = A.Λ
    L = whiten(A.ΣQR_PD, A.Kuf * inv(Λ))
    return tr(Λ \ B) - dot(L, L * B)
end

"""
    The matrix determinant lemma states that
        logdet(A+UWV') = logdet(W⁻¹ + V'A⁻¹U) + logdet(W) + logdet(A)
    So for
        Σ ≈ Kuf' Kuu⁻¹ Kuf + Λ
        logdet(Σ) = logdet(Kuu + Kuf Λ⁻¹ Kuf')       + logdet(Kuu⁻¹) + logdet(Λ)
                  = logdet(        ΣQR             ) - logdet(Kuu)   + logdet(Λ)
"""
logdet(a::FullyIndepPDMat) = logdet(a.ΣQR_PD) - logdet(a.Kuu) + logdet(a.Λ)#sum(log.(a.Λ))
function Base.Matrix(a::FullyIndepPDMat)
    Lk = whiten(a.Kuu, a.Kuf)
    Σ = Lk'Lk + a.Λ
    nobs = size(Σ,1)
    return Σ
end

function wrap_cK(cK::FullyIndepPDMat, inducing, ΣQR_PD, Kuu, Kuf, Λ::Diagonal)
    FullyIndepPDMat(inducing, ΣQR_PD, Kuu, Kuf, Λ)
end
function wrap_cK(cK::FullyIndepPDMat, inducing, ΣQR_PD, Kuu, Kuf, Λ::Vector)
    FullyIndepPDMat(inducing, ΣQR_PD, Kuu, Kuf, Diagonal(Λ))
end
"""
    tr(a::FullyIndepPDMat)

    Trace of the FITC approximation to the covariance matrix:

    tr(Σ) = tr(Kuf' Kuu⁻¹ Kuf + Λ)
          = tr(Kuf' Kuu⁻¹ Kuf) + tr(Λ)
          = tr(Kuf' Kuu^{-1/2} Kuu^{-1/2} Kuf) + tr(Λ)
                              ╰──────────────╯
                                 ≡  Lk
          = dot(Lk, Lk) + sum(diag(Λ))
"""
function LinearAlgebra.tr(a::FullyIndepPDMat)
    Lk = whiten(a.Kuu, a.Kuf)
    return tr(a.Λ) + dot(Lk, Lk)
end


"""
    Fully Independent Training Conditional (FITC) covariance strategy.
"""
struct FullyIndepStrat{M<:AbstractMatrix} <: SparseStrategy
    inducing::M
end
SubsetOfRegsStrategy(fitc::FullyIndepStrat) = SubsetOfRegsStrategy(fitc.inducing)
DeterminTrainCondStrat(fitc::FullyIndepStrat) = DeterminTrainCondStrat(fitc.inducing)

function alloc_cK(covstrat::FullyIndepStrat, nobs)
    # The objects that need to be allocated are very similar
    # to SoR, so we'll use that as a starting point:
    SoR = SubsetOfRegsStrategy(covstrat)
    cK_SoR = alloc_cK(SoR, nobs)
    # Additionally, we need to store the diagonal corrections.
    Λ = Diagonal(Vector{Float64}(undef, nobs))

    cK_FITC = FullyIndepPDMat(
        covstrat.inducing,
        cK_SoR.ΣQR_PD,
        cK_SoR.Kuu,
        cK_SoR.Kuf,
        Λ)
    return cK_FITC
end
function update_cK!(cK::FullyIndepPDMat, X::AbstractMatrix, kernel::Kernel,
                    logNoise::Real, kerneldata::KernelData, covstrat::FullyIndepStrat)
    inducing = covstrat.inducing
    Kuu = cK.Kuu
    Kuubuffer = mat(Kuu)
    cov!(Kuubuffer, kernel, inducing)
    Kuubuffer, chol = make_posdef!(Kuubuffer, cholfactors(cK.Kuu))
    Kuu_PD = wrap_cK(cK.Kuu, Kuubuffer, chol)
    Kuf = cov!(cK.Kuf, kernel, inducing, X)
    Kfu = Kuf'

    dim, nobs = size(X)
    Kdiag = [cov_ij(kernel, X, X, EmptyData(), i, i, dim) for i in 1:nobs]
    Qdiag = [invquad(Kuu_PD, Kuf[:,i]) for i in 1:nobs]
    Λ = Diagonal(exp(2*logNoise) .+ Kdiag .- Qdiag)

    ΣQR = Kuf * (Λ \ Kfu) + Kuu
    LinearAlgebra.copytri!(ΣQR, 'U')

    ΣQR, chol = make_posdef!(ΣQR, cholfactors(cK.ΣQR_PD))
    ΣQR_PD = wrap_cK(cK.ΣQR_PD, ΣQR, chol)
    return wrap_cK(cK, inducing, ΣQR_PD, Kuu_PD, Kuf, Λ)
end

#==========================================
  Log-likelihood gradients
===========================================#
function init_precompute(covstrat::FullyIndepStrat, X, y, kernel)
    # here we can re-use the Subset of Regressors pre-computations
    SoR = SubsetOfRegsStrategy(covstrat)
    return init_precompute(SoR, X, y, kernel)
end

"""
dmll_kern!(dmll::AbstractVector, kernel::Kernel, X::AbstractMatrix, cK::AbstractPDMat, kerneldata::KernelData,
                    alpha::AbstractVector, Kuu, Kuf, Kuu⁻¹Kuf, Kuu⁻¹KufΣ⁻¹y, Σ⁻¹Kfu, ∂Kuu, ∂Kuf,
                    covstrat::FullyIndepStrat)
Derivative of the log likelihood under the Fully Independent Training Conditional (FITC) approximation.

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

In the case of the FITC approximation, we have
    Σ = Λ + Qff
    The second component gives
    ∂(Qff) = ∂(Kfu Kuu⁻¹ Kuf)
    which is used by the gradient function for the subset of regressors approximation,
    and so I don't repeat it here.

    the ith element of diag(Kff-Qff) is
    Λi = Kii - Qii + σ²
       = Kii - Kui' Kuu⁻¹ Kui + σ²
    ∂Λi = ∂Kii - Kui' ∂(Kuu⁻¹) Kui - 2 ∂Kui' Kuu⁻¹ Kui
        = ∂Kii + Kui' Kuu⁻¹ ∂(Kuu) Kuu⁻¹ Kui - 2 ∂Kui' Kuu⁻¹ Kui
"""
function dmll_kern!(dmll::AbstractVector, kernel::Kernel, X::AbstractMatrix, 
                    cK::AbstractPDMat, kerneldata::SparseKernelData,
                    alpha::AbstractVector, Kuu, Kuf, Kuu⁻¹Kuf, Kuu⁻¹KufΣ⁻¹y, Σ⁻¹Kfu, ∂Kuu, ∂Kuf,
                    covstrat::FullyIndepStrat)
    # first compute the SoR component
    SoR = SubsetOfRegsStrategy(covstrat)
    dmll_kern!(dmll, kernel, X, cK, kerneldata, alpha,
               Kuu, Kuf, Kuu⁻¹Kuf, Kuu⁻¹KufΣ⁻¹y, Σ⁻¹Kfu, ∂Kuu, ∂Kuf,
               SoR)
    nparams = num_params(kernel)
    dim, nobs = size(X)
    inducing = covstrat.inducing
    for iparam in 1:nparams
        # TODO: the grad_slice! calls here are redundant with the ones in the dmll_kern! call above
        grad_slice!(∂Kuu, kernel, inducing, inducing, kerneldata.Kuu, iparam)
        grad_slice!(∂Kuf, kernel, inducing, X       , kerneldata.Kux1, iparam)

        ∂Λ = Diagonal([
               (dKij_dθp(kernel, X, X, EmptyData(), i, i, iparam, dim) # ∂Kii
                + dot(Kuu⁻¹Kuf[:,i], ∂Kuu * Kuu⁻¹Kuf[:,i])  # Kui' Kuu⁻¹ ∂(Kuu) Kuu⁻¹ Kui
                - 2 * dot(∂Kuf[:,i], Kuu⁻¹Kuf[:,i]))        # -2 ∂Kui' Kuu⁻¹ Kui
               for i in 1:nobs
               ])
        V = dot(alpha, ∂Λ * alpha)
        T = trinvAB(cK, ∂Λ)
        # # FOR DEBUG ONLY
        # Talt = tr(cK \ ∂Λ) # inefficient
        # @show T, Talt
        # @assert isapprox(T, Talt, atol=1e-5)
        # # END DEBUG

        dmll[iparam] += (V-T)/2
    end
    return dmll
end
function dmll_kern!(dmll::AbstractVector, gp::GPBase, precomp::SoRPrecompute, covstrat::FullyIndepStrat)
    return dmll_kern!(dmll, gp.kernel, gp.x, gp.cK, gp.data, gp.alpha,
                      gp.cK.Kuu, gp.cK.Kuf,
                      precomp.Kuu⁻¹Kuf, precomp.Kuu⁻¹KufΣ⁻¹y, precomp.Σ⁻¹Kfu,
                      precomp.∂Kuu, precomp.∂Kuf,
                      covstrat)
end

function dmll_noise(logNoise::Real, cK::FullyIndepPDMat, alpha::AbstractVector)
    Λ = cK.Λ
    Lk = whiten(cK.ΣQR_PD, cK.Kuf) * inv(Λ)
    # # DEBUG
    # nobs = length(alpha)
    # noiseT = sum(1 ./ Λ) - dot(Lk, Lk)
    # Talt = tr(cK \ Matrix(1.0*I, nobs, nobs))
    # @show noiseT, Talt # should be same
    # # END DEBUG
    return exp(2*logNoise) * ( # Jacobian
        dot(alpha, alpha)
        - tr(inv(Λ)) # sum(1 ./ Λ)
        + dot(Lk, Lk)
        )
end
"""
    dmll_noise(gp::GPE, precomp::SoRPrecompute, covstrat::FullyIndepStrat)

∂logp(Y|θ) = 1/2 y' Σ⁻¹ ∂Σ Σ⁻¹ y - 1/2 tr(Σ⁻¹ ∂Σ)

∂Σ = I for derivative wrt σ², so
∂logp(Y|θ) = 1/2 y' Σ⁻¹ Σ⁻¹ y - 1/2 tr(Σ⁻¹)
            = 1/2[ dot(α,α) - tr(Σ⁻¹) ]

We have:
    Σ⁻¹ = Λ⁻¹ - Λ⁻² Kuf'(        ΣQR       )⁻¹ Kuf
Use the identity tr(A'A) = dot(A,A) to get:
    tr(Σ⁻¹) = tr(Λ⁻¹) - dot(Lk, Lk) .
where
    Lk ≡ ΣQR^(-1/2) Kuf Λ⁻¹
"""
function dmll_noise(gp::GPE, precomp::SoRPrecompute, covstrat::FullyIndepStrat)
    dmll_noise(get_value(gp.logNoise), gp.cK, gp.alpha)
end


function get_alpha_u(Ktrain::FullyIndepPDMat, xtrain::AbstractMatrix, ytrain::AbstractVector, meanf::Mean)
    ΣQR_PD = Ktrain.ΣQR_PD
    Kuf = Ktrain.Kuf
    meantrain = mean(meanf, xtrain)
    Λ = Ktrain.Λ
    alpha_u = ΣQR_PD \ (Kuf * (Λ \ (ytrain-meantrain)) )
    return alpha_u
end

"""
predictMVN(xpred::AbstractMatrix, xtrain::AbstractMatrix, ytrain::AbstractVector,
                    kernel::Kernel, meanf::Mean, logNoise::Real,
                    alpha::AbstractVector,
                    covstrat::FullyIndepStrat, Ktrain::FullyIndepPDMat)
    See Quiñonero-Candela and Rasmussen 2005, equations 24b.
    Some derivations can be found below that are not spelled out in the paper.

    Notation: Qab = Kau Kuu⁻¹ Kub
              ΣQR = Kuu + σ⁻² Kuf Kuf'

              x: prediction (test) locations
              f: training (observed) locations
              u: inducing point locations

    We have
        Σ ≈ Kuf' Kuu⁻¹ Kuf + Λ
    By Woodbury
        Σ⁻¹ = Λ⁻¹ - Λ⁻² Kuf'(Kuu + Kuf Λ⁻¹ Kuf')⁻¹ Kuf
            = Λ⁻¹ - Λ⁻² Kuf'(        ΣQR       )⁻¹ Kuf

    The predictive mean can be derived (assuming zero mean function for simplicity)
    μ = Qxf (Qff + Λ)⁻¹ y
      = Kxu Kuu⁻¹ Kuf [Λ⁻¹ - Λ⁻² Kuf' ΣQR⁻¹ Kuf] y   # see Woodbury formula above.
      = Kxu Kuu⁻¹ [ΣQR - Kuf Λ⁻¹ Kfu] ΣQR⁻¹ Kuf Λ⁻¹ y # factoring out common terms
      = Kxu Kuu⁻¹ [Kuu] ΣQR⁻¹ Kuf Λ⁻¹ y               # using definition of ΣQR
      = Kxu ΣQR⁻¹ Kuf Λ⁻¹ y                           # matches equation 16b

    Similarly for the posterior predictive covariance:
    Σ = Σxx - Qxf (Qff + Λ)⁻¹ Qxf'
      = Σxx - Kxu ΣQR⁻¹ Kuf Λ⁻¹ Qxf'                # substituting result from μ
      = Σxx - Kxu ΣQR⁻¹  Kuf Λ⁻¹ Kfu Kuu⁻¹ Kux      # definition of Qxf
      = Σxx - Kxu ΣQR⁻¹ (ΣQR - Kuu) Kuu⁻¹ Kux       # using definition of ΣQR
      = Σxx - Kxu Kuu⁻¹ Kux + Kxu ΣQR⁻¹ Kux         # expanding
      = Σxx - Qxx           + Kxu ΣQR⁻¹ Kux         # definition of Qxx
"""
function predictMVN(xpred::AbstractMatrix,
                    xtrain::AbstractMatrix, ytrain::AbstractVector,
                    kernel::Kernel, meanf::Mean,
                    alpha::AbstractVector,
                    covstrat::FullyIndepStrat, Ktrain::FullyIndepPDMat)
    DTC = DeterminTrainCondStrat(covstrat)
    μ_DTC, Σ_DTC = predictMVN(xpred, xtrain, ytrain, kernel, meanf, alpha, DTC, Ktrain)
    return μ_DTC, Σ_DTC
end


function FITC(x::AbstractMatrix, inducing::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Real)
    covstrat = FullyIndepStrat(inducing)
    GPE(x, y, mean, kernel, logNoise, covstrat)
end
