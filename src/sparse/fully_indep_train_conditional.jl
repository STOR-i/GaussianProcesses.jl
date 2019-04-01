#========================================
 Fully Independent Training Conditional
=========================================#

"""
    Positive Definite Matrix for Fully Independent Training Conditional approximation.
"""
mutable struct FullyIndepPDMat{T,M<:AbstractMatrix,PD<:AbstractPDMat{T},M2<:AbstractMatrix{T}} <: SparsePDMat{T}
    inducing::M
    ΣQR_PD::PD
    Kuu::PD
    Kuf::M2
    Λ::Vector{Float64}
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
    return x./a.Λ - (a.Kuf'*(a.ΣQR_PD \ (a.Kuf * x))) ./ a.Λ.^2
end
"""
    The matrix determinant lemma states that
        logdet(A+UWV') = logdet(W⁻¹ + V'A⁻¹U) + logdet(W) + logdet(A)
    So for
        Σ ≈ Kuf' Kuu⁻¹ Kuf + (Λ+σ²I)
        logdet(Σ) = logdet(Kuu + Kuf (Λ+σ²I)⁻¹ Kuf') + logdet(Kuu⁻¹) + logdet(Λ+σ²I)
                  = logdet(        ΣQR             ) - logdet(Kuu)   + logdet(Λ+σ²I)
"""
logdet(a::FullyIndepPDMat) = logdet(a.ΣQR_PD) - logdet(a.Kuu) + sum(log.(a.Λ))

function wrap_cK(cK::FullyIndepPDMat, inducing, ΣQR_PD, Kuu, Kuf, Λ::Vector)
    FullyIndepPDMat(inducing, ΣQR_PD, Kuu, Kuf, Λ)
end
function LinearAlgebra.tr(a::FullyIndepPDMat)
    Lk = whiten(a.Kuu, a.Kuf)
    # log(sum(a.Λ)) + dot(a.Kuf, a.Kuu \ a.Kuf) # TODO: there may be a shortcut here
    return log(sum(a.Λ)) + dot(Lk, Lk)
end


"""
    Fully Independent Training Conditional (FITC) covariance strategy.
"""
struct FullyIndepStrat{M<:AbstractMatrix} <: CovarianceStrategy
    inducing::M
end
SubsetOfRegsStrategy(fitc::FullyIndepStrat) = SubsetOfRegsStrategy(fitc.inducing)

function alloc_cK(covstrat::FullyIndepStrat, nobs)
    # The objects that need to be allocated are very similar
    # to SoR, so we'll use that as a starting point:
    SoR = SubsetOfRegsStrategy(covstrat)
    cK_SoR = alloc_cK(SoR, nobs)
    # Additionally, we need to store the diagonal corrections.
    Λ = Vector{Float64}(undef, nobs)

    cK_FITC = FullyIndepPDMat(
        covstrat.inducing,
        cK_SoR.ΣQR_PD,
        cK_SoR.Kuu,
        cK_SoR.Kuf,
        Λ)
    return cK_FITC
end
function update_cK!(cK::FullyIndepPDMat, X::AbstractMatrix, kernel::Kernel, 
                    logNoise::Real, data::KernelData, covstrat::FullyIndepStrat)
    inducing = covstrat.inducing
    Kuu = cK.Kuu
    Kuubuffer = mat(Kuu)
    cov!(Kuubuffer, kernel, inducing)
    Kuubuffer, chol = make_posdef!(Kuubuffer, cholfactors(cK.Kuu))
    Kuu_PD = wrap_cK(cK.Kuu, Kuubuffer, chol)
    Kuf = cov!(cK.Kuf, kernel, inducing, X)
    Kfu = Kuf'

    dim, nobs = size(X)
    Kdiag = [cov_ij(kernel, X, X, data, i, i, dim) for i in 1:nobs]
    Qdiag = [invquad(Kuu_PD, Kuf[:,i]) for i in 1:nobs]
    Λ = exp(2*logNoise) .+ Kdiag.-Qdiag
    
    ΣQR = (Kuf ./ Λ') * Kfu + Kuu
    LinearAlgebra.copytri!(ΣQR, 'U')
    
    ΣQR, chol = make_posdef!(ΣQR, cholfactors(cK.ΣQR_PD))
    ΣQR_PD = wrap_cK(cK.ΣQR_PD, ΣQR, chol)
    return wrap_cK(cK, inducing, ΣQR_PD, Kuu_PD, Kuf, Λ)
end

#==========================================
  Log-likelihood gradients
===========================================#
function init_precompute(covstrat::FullyIndepStrat, X, y, k)
    # here we can re-use the Subset of Regressors pre-computations
    SoR = SubsetOfRegsStrategy(covstrat)
    return init_precompute(SoR, X, y, k)
end
    
function dmll_kern!(dmll::AbstractVector, gp::GPBase, precomp::SoRPrecompute, covstrat::FullyIndepStrat)
    SoR = SubsetOfRegsStrategy(covstrat)
    cK = gp.cK
    return dmll_kern!(dmll, gp.kernel, gp.x, cK, gp.data, gp.alpha, 
                      cK.Kuu, cK.Kuf,
                      precomp.Kuu⁻¹Kuf, precomp.Kuu⁻¹KufΣ⁻¹y, precomp.Σ⁻¹Kfu,
                      precomp.∂Kuu, precomp.∂Kfu,
                      SoR)
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
    Lk ≡ ΣQR^(-1/2) Kuf Λ⁻¹
which gives
    tr(Σ⁻¹) = tr(Λ⁻¹) - dot(Lk, Lk) .
"""
function dmll_noise(gp::GPE, precomp::SoRPrecompute, covstrat::FullyIndepStrat)
    nobs = gp.nobs
    cK = gp.cK
    Λ = cK.Λ
    Lk = whiten(cK.ΣQR_PD, cK.Kuf ./ Λ')
    return exp(2*gp.logNoise) * ( # Jacobian
        dot(gp.alpha, gp.alpha) 
        - sum(1 ./ Λ)
        + dot(Lk, Lk)
        )
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
function predictMVN(xpred::AbstractMatrix, xtrain::AbstractMatrix, ytrain::AbstractVector, 
                    kernel::Kernel, meanf::Mean, logNoise::Real,
                    alpha::AbstractVector,
                    covstrat::FullyIndepStrat, Ktrain::FullyIndepPDMat)
    ΣQR_PD = Ktrain.ΣQR_PD
    inducing = covstrat.inducing
    Kuf = Ktrain.Kuf
    Kuu = Ktrain.Kuu
    
    Kux = cov(kernel, inducing, xpred)
    
    meanx = mean(meanf, xpred)
    meanf = mean(meanf, xtrain)
    alpha_u = ΣQR_PD \ (Kuf * ((ytrain-meanf) ./ Ktrain.Λ) )
    mupred = meanx + (Kux' * alpha_u)
    
    Lck = PDMats.whiten(ΣQR_PD, Kux)
    Σ_SoR = Lck'Lck # Kux' * (ΣQR_PD \ Kux)
    LinearAlgebra.copytri!(Σ_SoR, 'U')

    Qxx = PDMats.Xt_invA_X(Kuu, Kux)
    Σxx = cov(kernel, xpred, xpred)

    Σ_FITC = Σxx - Qxx + Σ_SoR
    return mupred, Σ_FITC
end


function FITC(x::AbstractMatrix, inducing::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Real)
    nobs = length(y)
    covstrat = FullyIndepStrat(inducing)
    cK = alloc_cK(covstrat, nobs)
    GPE(x, y, mean, kernel, logNoise, covstrat, EmptyData(), cK)
end
