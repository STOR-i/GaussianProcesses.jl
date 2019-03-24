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

#========================================
 Subset of Regressors strategy
=========================================#

struct SubsetOfRegsStrategy{M<:AbstractMatrix} <: CovarianceStrategy
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
                    logNoise::Real, data::KernelData, covstrat::SubsetOfRegsStrategy)
    inducing = covstrat.inducing
    Kuu = cK.Kuu
    Kuubuffer = mat(Kuu)
    cov!(Kuubuffer, kernel, inducing)
    Kuubuffer, chol = make_posdef!(Kuubuffer, cholfactors(cK.Kuu))
    Kuu_PD = wrap_cK(cK.Kuu, Kuubuffer, chol)
    Kuf = cov!(cK.Kuf, kernel, inducing, x)
    Kfu = Kuf'
    
    ΣQR = exp(-2*logNoise) * Kuf * Kfu + Kuu
    LinearAlgebra.copytri!(ΣQR, 'U')
    
    ΣQR, chol = make_posdef!(ΣQR, cholfactors(cK.ΣQR_PD))
    ΣQR_PD = wrap_cK(cK.ΣQR_PD, ΣQR, chol)
    return wrap_cK(cK, inducing, ΣQR_PD, Kuu_PD, Kuf, logNoise)
end


function dmll_kern!(dmll::AbstractVector, k::Kernel, X::AbstractMatrix, cK::SubsetOfRegsPDMat, data::KernelData, ααinvcKI::AbstractMatrix, covstrat::SubsetOfRegsStrategy)
    dim, nobs = size(X)
    inducing = covstrat.inducing
    ninducing = size(inducing, 2)
    nparams = num_params(k)
    @assert nparams == length(dmll)
    dK_buffer = Vector{Float64}(undef, nparams)
    dmll[:] .= 0.0
    Kuf = cK.Kuf
    Kuu = cK.Kuu
    ∂Kuu = Matrix{Float64}(undef, ninducing, ninducing)
    ∂Kfu = Matrix{Float64}(undef, nobs, ninducing)
    term = Kuu \ Kuf # Kuu⁻¹Kuf appears in multiple places, so pre-compute
    for iparam in 1:nparams
        grad_slice!(∂Kuu, k, inducing, inducing, EmptyData(), iparam)
        grad_slice!(∂Kfu, k, X, inducing,        EmptyData(), iparam)
        ∂R = ∂Kfu * term
        @inbounds for i in 1:nobs
            ∂R[i,i] *= 2
            for j in 1:(i-1)
                s = ∂R[i,j] + ∂R[j,i]
                ∂R[i,j] = s
                ∂R[j,i] = s
            end
        end
        ∂R -= term' * ∂Kuu * term
        dmll[iparam] = dot(ααinvcKI, ∂R)/2
    end
    return dmll
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
function predictMVN(xpred::AbstractMatrix, xtrain::AbstractMatrix, ytrain::AbstractVector, 
                    kernel::Kernel, meanf::Mean, logNoise::Real,
                    alpha::AbstractVector,
                    covstrat::SubsetOfRegsStrategy, Ktrain::SubsetOfRegsPDMat)
    ΣQR_PD = Ktrain.ΣQR_PD
    inducing = covstrat.inducing
    Kuf = Ktrain.Kuf
    
    Kux = cov(kernel, inducing, xpred)
    
    meanx = mean(meanf, xpred)
    meanf = mean(meanf, xtrain)
    alpha_u = ΣQR_PD \ (Kuf * (ytrain-meanf))
    mupred = meanx + exp(-2*logNoise) * (Kux' * alpha_u)
    
    Lck = PDMats.whiten(ΣQR_PD, Kux)
    Σpred = Lck'Lck # Kux' * (ΣQR_PD \ Kux)
    LinearAlgebra.copytri!(Σpred, 'U')
    return mupred, Σpred
end


function SoR(x::AbstractMatrix, inducing::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Real)
    nobs = length(y)
    covstrat = SubsetOfRegsStrategy(inducing)
    cK = alloc_cK(covstrat, nobs)
    GPE(x, y, mean, kernel, logNoise, covstrat, EmptyData(), cK)
end

