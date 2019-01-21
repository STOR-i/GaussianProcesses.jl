#==========================================================
 Sparse Positive Definite Matrix for Subset of Regressors
===========================================================#
"""
    Subset of Regressors sparse positive definite matrix.
"""
mutable struct SubsetOfRegsPDMat{T,M<:AbstractMatrix,PD<:AbstractPDMat{T},M2<:AbstractMatrix{T}} <: SparsePDMat{T}
    inducing::M
    ΣQR_PD::PD
    Kuu::M2
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
function \(a::SubsetOfRegsPDMat, x::AbstractArray)
    return exp(-2*a.logNoise)*x - exp(-4*a.logNoise)*a.Kuf'*(a.ΣQR_PD \ (a.Kuf * x))
end
logdet(a::SubsetOfRegsPDMat) = logdet(a.ΣQR_PD) - logdet(a.Kuu) + 2*a.logNoise*size(a,1)

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
    Kuf  = Matrix{Float64}(undef, ninducing, nobs)
    ΣQR  = Matrix{Float64}(undef, ninducing, ninducing)
    chol = Matrix{Float64}(undef, ninducing, ninducing)
    cK = SubsetOfRegsPDMat(inducing, 
                            PDMats.PDMat(ΣQR, Cholesky(chol, 'U', 0)), # ΣQR_PD
                            Kuu, Kuf, 42.0)
    return cK
end
function update_cK!(covstrat::SubsetOfRegsStrategy, cK::SubsetOfRegsPDMat, x::AbstractMatrix, kernel::Kernel, logNoise::Real, data::KernelData)
    inducing = covstrat.inducing
    Kuu = cov!(cK.Kuu, kernel, inducing)
    Kuf = cov!(cK.Kuf, kernel, inducing, x)
    Kfu = Kuf'
    
    ΣQR = exp(-2*logNoise) * Kuf * Kfu + Kuu
    LinearAlgebra.copytri!(ΣQR, 'U')
    
    ΣQR, chol = make_posdef!(ΣQR, cholfactors(cK.ΣQR_PD))
    ΣQR_PD = wrap_cK(cK.ΣQR_PD, ΣQR, chol)
    return wrap_cK(cK, inducing, ΣQR_PD, Kuu, Kuf, logNoise)
end

"""
    See Quiñonero-Candela and Rasmussen 2005, equations 16b.
    Some derivations can be found below that are not spelled out in the paper.

    Notation: Qab = Kau Kuu⁻¹ Kub
              ΣQR = Kuu + σ⁻² Kuf Kuf'

              x: prediction (test) locations
              f: training (observed) locations
              u: inducing point locations

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
                    kernel::Kernel, meanf::Mean, logNoise::Float64,
                    alpha::AbstractVector,
                    covstrat::SubsetOfRegsStrategy, Ktrain::SubsetOfRegsPDMat)
    ΣQR_PD = Ktrain.ΣQR_PD
    inducing = covstrat.inducing
    Kuf = Ktrain.Kuf
    
    Kxu = cov(kernel, xpred, inducing)
    Kux = Matrix(Kxu') # annoying memory allocation
    
    meanx = mean(meanf, xpred)
    meanf = mean(meanf, xtrain)
    alpha_u = ΣQR_PD \ (Kuf * (ytrain-meanf))
    mupred = meanx + exp(-2*logNoise) * Kxu * alpha_u
    
    Σpred = Kxu * (ΣQR_PD \ Kux)
    LinearAlgebra.copytri!(Σpred, 'U')
    return mupred, Σpred
end


function SoR(x::AbstractMatrix, inducing::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Float64)
    nobs = length(y)
    covstrat = SubsetOfRegsStrategy(inducing)
    cK = alloc_cK(covstrat, nobs)
    GPE(x, y, mean, kernel, logNoise, covstrat, EmptyData(), cK)
end

