"""
    Subset of Regressors sparse positive definite matrix.
"""
mutable struct SubsetOfRegressors{T,M<:AbstractMatrix,PD<:AbstractPDMat{T},M2<:AbstractMatrix{T}} <: SparsePDMat{T}
    inducing::M
    ΣQR_PD::PD
    Kuu::M2
    Kuf::M2
    logNoise::Float64
end
size(a::SubsetOfRegressors) = (size(a.Kuf,2), size(a.Kuf,2))
size(a::SubsetOfRegressors, d::Int) = size(a.Kuf,2)
"""
    We have
        Σ ≈ Kuf' Kuu⁻¹ Kuf + σ²I
    By Woodbury
        Σ⁻¹ = σ⁻²I - σ⁻⁴ Kuf'(Kuu + σ⁻² Kuf Kuf')⁻¹ Kuf
            = σ⁻²I - σ⁻⁴ Kuf'(       ΣQR        )⁻¹ Kuf
"""
function \(a::SubsetOfRegressors, x::AbstractArray)
    return exp(-2*a.logNoise)*x - exp(-4*a.logNoise)*a.Kuf'*(a.ΣQR_PD \ (a.Kuf * x))
end
logdet(a::SubsetOfRegressors) = logdet(a.ΣQR_PD) - logdet(a.Kuu) + 2*a.logNoise*size(a,1)

function wrap_cK(cK::SubsetOfRegressors, inducing, ΣQR_PD, Kuu, Kuf, logNoise)
    SubsetOfRegressors(inducing, ΣQR_PD, Kuu, Kuf, logNoise)
end
function update_cK!(gp::GPE{X,Y,M,K,P,D}) where {X<:AbstractMatrix,Y<:AbstractVector,M<:Mean,K<:Kernel,P<:SubsetOfRegressors,D<:EmptyData}
    old_cK = gp.cK
    Kuu = cov!(old_cK.Kuu, gp.kernel, old_cK.inducing)
    Kuf = cov!(old_cK.Kuf, gp.kernel, old_cK.inducing, gp.x)
    Kfu = Kuf'
    
    ΣQR = exp(-2*gp.logNoise) * Kuf * Kfu + Kuu
    LinearAlgebra.copytri!(ΣQR, 'U')
    
    ΣQR, chol = make_posdef!(ΣQR, cholfactors(old_cK.ΣQR_PD))
    ΣQR_PD = wrap_cK(old_cK.ΣQR_PD, ΣQR, chol)
    gp.cK = wrap_cK(gp.cK, old_cK.inducing, ΣQR_PD, Kuu, Kuf, gp.logNoise)
end
function alloc_SoR(nobs, inducing)
    ninducing = size(inducing, 2)
    Kuu  = Matrix{Float64}(undef, ninducing, ninducing)
    Kuf  = Matrix{Float64}(undef, ninducing, nobs)
    ΣQR  = Matrix{Float64}(undef, ninducing, ninducing)
    chol = Matrix{Float64}(undef, ninducing, ninducing)
    cK = SubsetOfRegressors(inducing, 
                            PDMats.PDMat(ΣQR, Cholesky(chol, 'U', 0)), # ΣQR_PD
                            Kuu, Kuf, 42.0)
    return cK
end
function SoR(x::AbstractMatrix, inducing::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Float64)
    nobs = length(y)
    cK = alloc_SoR(nobs, inducing)
    GPE(x, y, mean, kernel, logNoise, EmptyData(), cK)
end

"""
    See Quiñonero-Candela and Rasmussen 2005, equations 16b.
    Some derivations can be found below that are not spelled out in the paper.

    Notation: Qab = Kau Kuu⁻¹ Kub
              ΣQR = Kuu + σ⁻² Kuf Kuf'

              x: test location
              f: training locations
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
function _predict(gp::GPE{X,Y,M,K,P,D}, x::AbstractMatrix)  where {X<:AbstractMatrix,Y<:AbstractVector,M<:Mean,K<:Kernel,P<:SubsetOfRegressors,D<:EmptyData}
    ΣQR_PD = gp.cK.ΣQR_PD
    inducing = gp.cK.inducing
    Kuf = gp.cK.Kuf
    
    Kxu = cov(gp.kernel, x, inducing)
    Kux = Matrix(Kxu') # annoying memory allocation
    
    meanx = mean(gp.mean, x)
    meanf = mean(gp.mean, gp.x)
    alpha_u = ΣQR_PD \ (Kuf * (gp.y-meanf))
    mu = meanx + exp(-2*gp.logNoise) * Kxu * alpha_u
    
    Sigma_raw = Kxu * (ΣQR_PD \ Kux)
    LinearAlgebra.copytri!(Sigma_raw, 'U')
    m, chol = make_posdef!(Sigma_raw)
    return mu, PDMat(m, chol)
end
