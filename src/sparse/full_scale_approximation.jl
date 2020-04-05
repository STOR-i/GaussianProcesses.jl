import Base: +
import LinearAlgebra: tr
const BlockIndices = Vector{<:AbstractVector{Int}}

struct BlockDiagPDMat{T, V<:BlockIndices, PD<:AbstractPDMat{T}} <: AbstractPDMat{T}
    blockindices::V
    blockPD::Vector{PD}
end
function BlockDiagPDMat(blockindices::BlockIndices)
    blockPD = [alloc_cK(length(ind)) for ind in blockindices]
    return BlockDiagPDMat(blockindices, blockPD)
end
size(a::BlockDiagPDMat) = (sum(length,a.blockindices), sum(length,a.blockindices))
size(a::BlockDiagPDMat, d::Int) = sum(length,a.blockindices)
function \(a::BlockDiagPDMat, x::AbstractVector)
    out = similar(x)
    for (pd,ind) in zip(a.blockPD,a.blockindices)
        out[ind] = pd \ x[ind]
    end
    return out
end
function \(a::BlockDiagPDMat, x::AbstractMatrix)
    out = similar(x)
    for (pd,ind) in zip(a.blockPD,a.blockindices)
        out[ind,:] = pd \ x[ind,:]
    end
    return out
end
\(a::PDMat, x::UniformScaling) = a.chol \ Diagonal{Float64}(x, size(a,1))
\(a::PDMat, x::Diagonal) = a.chol \ x
function \(a::BlockDiagPDMat, x::Diagonal)
    out = zeros(size(x)) # unfortunate! should maybe be block diagonal matrix.
    for (pd,ind) in zip(a.blockPD,a.blockindices)
        out[ind,ind] = pd \ Diagonal(x.diag[ind])
    end
    return out
end
logdet(a::BlockDiagPDMat) = sum(logdet, a.blockPD)
tr(pd::AbstractPDMat) = tr(mat(pd))
tr(a::BlockDiagPDMat) = sum(tr, a.blockPD)
"""
    trinv(pd::AbstractPDMat)

Trace of the inverse of a positive definite matrix.
"""
function trinv(pd::AbstractPDMat)
    return tr(pd \ I)
    # Lk = inv(cholfactors(pd))
    # return dot(Lk, Lk)
end
"""
    trinv(a::BlockDiagPDMat)

Trace of the inverse of a block diagonal positive definite matrix.

This is obtained as the sum of the traces of the inverse of each block.
"""
trinv(a::BlockDiagPDMat) = sum(trinv, a.blockPD)
function add!(a::AbstractMatrix, b::BlockDiagPDMat)
    for (pd,ind) in zip(a.blockPD,a.blockindices)
        block = @view(a[ind,ind])
        block[:,:] += mat(pd)
    end
    return a
end
+(a::AbstractMatrix, b::BlockDiagPDMat) = add!(copy(a), b)
function Base.Matrix(a::BlockDiagPDMat)
    K = zeros(size(a))
    for (pd,ind) in zip(a.blockPD,a.blockindices)
        K[ind, ind] = mat(pd)
    end
    return K # + Diagonal(repeat([1e-10], size(K, 1)))
end
mat(a::BlockDiagPDMat) = Matrix(a)

"""
    Positive Definite Matrix for Full Scale Approximation.
"""
mutable struct FullScalePDMat{T,M<:AbstractMatrix,PD<:AbstractPDMat{T},M2<:AbstractMatrix{T},PD2<:BlockDiagPDMat} <: SparsePDMat{T}
    inducing::M
    ΣQR_PD::PD
    Kuu::PD
    Kuf::M2
    Λ::PD2
end
size(a::FullScalePDMat) = (size(a.Kuf,2), size(a.Kuf,2))
size(a::FullScalePDMat, d::Int) = size(a.Kuf,2)
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
        Σ⁻¹ = Λ⁻¹ - Λ⁻¹ Kuf'(Kuu + Kuf Λ⁻¹ Kuf')⁻¹ Kuf Λ⁻¹
            = Λ⁻¹ - Λ⁻¹ Kuf'(        ΣQR       )⁻¹ Kuf Λ⁻¹
"""
function \(a::FullScalePDMat, x)
    Lk = whiten(a.ΣQR_PD, a.Kuf)
    return a.Λ \ (x .- Lk' * (Lk * (a.Λ \ x)))
end

"""
    trinvAB(A::FullScalePDMat, B::Diagonal)

    Computes tr(A⁻¹ B) efficiently under the FSA approximation:
        Σ ≈ Kuf' Kuu⁻¹ Kuf + Λ

    Derivation:
        tr(Σ⁻¹ B) = tr[ (Λ⁻¹ - Λ⁻² Kuf'(Kuu + Kuf Λ⁻¹ Kuf')⁻¹ Kuf) B ]
                  = tr(Λ⁻¹ B) - tr( Λ⁻¹ Kuf' ΣQR⁻¹ Kuf Λ⁻¹ B )
                  = tr(Λ⁻¹ B) - tr( Λ⁻¹ Kuf' ΣQR^(-1/2) ΣQR^(-1/2) Kuf Λ⁻¹ B )
                                                        ╰────────────────╯
                                                            ≡ L'
                  = tr(Λ⁻¹ B) - tr(L*L'*B)
                  = tr(Λ⁻¹ B) - dot(L, B*L)

    See also: [`\`](@ref).
"""
function trinvAB(A::FullScalePDMat, B::Diagonal)
    Λ = A.Λ
    ΣPDinvKfu = convert(Matrix{Float64}, whiten(A.ΣQR_PD, A.Kuf)')
    L = Λ \ ΣPDinvKfu # is there a better way?
    return tr(Λ \ B) - dot(L, B * L)
end

"""
    The matrix determinant lemma states that
        logdet(A+UWV') = logdet(W⁻¹ + V'A⁻¹U) + logdet(W) + logdet(A)
    So for
        Σ ≈ Kuf' Kuu⁻¹ Kuf + Λ
        logdet(Σ) = logdet(Kuu + Kuf Λ⁻¹ Kuf')       + logdet(Kuu⁻¹) + logdet(Λ)
                  = logdet(        ΣQR             ) - logdet(Kuu)   + logdet(Λ)
"""
logdet(a::FullScalePDMat) = logdet(a.ΣQR_PD) - logdet(a.Kuu) + logdet(a.Λ)#sum(log.(a.Λ))
function Base.Matrix(a::FullScalePDMat)
    Lk = whiten(a.Kuu, a.Kuf)
    Σ = Lk'Lk + mat(a.Λ)
    nobs = size(Σ,1)
    return Σ
end

function wrap_cK(cK::FullScalePDMat, inducing, ΣQR_PD, Kuu, Kuf, Λ::BlockDiagPDMat)
    FullScalePDMat(inducing, ΣQR_PD, Kuu, Kuf, Λ)
end
"""
    tr(a::FullScalePDMat)

    Trace of the FSA approximation to the covariance matrix:

    tr(Σ) = tr(Kuf' Kuu⁻¹ Kuf + Λ)
          = tr(Kuf' Kuu⁻¹ Kuf) + tr(Λ)
          = tr(Kuf' Kuu^{-1/2} Kuu^{-1/2} Kuf) + tr(Λ)
                              ╰──────────────╯
                                 ≡  Lk
          = dot(Lk, Lk) + sum(diag(Λ))
"""
function LinearAlgebra.tr(a::FullScalePDMat)
    Lk = whiten(a.Kuu, a.Kuf)
    return tr(a.Λ) + dot(Lk, Lk)
end


"""
    Fully Independent Training Conditional (FSA) covariance strategy.
"""
struct FullScaleApproxStrat{M<:AbstractMatrix, V<:BlockIndices} <: SparseStrategy
    inducing::M
    blockindices::V
end
SubsetOfRegsStrategy(fsa::FullScaleApproxStrat) = SubsetOfRegsStrategy(fsa.inducing)
DeterminTrainCondStrat(fsa::FullScaleApproxStrat) = DeterminTrainCondStrat(fsa.inducing)

function alloc_cK(covstrat::FullScaleApproxStrat, nobs)
    # The objects that need to be allocated are very similar
    # to SoR, so we'll use that as a starting point:
    SoR = SubsetOfRegsStrategy(covstrat)
    cK_SoR = alloc_cK(SoR, nobs)
    # Additionally, we need to store the diagonal corrections.
    Λ = BlockDiagPDMat(covstrat.blockindices)

    cK_fsa = FullScalePDMat(
        covstrat.inducing,
        cK_SoR.ΣQR_PD,
        cK_SoR.Kuu,
        cK_SoR.Kuf,
        Λ)
    return cK_fsa
end
function update_cK!(cK::FullScalePDMat, X::AbstractMatrix, kernel::Kernel,
                    logNoise::Real, kerneldata::KernelData, covstrat::FullScaleApproxStrat)
    inducing = covstrat.inducing
    blockindices = covstrat.blockindices

    Kuu = cK.Kuu
    Kuubuffer = mat(Kuu)
    cov!(Kuubuffer, kernel, inducing)
    Kuubuffer, chol = make_posdef!(Kuubuffer, cholfactors(cK.Kuu))
    Kuu_PD = wrap_cK(cK.Kuu, Kuubuffer, chol)
    Kuf = cov!(cK.Kuf, kernel, inducing, X)
    Kfu = Kuf'

    dim, nobs = size(X)
    Λ = cK.Λ
    @assert Λ.blockindices == blockindices
    for (pd,ind) in zip(Λ.blockPD,Λ.blockindices)
        Xblock = @view(X[:,ind])
        Kblock = cov(kernel, Xblock)
        Qchol = whiten(Kuu_PD, Kuf[:,ind])

        blockbuffer = mat(pd)
        # TODO: could use memory more efficiently
        blockbuffer[:,:] = Kblock - Qchol'Qchol + exp(2*logNoise)*I
        blockbuffer, chol = make_posdef!(blockbuffer, cholfactors(pd))
    end

    ΣQR = Kuf * (Λ \ Kfu) + Kuu
    LinearAlgebra.copytri!(ΣQR, 'U')

    ΣQR, chol = make_posdef!(ΣQR, cholfactors(cK.ΣQR_PD))
    ΣQR_PD = wrap_cK(cK.ΣQR_PD, ΣQR, chol)
    return wrap_cK(cK, inducing, ΣQR_PD, Kuu_PD, Kuf, Λ)
end

#==========================================
  Log-likelihood gradients
===========================================#
function init_precompute(covstrat::FullScaleApproxStrat, X, y, kernel)
    # here we can re-use the Subset of Regressors pre-computations
    SoR = SubsetOfRegsStrategy(covstrat)
    return init_precompute(SoR, X, y, kernel)
end

"""
dmll_kern!(dmll::AbstractVector, kernel::Kernel, X::AbstractMatrix, cK::AbstractPDMat, kerneldata::KernelData,
                    alpha::AbstractVector, Kuu, Kuf, Kuu⁻¹Kuf, Kuu⁻¹KufΣ⁻¹y, Σ⁻¹Kfu, ∂Kuu, ∂Kuf,
                    covstrat::FullScaleApproxStrat)
Derivative of the log likelihood under the Fully Independent Training Conditional (fsa) approximation.

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

In the case of the FSA approximation, we have
    Σ = Λ + Qff
    The second component gives
    ∂(Qff) = ∂(Kfu Kuu⁻¹ Kuf)
    which is used by the gradient function for the subset of regressors approximation,
    and so I don't repeat it here.

    for ∂Λ we have (with `i` indexing each block)
    Λi = Ki - Qi + σ²I
       = Ki - Kui' Kuu⁻¹ Kui + σ²
    ∂Λi = ∂Ki - Kui' ∂(Kuu⁻¹) Kui - 2 ∂Kui' Kuu⁻¹ Kui
        = ∂Ki + Kui' Kuu⁻¹ ∂(Kuu) Kuu⁻¹ Kui - 2 ∂Kui' Kuu⁻¹ Kui
"""
function dmll_kern!(dmll::AbstractVector, kernel::Kernel, X::AbstractMatrix, 
                    cK::AbstractPDMat, kerneldata::SparseKernelData,
                    alpha::AbstractVector, Kuu, Kuf, Kuu⁻¹Kuf, Kuu⁻¹KufΣ⁻¹y, Σ⁻¹Kfu, ∂Kuu, ∂Kuf,
                    covstrat::FullScaleApproxStrat)
    # first compute the SoR component
    SoR = SubsetOfRegsStrategy(covstrat)
    dmll_kern!(dmll, kernel, X, cK, kerneldata, alpha,
               Kuu, Kuf, Kuu⁻¹Kuf, Kuu⁻¹KufΣ⁻¹y, Σ⁻¹Kfu, ∂Kuu, ∂Kuf,
               SoR)
    nparams = num_params(kernel)
    dim, nobs = size(X)
    inducing = covstrat.inducing
    Λ = cK.Λ
    for iparam in 1:nparams
        # TODO: the grad_slice! calls here are redundant with the ones in the dmll_kern! call above
        grad_slice!(∂Kuu, kernel, inducing, inducing, kerneldata.Kuu, iparam)
        grad_slice!(∂Kuf, kernel, inducing, X       , kerneldata.Kux1, iparam)
        V, T = 0.0, 0.0
        # ∂Λdense = zeros(nobs, nobs)
        for (pd,ind) in zip(Λ.blockPD,Λ.blockindices)
            Kuu⁻¹Kui = Kuu⁻¹Kuf[:,ind]
            ∂Ki = similar(mat(pd))
            Kui, ∂Kui = Kuf[:,ind], ∂Kuf[:,ind]
            Xi = X[:,ind]
            grad_slice!(∂Ki, kernel, Xi, Xi, EmptyData(), iparam)
            ∂Λi = (∂Ki
                  + Kuu⁻¹Kui' * ∂Kuu * Kuu⁻¹Kui
                  - 2 * ∂Kui' * Kuu⁻¹Kui)
            V += dot(alpha[ind], ∂Λi*alpha[ind])

            # see trinvAB to understand next 3 lines
            ΣPDinvKiu = convert(Matrix{Float64}, whiten(cK.ΣQR_PD, Kui)')
            L = pd \ ΣPDinvKiu
            T += trinvAB(pd, ∂Λi) - dot(L, ∂Λi*L)
            # ∂Λdense[ind,ind] = ∂Λi
        end
        # # FOR DEBUG ONLY
        # Talt = tr(cK \ ∂Λdense) # inefficient
        # Valt = dot(alpha, ∂Λdense * alpha)
        # @show T, Talt
        # @show V, Valt
        # # END DEBUG
        dmll[iparam] += (V-T)/2
    end
    return dmll
end
function dmll_kern!(dmll::AbstractVector, gp::GPBase, precomp::SoRPrecompute, covstrat::FullScaleApproxStrat)
    return dmll_kern!(dmll, gp.kernel, gp.x, gp.cK, gp.data, gp.alpha,
                      gp.cK.Kuu, gp.cK.Kuf,
                      precomp.Kuu⁻¹Kuf, precomp.Kuu⁻¹KufΣ⁻¹y, precomp.Σ⁻¹Kfu,
                      precomp.∂Kuu, precomp.∂Kuf,
                      covstrat)
end

function dmll_noise(logNoise::Real, cK::FullScalePDMat, alpha::AbstractVector)
    Λ = cK.Λ
    Lk = Λ \ whiten(cK.ΣQR_PD, cK.Kuf)'
    # # DEBUG
    # noiseT = sum(1 ./ Λ) - dot(Lk, Lk)
    # nobs = length(alpha)
    # Talt = tr(cK \ Matrix(1.0*I, nobs, nobs))
    # @show noiseT, Talt # should be same
    # # END DEBUG
    # @show tr(inv(Matrix(cK)))
    # @show trinv(Λ) - dot(Lk, Lk)
    return exp(2*logNoise) * ( # Jacobian
        dot(alpha, alpha)
        - trinv(Λ) # sum(1 ./ Λ)
        + dot(Lk, Lk)
        )
end
"""
    dmll_noise(gp::GPE, precomp::SoRPrecompute, covstrat::FullScaleApproxStrat)

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
function dmll_noise(gp::GPE, precomp::SoRPrecompute, covstrat::FullScaleApproxStrat)
    dmll_noise(get_value(gp.logNoise), gp.cK, gp.alpha)
end


function get_alpha_u(Ktrain::FullScalePDMat, xtrain::AbstractMatrix, ytrain::AbstractVector, meanf::Mean)
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
                    covstrat::FullScaleApproxStrat, Ktrain::FullScalePDMat)
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
    μ = (Qxf+Λxf) (Qff + Λff)⁻¹ y
      = Qxf (Qff + Λff)⁻¹ y   + Λxf (Qff + Λff)⁻¹ y
        ╰─────────────────╯
            same as FITC
      = Kxu ΣQR⁻¹ Kuf Λff⁻¹ y + Λxf (Qff + Λff)⁻¹ y
           ╰────────────────╯       ╰─────────────╯
              ≡ alpha_u                ≡ alpha

    Similarly for the posterior predictive covariance:
    Σ = Σxx - (Qxf+Λxf) (Qff + Λff)⁻¹ (Qxf+Λxf)'
      = Σxx - Kxu ΣQR⁻¹ Kuf Λ⁻¹ Qxf'                # substituting result from μ
      = Σxx - Kxu ΣQR⁻¹  Kuf Λ⁻¹ Kfu Kuu⁻¹ Kux      # definition of Qxf
      = Σxx - Kxu ΣQR⁻¹ (ΣQR - Kuu) Kuu⁻¹ Kux       # using definition of ΣQR
      = Σxx - Kxu Kuu⁻¹ Kux + Kxu ΣQR⁻¹ Kux         # expanding
      = Σxx - Qxx           + Kxu ΣQR⁻¹ Kux         # definition of Qxx
"""
function predictMVN(xpred::AbstractMatrix, blockindpred::BlockIndices,
                    xtrain::AbstractMatrix, ytrain::AbstractVector,
                    kernel::Kernel, meanf::Mean,
                    alpha::AbstractVector,
                    covstrat::FullScaleApproxStrat, Ktrain::FullScalePDMat)
    blockindtrain = covstrat.blockindices
    ΣQR_PD = Ktrain.ΣQR_PD
    inducing = covstrat.inducing

    Kux = cov(kernel, inducing, xpred)

    nx = size(xpred,2)
    nf = size(xtrain,2)
    # TODO: Λxf is fairly sparse: it should be possible to avoid its construction
    Λxf = zeros(nx, nf)
    for (predblock, trainblock) in zip(blockindpred, blockindtrain)
        Xpredblock, Xtrainblock = xpred[:,predblock], xtrain[:,trainblock]
        sparseblockdata = SparseKernelData(kernel, inducing, Xpredblock, Xtrainblock)
        denseblockdata = KernelData(kernel, Xpredblock, Xtrainblock)
        Kxf_block = cov(kernel, Xpredblock, Xtrainblock, denseblockdata)
        Qxf_block = getQab(Ktrain, kernel, Xpredblock, Xtrainblock, sparseblockdata)
        Λxf[predblock,trainblock] = Kxf_block - Qxf_block
    end

    meanx = mean(meanf, xpred)
    alpha_u = get_alpha_u(Ktrain, xtrain, ytrain, meanf)
    mupred = meanx + (Kux' * alpha_u) + Λxf * alpha

    sparsedata = SparseKernelData(kernel, inducing, xpred, xtrain)
    Qxf = getQab(Ktrain, kernel, xpred, xtrain, sparsedata)
    ΛplusQxf = Qxf + Λxf
    Σxx = cov(kernel, xpred, xpred)
    Σ_FSA = Σxx - ΛplusQxf * (Ktrain \ ΛplusQxf')
    return mupred, Σ_FSA
end

"""
    predict_f(gp::GPBase, X::Matrix{Float64}[; full_cov::Bool = false])

Return posterior mean and variance of the Gaussian Process `gp` at specfic points which are
given as columns of matrix `X`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""
function predict_f(gp::GPBase, x::AbstractMatrix, blockindpred::BlockIndices; full_cov::Bool=false)
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
    if full_cov
        return predict_full(gp, x, blockindpred)
    else
        ## Calculate prediction for each point independently
        μ = Array{eltype(x)}(undef, size(x,2))
        σ2 = similar(μ)
        iblock = Vector{Int}(undef, size(x,2))
        for (j,predblock) in enumerate(blockindpred)
            iblock[predblock] = j
        end
        numblocks = length(blockindpred)
        for k in 1:size(x,2)
            subblockind = [j==iblock[k] ? [1] : [] for j in 1:numblocks]
            m, sig = predict_full(gp, x[:,k:k], subblockind)
            μ[k] = m[1]
            σ2[k] = max(diag(sig)[1], 0.0)
        end
        return μ, σ2
    end
end
predict_full(gp::GPE, xpred::AbstractMatrix, blockindpred::BlockIndices) = predictMVN(xpred, blockindpred, gp.x, gp.y, gp.kernel, gp.mean, gp.alpha, gp.covstrat, gp.cK)

function FSA(x::AbstractMatrix, inducing::AbstractMatrix, blockindices::BlockIndices, 
             y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Real)
    covstrat = FullScaleApproxStrat(inducing, blockindices)
    GPE(x, y, mean, kernel, logNoise, covstrat)
end
