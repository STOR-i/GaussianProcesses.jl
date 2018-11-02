# abstract type AutoDiffKernel end
abstract type AutoDiffKernel <: Kernel end

raw(k::AutoDiffKernel) = k.raw
dual(k::AutoDiffKernel) = k.dual
priors(k::AutoDiffKernel) = k.priors

mutable struct ADkernel{Kraw<:Kernel, Kdual<:Kernel, D<:Dual,CFG<:GradientConfig} <: AutoDiffKernel
    raw::Kraw
    dual::Kdual
    priors::Array          # Array of priors for kernel parameters
    cfg::CFG
end
# mutable struct IsotropicADkernel{K<:Isotropic{SqEuclidean},D<:Dual,CFG<:GradientConfig} <: Isotropic{SqEuclidean}, AutoDiffKernel
    # raw::K{Float64}
    # dual::K{D}
    # priors::Array          # Array of priors for kernel parameters
    # cfg::CFG
# end

function autodiff(k::Kernel)
    cfg = GradientConfig(nothing, zeros(num_params(k)))
    D = eltype(cfg)
    CFG = typeof(cfg)
    dual = typeof(k).name.wrapper(cfg.duals...) # a little hacky
    ad = ADkernel{typeof(k),typeof(dual),eltype(cfg),typeof(cfg)}(k, dual, [], cfg)
    return ad
end
# function RQIsoRaw(θ::Vector{T}) where T<:Real
    # ll, lσ, lα = θ
    # return RQIsoRaw{T}(exp(2.0*ll), exp(2.0*lσ), exp(lα))
# end

# get_params(rq::RQIsoRaw) = SVector{3}(log(rq.ℓ2)/2.0, log(rq.σ2)/2.0, log(rq.α))

# delegate (I think there are macros that would do this automatically)
num_params(ad::AutoDiffKernel) = num_params(raw(ad))
get_params(ad::AutoDiffKernel) = get_params(raw(ad))
get_param_names(ad::AutoDiffKernel) = get_param_names(raw(ad))
set_params!(ad::AutoDiffKernel, hyp) = set_params!(raw(ad), hyp)
cov(ad::AutoDiffKernel, r::Real) = cov(raw(ad), r)
cov(ad::AutoDiffKernel, x1::AbstractVector, x2::AbstractVector) = cov(raw(ad), x1, x2)
cov_ij(ad::AutoDiffKernel, X::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int) = cov_ij(raw(ad), X, data, i, j, dim)
cov_ij(ad::AutoDiffKernel, X::AbstractMatrix, data::EmptyData, i::Int, j::Int, dim::Int) = cov_ij(raw(ad), X, i, j, dim)

KernelData(ad::AutoDiffKernel, X::AbstractMatrix) = KernelData(raw(ad), X)

# function dKij_dθ!(dK::AbstractVector, ad::AutoDiffKernel, X::AbstractMatrix, data::IsotropicData, 
                                # i::Int, j::Int, dim::Int, npars::Int)
    # raw = raw(ad)
    # dual = dual(ad)
    # lθ = get_params(raw)
    # seed!(ad.cfg.duals, lθ, ad.cfg.seeds)
    # r = data.R[i,j]
    # set_params!(dual, ad.cfg.duals)
    # k = cov(dual, r)
    # copyto!(dK, partials(k))
    # return dK
# end

function dKij_dθp(ad::AutoDiffKernel, X::AbstractMatrix, data::EmptyData, i::Int, j::Int, p::Int, dim::Int)
    lθ = get_params(raw(ad))
    seed!(ad.cfg.duals, lθ, ad.cfg.seeds)
    set_params!(dual(ad), ad.cfg.duals)
    k_eval = cov_ij(dual(ad), X, i, j, dim)
    return partials(k_eval)[p]
end
function dKij_dθp(ad::AutoDiffKernel, X::AbstractMatrix, data::KernelData, i::Int, j::Int, p::Int, dim::Int)
    lθ = get_params(raw(ad))
    seed!(ad.cfg.duals, lθ, ad.cfg.seeds)
    set_params!(dual(ad), ad.cfg.duals)
    k_eval = cov_ij(dual(ad), X, data, i, j, dim)
    return partials(k_eval)[p]
end
