# abstract type AutoDiffKernel end
abstract type AutoDiffKernel <: Kernel end

raw(k::AutoDiffKernel) = k.raw
dual(k::AutoDiffKernel) = k.dual
priors(k::AutoDiffKernel) = k.priors

mutable struct ADkernel{Kraw<:Kernel, Kdual<:Kernel, V<:AbstractVector, D<:Dual,CFG<:GradientConfig} <: AutoDiffKernel
    raw::Kraw
    dual::Kdual
    hyp::V
    priors::Array          # Array of priors for kernel parameters
    cfg::CFG
end

function autodiff(k::Kernel)
    cfg = GradientConfig(nothing, zeros(num_params(k)))
    D = eltype(cfg)
    CFG = typeof(cfg)
    dual = typeof(k).name.wrapper(cfg.duals...) # a little hacky
    hyp = get_params(k)
    ad = ADkernel{typeof(k),typeof(dual),typeof(hyp),eltype(cfg),typeof(cfg)}(k, dual, hyp, [], cfg)
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
function set_params!(ad::AutoDiffKernel, hyp)
    set_params!(raw(ad), hyp)
    copyto!(ad.hyp, hyp)
end
function seed!(ad::AutoDiffKernel)
    seed!(ad.cfg.duals, ad.hyp, ad.cfg.seeds)
    set_params!(dual(ad), ad.cfg.duals)
end
cov(ad::AutoDiffKernel, r::Real) = cov(raw(ad), r)
cov(ad::AutoDiffKernel, x1::AbstractVector, x2::AbstractVector) = cov(raw(ad), x1, x2)
cov_ij(ad::AutoDiffKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int) = cov_ij(raw(ad), X1, X2, data, i, j, dim)
cov_ij(ad::AutoDiffKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::EmptyData, i::Int, j::Int, dim::Int) = cov_ij(raw(ad), X1, X2, i, j, dim)

KernelData(ad::AutoDiffKernel, X1::AbstractMatrix, X2::AbstractMatrix) = KernelData(raw(ad), X1, X2)
kernel_data_key(ad::AutoDiffKernel, X1::AbstractMatrix, X2::AbstractMatrix) = kernel_data_key(raw(ad), X1, X2)

function dKij_dθ!(dK::AbstractVector, ad::AutoDiffKernel, X::AbstractMatrix, 
                  i::Int, j::Int, dim::Int, npars::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, i, j, dim)
    copyto!(dK, partials(k_eval))
    return dK
end
function dKij_dθ!(dK::AbstractVector, ad::AutoDiffKernel, X::AbstractMatrix, data::IsotropicData, 
                  i::Int, j::Int, dim::Int, npars::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, data, i, j, dim)
    copyto!(dK, partials(k_eval))
    return dK
end
function dKij_dθp(ad::AutoDiffKernel, X::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, i, j, dim)
    return partials(k_eval)[p]
end
function dKij_dθp(ad::AutoDiffKernel, X::AbstractMatrix, data::EmptyData, i::Int, j::Int, p::Int, dim::Int)
    dKij_dθp(ad, X, i, j, p, dim)
end
function dKij_dθp(ad::AutoDiffKernel, X::AbstractMatrix, data::KernelData, i::Int, j::Int, p::Int, dim::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, data, i, j, dim)
    return partials(k_eval)[p]
end
