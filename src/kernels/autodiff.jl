# abstract type AutoDiffKernel end
abstract type AutoDiffKernel <: Kernel end

@inline raw(k::AutoDiffKernel) = k.raw
@inline dual(k::AutoDiffKernel) = k.dual
priors(k::AutoDiffKernel) = k.priors

mutable struct ADkernel{Kraw<:Kernel, Kdual<:Kernel, V<:AbstractVector, D<:Dual,CFG<:GradientConfig} <: AutoDiffKernel
    raw::Kraw
    dual::Kdual
    hyp::V
    priors::Array          # Array of priors for kernel parameters
    cfg::CFG
end
function to_autodiff(k::Kernel, duals::Vector{D}) where {D<:Dual}
    kerneltype = typeof(k)
    @assert !kerneltype.abstract
    @assert !kerneltype.hasfreetypevars

    typeparams = kerneltype.parameters

    fnames = fieldnames(kerneltype)
    values = []
    newtypes = []
    for field in fnames
        ftype = fieldtype(kerneltype, field)
        if ftype<:Float64
            push!(newtypes, D)
            push!(values, duals[1]) # whatever
        elseif ftype<:Vector{Float64}
            push!(newtypes, Vector{D})
            push!(values, duals[1:1]) # whatever
        elseif ftype<:Kernel
            subkernel = getfield(k, field)
            ad_subkernel = to_autodiff(subkernel, duals)
            push!(newtypes, typeof(ad_subkernel))
            push!(values, ad_subkernel)
        else
            push!(newtypes, ftype)
            push!(values, getfield(k, field))
        end
    end
    kerneltype.name.wrapper(values...)
end
function autodiff(k::Kernel)
    cfg = GradientConfig(nothing, zeros(num_params(k)))
    D = eltype(cfg)
    CFG = typeof(cfg)
    hyp = get_params(k)
    duals = cfg.duals
    dual = to_autodiff(k, duals)
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
@inline cov(ad::AutoDiffKernel, r::Real) = cov(raw(ad), r)
@inline cov(ad::AutoDiffKernel, x1::AbstractVector, x2::AbstractVector) = cov(raw(ad), x1, x2)
@inline cov_ij(ad::AutoDiffKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int) = cov_ij(raw(ad), X1, X2, data, i, j, dim)
@inline cov_ij(ad::AutoDiffKernel, X1::AbstractMatrix, X2::AbstractMatrix, data::EmptyData, i::Int, j::Int, dim::Int) = cov_ij(raw(ad), X1, X2, i, j, dim)

KernelData(ad::AutoDiffKernel, X1::AbstractMatrix, X2::AbstractMatrix) = KernelData(raw(ad), X1, X2)
kernel_data_key(ad::AutoDiffKernel, X1::AbstractMatrix, X2::AbstractMatrix) = kernel_data_key(raw(ad), X1, X2)

@inline function dKij_dθ!(dK::AbstractVector, ad::AutoDiffKernel, X::AbstractMatrix,
                  i::Int, j::Int, dim::Int, npars::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, i, j, dim)
    copyto!(dK, partials(k_eval))
    return dK
end
@inline function dKij_dθ!(dK::AbstractVector, ad::AutoDiffKernel, X::AbstractMatrix, data::EmptyData,
                  i::Int, j::Int, dim::Int, npars::Int)
    dKij_dθ!(dK, ad, X, i, j, dim, npars)
end
@inline function dKij_dθ!(dK::AbstractVector, ad::AutoDiffKernel, X::AbstractMatrix, data::KernelData,
                  i::Int, j::Int, dim::Int, npars::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, data, i, j, dim)
    copyto!(dK, partials(k_eval))
    return dK
end
@inline function dKij_dθp(ad::AutoDiffKernel, X::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, i, j, dim)
    return partials(k_eval)[p]
end
@inline function dKij_dθp(ad::AutoDiffKernel, X::AbstractMatrix, data::EmptyData, i::Int, j::Int, p::Int, dim::Int)
    dKij_dθp(ad, X, i, j, p, dim)
end
@inline function dKij_dθp(ad::AutoDiffKernel, X::AbstractMatrix, data::KernelData, i::Int, j::Int, p::Int, dim::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, data, i, j, dim)
    return partials(k_eval)[p]
end
