mutable struct ADkernel{Kraw<:Kernel, Kdual<:Kernel, V<:AbstractVector, D<:Dual,CFG<:GradientConfig} <: Kernel
    raw::Kraw
    dual::Kdual
    hyp::V
    cfg::CFG
end
@inline raw(k::ADkernel) = k.raw
@inline dual(k::ADkernel) = k.dual

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
            val = getfield(k, field)
            push!(values, duals[1:length(val)]) # whatever
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
    # TODO: smaller chunk sizes
    cfg = GradientConfig(nothing, zeros(num_params(k)), Chunk{num_params(k)}())
    D = eltype(cfg)
    CFG = typeof(cfg)
    hyp = get_params(k)
    duals = cfg.duals
    kdual = to_autodiff(k, duals)
    set_params!(kdual, duals)
    ad = ADkernel{typeof(k),typeof(kdual),typeof(hyp),eltype(cfg),typeof(cfg)}(k, kdual, hyp, cfg)
    return ad
end
# function RQIsoRaw(θ::Vector{T}) where T<:Real
    # ll, lσ, lα = θ
    # return RQIsoRaw{T}(exp(2.0*ll), exp(2.0*lσ), exp(lα))
# end

# get_params(rq::RQIsoRaw) = SVector{3}(log(rq.ℓ2)/2.0, log(rq.σ2)/2.0, log(rq.α))

# delegate (I think there are macros that would do this automatically)
num_params(ad::ADkernel) = num_params(raw(ad))
get_params(ad::ADkernel) = get_params(raw(ad))
get_param_names(ad::ADkernel) = get_param_names(raw(ad))
function set_params!(ad::ADkernel, hyp)
    set_params!(raw(ad), hyp)
    ad.hyp = get_params(raw(ad))
end
function seed!(ad::ADkernel)
    seed!(ad.cfg.duals, ad.hyp, ad.cfg.seeds)
    set_params!(dual(ad), ad.cfg.duals)
end
@inline cov(ad::ADkernel, r::Real) = cov(raw(ad), r)
@inline cov(ad::ADkernel, x1::AbstractVector, x2::AbstractVector) = cov(raw(ad), x1, x2)
@inline cov_ij(ad::ADkernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int) = cov_ij(raw(ad), X1, X2, data, i, j, dim)
@inline cov_ij(ad::ADkernel, X1::AbstractMatrix, X2::AbstractMatrix, data::EmptyData, i::Int, j::Int, dim::Int) = cov_ij(raw(ad), X1, X2, i, j, dim)

KernelData(ad::ADkernel, X1::AbstractMatrix, X2::AbstractMatrix) = KernelData(raw(ad), X1, X2)
kernel_data_key(ad::ADkernel, X1::AbstractMatrix, X2::AbstractMatrix) = kernel_data_key(raw(ad), X1, X2)

@inline function dKij_dθ!(dK::AbstractVector, ad::ADkernel, X::AbstractMatrix,
                  i::Int, j::Int, dim::Int, npars::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, i, j, dim)
    copyto!(dK, partials(k_eval))
    return dK
end
@inline function dKij_dθ!(dK::AbstractVector, ad::ADkernel, X::AbstractMatrix, data::EmptyData,
                  i::Int, j::Int, dim::Int, npars::Int)
    dKij_dθ!(dK, ad, X, i, j, dim, npars)
end
@inline function dKij_dθ!(dK::AbstractVector, ad::ADkernel, X::AbstractMatrix, data::KernelData,
                  i::Int, j::Int, dim::Int, npars::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, data, i, j, dim)
    copyto!(dK, partials(k_eval))
    return dK
end
@inline function dKij_dθp(ad::ADkernel, X::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, i, j, dim)
    return partials(k_eval)[p]
end
@inline function dKij_dθp(ad::ADkernel, X::AbstractMatrix, data::EmptyData, i::Int, j::Int, p::Int, dim::Int)
    dKij_dθp(ad, X, i, j, p, dim)
end
@inline function dKij_dθp(ad::ADkernel, X::AbstractMatrix, data::KernelData, i::Int, j::Int, p::Int, dim::Int)
    seed!(ad)
    k_eval = cov_ij(dual(ad), X, X, data, i, j, dim)
    return partials(k_eval)[p]
end

# priors
get_priors(ad::ADkernel) = get_priors(raw(ad))
set_priors!(ad::ADkernel, priors) = set_priors!(raw(ad), priors)
