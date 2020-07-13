abstract type PairKernel{K1<:Kernel,K2<:Kernel} <: CompositeKernel end

leftkern(k::PairKernel) = throw(MethodError(leftkern, (k,)))
rightkern(k::PairKernel) = throw(MethodError(rightkern, (k,)))
components(k::PairKernel) = [leftkern(k), rightkern(k)]

function Base.show(io::IO, pairkern::PairKernel, depth::Int = 0)
    pad = repeat(" ", 2 * depth)
    println(io, "$(pad)Type: $(typeof(pairkern))")
    show(io, leftkern(pairkern), depth+1)
    show(io, rightkern(pairkern), depth+1)
end

num_params(pairkern::PairKernel) = num_params(leftkern(pairkern))+num_params(rightkern(pairkern))
get_params(pairkern::PairKernel) = vcat(get_params(leftkern(pairkern)), get_params(rightkern(pairkern)))
get_param_names(pairkern::PairKernel) = composite_param_names([leftkern(pairkern), rightkern(pairkern)], :sk)

function set_params!(pairkern::PairKernel, hyp::AbstractVector)
    npl = num_params(leftkern(pairkern))
    hyp_left = hyp[1:npl]
    hyp_right = hyp[npl+1:end]
    set_params!(leftkern(pairkern), hyp_left)
    set_params!(rightkern(pairkern), hyp_right)
end

##########
# Priors #
##########

function set_priors!(pairkern::PairKernel, priors::Array)
    npl = num_params(leftkern(pairkern))
    priors_left = priors[1:npl]
    priors_right = priors[npl+1:end]
    set_priors!(leftkern(pairkern), priors_left)
    set_priors!(rightkern(pairkern), priors_right)
end

get_priors(pairkern::PairKernel) = vcat(get_priors(leftkern(pairkern)), get_priors(rightkern(pairkern)))

#################
# PairData      #
#################

struct PairData{KD1 <: KernelData, KD2 <: KernelData} <: KernelData
    data1::KD1
    data2::KD2
end
const KernelDict = Dict{String,KernelData}
KernelData(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, cache::KernelDict) = KernelData(k, X1, X2)
function KernelData(masked::Masked, X1::AbstractMatrix, X2::AbstractMatrix, cache::KernelDict)
    X1view = view(X1,masked.active_dims,:)
    X2view = view(X2,masked.active_dims,:)
    wrappeddata = KernelData(masked.kernel, X1view, X2view, cache)
    return MaskedData(X1view, X2view, wrappeddata)
end
KernelData(k::FixedKernel, X1::AbstractMatrix, X2::AbstractMatrix, cache::KernelDict) = KernelData(k.kernel, X1, X2, cache)
function KernelData(pairkern::PairKernel, X1::AbstractMatrix, X2::AbstractMatrix, cache::KernelDict=KernelDict())
    leftk  = leftkern(pairkern)
    rightk = rightkern(pairkern)
    # this is a bit broken:
    leftkey = kernel_data_key(leftk, X1, X2)
    rightkey = kernel_data_key(rightk, X1, X2)
    if leftkey ∉ keys(cache)
        leftdata = KernelData(leftk, X1, X2, cache)
        cache[leftkey] = leftdata
    else
        leftdata = cache[leftkey]
    end
    if rightkey ∉ keys(cache)
        rightdata = KernelData(rightk, X1, X2, cache)
        cache[rightkey] = rightdata
    else
        rightdata = cache[rightkey]
    end
    return PairData(leftdata, rightdata)
end
function kernel_data_key(pairkern::PairKernel, X1::AbstractMatrix, X2::AbstractMatrix)
    kl = leftkern(pairkern)
    kr = rightkern(pairkern)
    @sprintf("PairData:%s+%s", kernel_data_key(kl, X1, X2), kernel_data_key(kr, X1, X2))
end
