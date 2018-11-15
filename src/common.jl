const MeanOrKernelOrLikelihood = Union{Mean,Kernel,Likelihood}
const CompositeMeanOrKernel = Union{CompositeMean,CompositeKernel}

##########¤
# Display #
###########

function Base.show(io::IO, obj::MeanOrKernelOrLikelihood, depth::Int = 0)
    for _ in 1:depth
        print(io, "  ")
    end
    print(io, "Type: ", typeof(obj), ", Params: ")
    show(io, get_params(obj))
end

function Base.show(io::IO, obj::CompositeMeanOrKernel, depth::Int = 0)
    for _ in 1:depth
        print(io, "  ")
    end
    println(io, "Type: ", typeof(obj))
    for c in components(obj)
        show(io, c, depth + 1)
    end
end

##############################
# Parameter name definitions #
##############################

# This generates names like [:ll_1, :ll_2, ...] for parameter vectors
get_param_names(n::Int, prefix::Symbol) = [Symbol(prefix, :_, i) for i in 1:n]
get_param_names(v::Vector, prefix::Symbol) = get_param_names(length(v), prefix)

"""
    composite_param_names(objects, prefix)

Call `get_param_names` on each element of `objects` and prefix the returned name of the
element at index `i` with `prefix * i * '_'`.

# Examples
```
julia> GaussianProcesses.get_param_names(ProdKernel(Mat12Iso(1/2, 1/2), SEArd([0.0, 1.0], 0.0)))
5-element Array{Symbol,1}:
 :pk1_ll
 :pk1_lσ
 :pk2_ll_1
 :pk2_ll_2
 :pk2_lσ
```
"""
function composite_param_names(objects, prefix)
    p = Symbol[]
    for (i, obj) in enumerate(objects)
        append!(p, [Symbol(prefix, i, :_, sym) for sym in get_param_names(obj)])
    end
    p
end

##############
# Parameters #
##############

function get_params(obj::CompositeMeanOrKernel)
    p = Array{Float64}(undef, 0)
    for c in components(obj)
        append!(p, get_params(c))
    end
    p
end

num_params(obj::CompositeMeanOrKernel) = sum(num_params, components(obj))

function set_params!(obj::CompositeMeanOrKernel, hyp::AbstractVector)
    length(hyp) == num_params(obj) ||
        throw(ArgumentError("$(typeof(obj)) object requires $(num_params(obj)) hyperparameters"))
    i = 1
    @inbounds for c in components(obj)
        j = i + num_params(c)
        set_params!(c, view(hyp, i:(j - 1)))
        i = j
    end
end

##########
# Priors #
##########

get_priors(obj::MeanOrKernelOrLikelihood) = obj.priors

function get_priors(obj::CompositeMeanOrKernel)
    p = []
    for c in components(obj)
        append!(p, get_priors(c))
    end
    p
end

function set_priors!(obj::MeanOrKernelOrLikelihood, priors::Array)
    length(priors) == num_params(obj) ||
        throw(ArgumentError("$(typeof(obj)) object requires $(num_params(obj)) priors"))

    obj.priors = priors
end

function set_priors!(obj::CompositeMeanOrKernel, priors::Array)
    length(priors) == num_params(obj) ||
        throw(ArgumentError("$(typeof(obj)) object requires $(num_params(obj)) priors"))
    i = 1
    @inbounds for c in components(obj)
        j = i + num_params(c)
        set_priors!(c, view(priors, i:(j - 1)))
        i = j
    end
end

function prior_logpdf(obj::MeanOrKernelOrLikelihood)
    num_params(obj) == 0 && return 0.0

    priors = get_priors(obj)
    isempty(priors) && return 0.0

    sum(logpdf(prior, param) for (prior, param) in zip(priors, get_params(obj)))
end

function prior_gradlogpdf(obj::MeanOrKernelOrLikelihood)
    num_params(obj) == 0 && return Float64[]

    priors = get_priors(obj)
    isempty(priors) && return zeros(num_params(obj))

    [gradlogpdf(prior, param) for (prior, param) in zip(priors, get_params(obj))]
end
