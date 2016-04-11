# Implement the ScikitLearnBase.jl interface

import ScikitLearnBase

ScikitLearnBase.is_classifier(::GP) = false

function ScikitLearnBase.fit!(gp::GP, X::Matrix{Float64}, y::Vector{Float64})
    gp.x = X' # ScikitLearn's X is (n_samples, n_features)
    gp.y = y
    gp.dim, gp.nobsv = size(gp.x)
    length(y) == gp.nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
    update_mll!(gp)
    return gp
end

function ScikitLearnBase.predict(gp::GP, X::Matrix{Float64}; eval_MSE::Bool=false)
    mu, Sigma = predict(gp, X'; full_cov=false)
    if eval_MSE
        return mu, Sigma
    else
        return mu
    end
end

# This is a default - arbitrary scoring functions can be passed to GridSearchCV
function ScikitLearnBase.score(gp::GP, x, y)
    # I use MSE here for simplicity. Is that equivalent to the likelihood for
    # gaussian processes?
    # It would make sense to use the marginal likelihood for hyperparameter
    # tuning (like optimize.jl does), but GridSearchCV isn't designed for that.
    return -mean((ScikitLearnBase.predict(gp, x) - y) .^ 2)
end

ScikitLearnBase.clone(gp::GP) = GP(; m=ScikitLearnBase.clone(gp.m),
                                   k=ScikitLearnBase.clone(gp.k),
                                   logNoise=gp.logNoise)

# Means and Kernels are small objects, and they are not fit to the data (except
# through hyperparameter tuning), so `deepcopy` is appropriate (if a bit lazy)
ScikitLearnBase.clone(m::Mean) = deepcopy(m)
ScikitLearnBase.clone(k::Kernel) = deepcopy(k)

################################################################################
## GaussianProcesses.jl has a get_params function, which returns a
## Vector{Float64}. We use it to implement ScikitLearnBase.get_params, which
## must return a Symbol => value dictionary. Likewise for set_params!

function ScikitLearnBase.get_params(gp::GP)
    add_prefix(pref, di) =
        Dict([symbol(pref, param)=>value for (param, value) in di])
    merge(add_prefix(:m_, ScikitLearnBase.get_params(gp.m)),
          add_prefix(:k_, ScikitLearnBase.get_params(gp.k)),
          Dict(:logNoise=>gp.logNoise))
end

function ScikitLearnBase.get_params(obj::Union{Mean, Kernel})
    params = get_params(obj) # GaussianProcesses' get_params
    names = get_param_names(obj)
    @assert length(params) == length(names) # sanity check
    return Dict(zip(names, params))
end


function ScikitLearnBase.set_params!(gp::GP; params...)
    m_params = Dict()
    k_params = Dict()
    for (name, value) in params
        sname = string(name)
        if startswith(sname, "m_")
            m_params[symbol(sname[3:end])] = value
        elseif startswith(sname, "k_")
            k_params[symbol(sname[3:end])] = value
        else
            @assert name == :logNoise "Unknown parameter passed to set_params!: $name"
            gp.logNoise = value
        end
    end
    ScikitLearnBase.set_params!(gp.m; m_params...)
    ScikitLearnBase.set_params!(gp.k; k_params...)
    gp
end

function ScikitLearnBase.set_params!(obj::Union{Mean, Kernel}; params...)
    # In ScikitLearn, params need not contain all the parameters, so the
    # code is different from get_params
    names = get_param_names(obj)
    # Get current parameter vector
    hyp = copy(get_params(obj))
    # Update it
    for (name, value) in params
        ind = findfirst(names, name)
        @assert ind > 0 "Unknown parameter passed to set_params!: $name"
        hyp[ind] = value
    end
    set_params!(obj, hyp)
    obj
end

################################################################################
# get_param_names definitions
# Maybe they should be moved next to each Kernel and Mean definition?

# This generates names like [:ll_1, :ll_2, ...] for parameter vectors
get_param_names(n::Int, prefix::Symbol) = [symbol(prefix, :_, i) for i in 1:n]
get_param_names(v::Vector, prefix::Symbol) = get_param_names(length(v), prefix)

# Kernels
get_param_names(::SEIso) = [:ll, :lσ]
get_param_names(k::SEArd) = vcat(get_param_names(k.ℓ2, :ll), :lσ)

# Means
get_param_names(::MeanConst) = [:β]

# TODO: fill in for all Kernels and Means
