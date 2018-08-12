abstract type CompositeMean <: Mean end

submeans(m::CompositeMean) = throw(MethodError(submeans, (m,)))

function Base.show(io::IO, cm::CompositeMean, depth::Int = 0)
    pad = repeat(" ", 2 * depth)
    println(io, "$(pad)Type: $(typeof(cm))")
    for m in submeans(cm)
        show(io, m, depth+1)
    end
end

num_params(cm::CompositeMean) = sum(num_params(m) for m in submeans(cm))

function get_params(cm::CompositeMean)
    p = Array{Float64}(undef, 0)
    for m in submeans(cm)
        append!(p, get_params(m))
    end
    p
end

function set_params!(cm::CompositeMean, hyp::Vector{Float64})
    i, n = 1, num_params(cm)
    length(hyp) == num_params(cm) || throw(ArgumentError("$(typeof(cm)) object requires $(n) hyperparameters"))
    for m in submeans(cm)
        np = num_params(m)
        set_params!(m, hyp[i:(i+np-1)])
        i += np
    end
end

##########
# Priors #
##########

function set_priors!(cm::CompositeMean, priors::Array)
    i, n = 1, num_params(cm)
    length(hyp) == n || throw(ArgumentError("$(typeof(cm)) object requires $(n) priors"))
    for m in submeans(cm)
        np = num_params(m)
        set_priors!(m, hyp[i:(i+np-1)])
        i += np
    end
end

function get_priors(cm::CompositeMean)
    p = []
    for m in submeans(cm)
        append!(p, get_priors(m))
    end
    p
end

include("sum_mean.jl")       # Sum mean functions
include("prod_mean.jl")      # Product of mean functions
