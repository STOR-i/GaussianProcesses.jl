type SumMean <: Mean
    means::Vector{Mean}
    function SumMean(args...)
        means = Array(Mean, length(args))
        for (i,m) in enumerate(args)
            isa(m, Mean) || throw(ArgumentError("All arguments of SumMean must be Mean objects"))
            means[i] = m
        end
        return new(means)
    end
end

function show(io::IO, sm::SumMean, depth::Int = 0)
    pad = repeat(" ", 2 * depth)
    println(io, "$(pad)Type: $(typeof(sm))")
    for m in sm.means
        show(io, m, depth+1)
    end
end

function mean(summean::SumMean, x::Matrix{Float64})
    s = 0.0
    for m in summean.means
        s += mean(m, x)
    end
    return s
end

function get_params(summean::SumMean)
    p = Array(Float64, 0)
    for m in summean.means
        append!(p, get_params(m))
    end
    p
end

get_param_names(summean::SumMean) = composite_param_names(summean.means, :sm)

function num_params(summean::SumMean)
    n = 0
    for m in summean.means
        n += num_params(m)
    end
    n
end

function set_params!(summean::SumMean, hyp::Vector{Float64})
    i, n = 1, num_params(summean)
    length(hyp) == num_params(summean) || throw(ArgumentError("SumMean object requires $(n) hyperparameters"))
    for m in summean.means
        np = num_params(m)
        set_params!(m, hyp[i:(i+np-1)])
        i += np
    end
end

function grad_mean(summean::SumMean, x::Vector{Float64})
     dm = Array(Float64, 0)
      for m in summean.means
        append!(dm,grad_meanf(m, x))
      end
    dm
end

function +(m1::SumMean, m2::Mean)
    means = [m1.means, m2]
    SumMean(means...)
end
function +(m1::SumMean, m2::SumMean)
    means = [m1.means, m2.means]
    SumMean(means...)
end
+(m1::Mean, m2::Mean) = SumMean(m1,m2)
+(m1::Mean, m2::SumMean) = +(m2,m1)
