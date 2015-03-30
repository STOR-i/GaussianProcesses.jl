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

function meanf(summean::SumMean, x::Matrix{Float64})
    s = 0.0
    for m in summean.means
        s += meanf(m, x)
    end
    return s
end

function params(summean::SumMean)
    p = Array(Float64, 0)
    for m in summean.means
        append!(p, params(m))
    end
    p
end

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

function grad_meanf(summean::SumMean, x::Vector{Float64})
     dm = Array(Float64, 0)
      for m in summean.means
        append!(dm,grad_meanf(m, x))
      end
    dm
end
