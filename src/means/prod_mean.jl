type ProdMean <: Mean
    means::Vector{Mean}
    function ProdMean(args...)
        means = Array(Mean, length(args))
        for (i,m) in enumerate(args)
            isa(m, Mean) || throw(ArgumentError("All arguments of ProdMean must be Mean objects"))
            means[i] = m
        end
        return new(means)
    end
end

function meanf(prodmean::ProdMean, x::Matrix{Float64})
    s = 0.0
    for m in prodmean.means
        s = s.*meanf(m, x)
    end
    return s
end

function params(prodmean::ProdMean)
    p = Array(Float64, 0)
    for m in prodmean.means
        append!(p, params(m))
    end
    p
end

function num_params(prodmean::ProdMean)
    n = 0
    for m in prodmean.means
        n += num_params(m)
    end
    n
end

function set_params!(prodmean::ProdMean, hyp::Vector{Float64})
    i, n = 1, num_params(prodmean)
    length(hyp) == num_params(prodmean) || throw(ArgumentError("ProdMean object requires $(n) hyperparameters"))
    for m in prodmean.means
        np = num_params(m)
        set_params!(m, hyp[i:(i+np-1)])
        i += np
    end
end

#Needs fixing
function grad_meanf(prodmean::ProdMean, x::Matrix{Float64})
     dm = Array(Float64, 0)
      for m in prodmean.means
        append!(dm,grad_meanf(m, x).*meanf(m,x)) #Need dmⱼ*m_{-j}
      end
    dm
end
