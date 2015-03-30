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
    p = 1.0
    for m in prodmean.means
        p = p.*meanf(m, x)
    end
    return p
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


function grad_meanf(prodmean::ProdMean, x::Vector{Float64})
     dm = Array(Float64, 0)
      for m in prodmean.means
          p = 1.0
          for j in prodkern.means[find(k.!=prodkern.means)]
              p = p.*meanf(j, x)
          end
        append!(dm,grad_meanf(m, x).*p)
      end
    dm
end
