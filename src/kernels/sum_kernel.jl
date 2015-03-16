type SumKernel <: Kernel
    kerns::Vector{Kernel}
    function SumKernel(args...)
        kerns = Array(Kernel, length(args))
        for (i,k) in enumerate(args)
            isa(k, Kernel) || throw(ArgumentError("All arguments of SumKernel must be Kernel objects"))
            kerns[i] = k
        end
        return new(kerns)
    end
end

function kern(sumkern::SumKernel, x::Vector{Float64}, y::Vector{Float64})
    s = 0.0
    for k in sumkern.kerns
        s += kern(k, x, y)
    end
    return s
end

function params(sumkern::SumKernel)
    p = Array(Float64, 0)
    for k in sumkern.kerns
        append!(p, params(k))
    end
    p
end

function num_params(sumkern::SumKernel)
    n = 0
    for k in sumkern.kerns
        n += num_params(k)
    end
    n
end

function set_params!(sumkern::SumKernel, hyp::Vector{Float64})
    i, n = 1, num_params(sumkern)
    length(hyp) == num_params(sumkern) || throw(ArgumentError("SumKernel object requires $(n) hyperparameters"))
    for k in sumkern.kerns
        np = num_params(k)
        set_params!(k, hyp[i:(i+np-1)])
        i += np
    end
end

function grad_kern(sumkern::SumKernel, x::Vector{Float64}, y::Vector{Float64})
     dk = Array(Float64, 0)
      for k in sumkern.kerns
        append!(dk,grad_kern(k, x, y))
      end
    dk
end
