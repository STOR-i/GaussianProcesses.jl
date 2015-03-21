type ProdKernel <: Kernel
    kerns::Vector{Kernel}
    function ProdKernel(args...)
        kerns = Array(Kernel, length(args))
        for (i,k) in enumerate(args)
            isa(k, Kernel) || throw(ArgumentError("All arguments of ProdKernel must be Kernel objects"))
            kerns[i] = k
        end
        return new(kerns)
    end
end

function kern(prodkern::ProdKernel, x::Vector{Float64}, y::Vector{Float64})
    s = 0.0
    for k in prodkern.kerns
        s = s.*kern(k, x, y)
    end
    return s
end

function params(prodkern::ProdKernel)
    p = Array(Float64, 0)
    for k in prodkern.kerns
        append!(p, params(k))
    end
    p
end

function num_params(prodkern::ProdKernel)
    n = 0
    for k in prodkern.kerns
        n += num_params(k)
    end
    n
end

function set_params!(prodkern::ProdKernel, hyp::Vector{Float64})
    i, n = 1, num_params(prodkern)
    length(hyp) == num_params(prodkern) || throw(ArgumentError("ProdKernel object requires $(n) hyperparameters"))
    for k in prodkern.kerns
        np = num_params(k)
        set_params!(k, hyp[i:(i+np-1)])
        i += np
    end
end

#Needs fixing
function grad_kern(prodkern::ProdKernel, x::Vector{Float64}, y::Vector{Float64})
     dk = Array(Float64, 0)
      for k in prodkern.kerns
        append!(dk,grad_kern(k, x, y).*kern(k,x,y)) #Need dKⱼ,*K_{-j}
      end
    dk
end
