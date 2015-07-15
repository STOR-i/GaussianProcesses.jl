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

function show(io::IO, prodkern::ProdKernel, depth::Int = 0)
    pad = repeat(" ", 2 * depth)
    println(io, "$(pad)Type: $(typeof(prodkern))")
    for k in prodkern.kerns
        show(io, k, depth+1)
    end
end

function kern(prodkern::ProdKernel, x::Vector{Float64}, y::Vector{Float64})
    p = 1.0
    for k in prodkern.kerns
        p = p.*kern(k, x, y)
    end
    return p
end

function crossKern(X::Matrix{Float64}, prodkern::ProdKernel)
    d, nobsv = size(X)
    p = ones(nobsv, nobsv)
    for k in prodkern.kerns
        p[:,:] = p .* crossKern(X,k)
    end
    return p
end

function get_params(prodkern::ProdKernel)
    p = Array(Float64, 0)
    for k in prodkern.kerns
        append!(p, get_params(k))
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

# This function is extremely inefficient
function grad_kern(prodkern::ProdKernel, x::Vector{Float64}, y::Vector{Float64})
     dk = Array(Float64, 0)
      for k in prodkern.kerns
          p = 1.0
          for j in prodkern.kerns[find(k.!=prodkern.kerns)]
              p = p.*kern(j, x, y)
          end
        append!(dk,grad_kern(k, x, y).*p) 
      end
    dk
end

function grad_stack!(stack::AbstractArray, X::Matrix{Float64}, prodkern::ProdKernel)
    d, nobsv = size(X)
    num_kerns = length(prodkern.kerns)
    
    cross_kerns = Array(Float64, nobsv, nobsv, num_kerns)
    for (i,kern) in enumerate(prodkern.kerns)
        cross_kerns[:,:,i] = crossKern(X, kern)
    end

    s = 1
    for (i,kern) in enumerate(prodkern.kerns)
        np = num_params(kern)
        grad_stack!(view(stack,:,:,s:(s+np-1)), X, kern)
        for j in 1:num_kerns
            if j != i
                broadcast!(.*, view(stack,:,:,s:(s+np-1)), view(stack,:,:,s:(s+np-1)), view(cross_kerns, :,:,j))
            end
        end
        s += np
    end
    stack
end

# Multiplication operators
function *(k1::ProdKernel, k2::Kernel)
    kerns = [k1.kerns, k2]
    ProdKernel(kerns...)
end
function *(k1::ProdKernel, k2::ProdKernel)
    kerns = [k1.kerns, k2.kerns]
    ProdKernel(kerns...)
end
*(k1::Kernel, k2::Kernel) = ProdKernel(k1,k2)
*(k1::Kernel, k2::ProdKernel) = *(k2,k1)
