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

function show(io::IO, sumkern::SumKernel, depth::Int = 0)
    pad = repeat(" ", 2 * depth)
    println(io, "$(pad)Type: $(typeof(sumkern))")
    for k in sumkern.kerns
        show(io, k, depth+1)
    end
end

function kern(sumkern::SumKernel, x::Vector{Float64}, y::Vector{Float64})
    s = 0.0
    for k in sumkern.kerns
        s += kern(k, x, y)
    end
    return s
end

# This slows down crossKern...

## function crossKern(X::Matrix{Float64}, sumkern::SumKernel)
##     d, nobsv = size(X)
##     s = zeros(nobsv, nobsv)
##     for k in sumkern.kerns
##         BLAS.axpy!(nobsv*nobsv, 1.0, crossKern(X,k), 1, s, 1)
##         #s += crossKern(X, k)
##     end
##     return s
## end

## function add_matrices!(X::AbstractMatrix, Y::AbstractMatrix)
##     m,n = size(X)
##     for i in 1:m, j in 1:n
##         X[i,j] += Y[i,j]
##     end
## end

function crossKern(X::Matrix{Float64}, sumkern::SumKernel)
    d, nobsv = size(X)
    s = zeros(nobsv, nobsv)
    for k in sumkern.kerns
        #add_matrices!(s, crossKern(X,k))
        s[:,:] = s + crossKern(X,k)
    end
    return s
end
    
function get_params(sumkern::SumKernel)
    p = Array(Float64, 0)
    for k in sumkern.kerns
        append!(p, get_params(k))
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
   p end
end

function grad_kern(sumkern::SumKernel, x::Vector{Float64}, y::Vector{Float64})
     dk = Array(Float64, 0)
      for k in sumkern.kerns
        append!(dk,grad_kern(k, x, y))
      end
    dk
end

function grad_stack!(stack::AbstractArray, X::Matrix{Float64}, sumkern::SumKernel)
    s = 1
    for kern in sumkern.kerns
        np = num_params(kern)
        grad_stack!(view(stack,:, :, s:(s+np-1)), X, kern)
        s += np
    end
    return stack
end
        
# Addition operators
function +(k1::SumKernel, k2::Kernel)
    kerns = [k1.kerns, k2]
    SumKernel(kerns...)
end
function +(k1::SumKernel, k2::SumKernel)
    kerns = [k1.kerns, k2.kerns]
    SumKernel(kerns...)
end
+(k1::Kernel, k2::Kernel) = SumKernel(k1,k2)
+(k1::Kernel, k2::SumKernel) = +(k2,k1)
