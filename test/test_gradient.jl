using GaussianProcesses
using GaussianProcesses: set_params!, ll!, dL, KernelData
import Calculus

X = collect(linspace(-3,3,20))';
#Y = [rand(Distributions.Exponential(sin(X[i]).^2)) for i in 1:20];

k = SEIso(2.0, 1.0)
m = MeanZero()

k = SE(0.0, 0.0)
like = Exponential()

#gp = GPMC{Float64}(X', vec(Y), MeanZero(), k, like)


function dL_exact(hyp::Vector{Float64}, iparam::Int)
    set_params!(k, hyp)
    nobsv = size(X, 2)
    data = KernelData(k, X)
    Σ = cov(k, X, data)
    Kgrad = Array(Float64, nobsv, nobsv)
    GaussianProcesses.grad_slice!(Kgrad, k, X, data, iparam)
    return dL(Σ, Kgrad)
end



function dL_numeric(hyp::Vector{Float64}, iparam::Int)
    set_params!(k, hyp)
    nobsv = size(X, 2)
    function calc_L(θ, i, j)
        temp_hyp = copy(hyp)
        temp_hyp[iparam] = θ
        set_params!(k, temp_hyp)
        Σ = cov(k, X)
        L = chol(Σ + 1e-8*eye(nobsv))'
        return L[i,j]
    end
        
    deriv_L = zeros(Float64, nobsv, nobsv)
    for i in 1:nobsv, j in 1:i
        deriv_L[i,j] = Calculus.gradient(θ->calc_L(θ,i,j), hyp[iparam])
    end
    return deriv_L
end    

hyp = [1.0, 1.0]
iparam = 1

ex = dL_exact(hyp, iparam)
num = dL_numeric(hyp, iparam)

norm(ex-num)

