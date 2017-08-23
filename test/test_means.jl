using GaussianProcesses, Base.Test
using GaussianProcesses: Mean, grad_mean, grad_stack, get_params, get_param_names
import Calculus

function check_functions_run(m::Mean, X::Matrix{Float64})
    mean(m, X[:,1])
    mean(m, X)
    grad_mean(m, X[:,1])
    grad_stack(m, X)
    show(DevNull, m)
    p = get_params(m)
    set_params!(m, p)
    get_param_names(m)
end

function test_mean_functions_consistent(m::Mean, X::Matrix{Float64})
    spec = mean(m, X)
    gen = invoke(mean, Tuple{Mean, Matrix{Float64}}, m, X)
    @test spec ≈ gen
end

function test_gradient_correct(m::Mean, x::Vector{Float64})
    init_params = get_params(m)
    nobsv = length(x)
    function mean_func(hyp)
        set_params!(m, hyp)
        return mean(m, x)
    end
    theor_grad = grad_mean(m, x)
    num_grad = Calculus.gradient(mean_func, init_params)
    @test theor_grad ≈ num_grad
end

n = 5 # number of observations
d = 4 # dimension
D = 3 # degree of polynomial mean function

means = [MeanZero(),
         MeanConst(3.0),
         MeanLin(rand(d)),
         MeanPoly(rand(d, D)), 
         MeanConst(3.0)*MeanLin(rand(d)),
         MeanLin(rand(d)) + MeanPoly(rand(d, D))]

X = rand(d, n)

for m in means
    println("\tTesting mean function $(typeof(m))...")
    check_functions_run(m, X)
    test_mean_functions_consistent(m, X)
    for i in 1:size(X,2)
        test_gradient_correct(m, X[:,i])
    end
end
