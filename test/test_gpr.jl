module TestGPR
using GaussianProcesses, Distributions
using Test, Random
using Optim
Random.seed!(123)

@testset "GPR" begin
    n = 20
    x = cat(collect(-1:0.05:1), dims=2);
    y = 2*cos.(2*x);

    kernel = SquaredExponential(0.5, 1.0)
    f = GPR(kernel)

    mll_val = marginal_log_likelihood(f, x, y)
    print("Initial marginal log-likelihood: $mll_val")

    # Train    
    function loss(x, y)
        return -marginal_log_likelihood(f, x, y)
    end

    objective = get_objective(f, x, y)
    grad = gradient(objective, [1, 1])

    function optimise()
        param_set = [1., 1.]
        α = 0.1
        for i in 1:10
            ∇ = gradient(objective, param_set)[1]
            param_set = param_set - α*∇
            obj = -marginal_log_likelihood(f, x, y)
            println("Objective: $obj")
        end
        println(param_set)
    end

    # ℓ = f.Kernel.lengthscale.value
    # σ = f.Kernel.variance.value
    
    # θ = Flux.params([ℓ, σ])
    # grads = gradient(() -> loss(x, y), θ)

    # using Flux.Optimise: update!
    # η = 0.1 # Learning Rate
    # for p in (ℓ, σ)
    #     update!(p, -η * grads[p])
    # end
end
end


#  mean_func = Zero()
#  kernel = SquaredExponential(1.0, 1.0)
#  f = Construct_GPR(mean_func, kernel)
#  x = cat(collect(-1:0.01:1), dims=2)'

#  samples = rand(f, x; n_samples=10)
#  p = plot(vec(x), samples, color=:blue, alpha=0.5)



#  μ = mean(mean_func, x)
#  Kxx = cov(kernel, x)
#  Kxx2 = cov(kernel,x, x)
#  @assert Kxx==Kxx2



