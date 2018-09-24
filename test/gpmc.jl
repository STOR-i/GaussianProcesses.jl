module TestGPMC
using GaussianProcesses, Calculus, Distributions
using Test, Random

Random.seed!(1)

@testset "GPMC" begin
    d, n = 3, 20
    ll = rand(d)
    X = 2π * rand(d, n)
    f = [sum(sin, view(X, :, i)) / d for i in 1:n]

    # Test Bernoulli, binomial, exponential, Gaussian, Poisson, and Student-t likelihood
    liks = (BernLik(), BinLik(n), ExpLik(), GaussLik(-1.0), PoisLik(), StuTLik(3, 0.1))
    ys = ([Bool(rand(Distributions.Bernoulli(abs(f[i])))) for i in 1:n],
          [rand(Distributions.Binomial(n,exp(f[i]) / (1 + exp(f[i])))) for i in 1:n],
          [rand(Distributions.Exponential(f[i]^2)) for i in 1:n],
          f,
          [rand(Distributions.Poisson(exp(f[i]))) for i in 1:n],
          f .+ rand(Distributions.TDist(3), n))

    # Mean and kernel function
    mZero = MeanZero()
    kern = SE(0.0, 0.0)

    @testset "Likelihood $(typeof(lik))" for (lik, y) in zip(liks, ys)
        println("\tTesting ", nameof(typeof(lik)), "...")

        # Fit GP
        gp = GP(X, y, mZero, kern, lik)

        # Sample random parameters
        params = 0.5 * randn(GaussianProcesses.num_params(gp))
        set_params!(gp, params)

        # Exact gradient
        GaussianProcesses.update_target_and_dtarget!(gp)
        exact_grad = gp.dtarget

        # Numerical approximation
        num_grad = Calculus.gradient(params) do params
            set_params!(gp, params)
            update_target!(gp)
            gp.target
        end

        @test num_grad ≈ exact_grad
    end

end
end
