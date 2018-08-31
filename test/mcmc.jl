module TestMCMC
using GaussianProcesses, Distributions
using Test, Random

Random.seed!(1)

@testset "MCMC" begin
    d, n = 2, 20
    ll = rand(d)
    X = 2π * rand(d, n)
    y = randn(n) .+ 0.5

    kern = RQ(1.0, 1.0, 1.0)

    # Just checks that it doesn't crash
    @testset "Basic" begin
        @testset "Without likelihood" begin
            gp = GP(X, y, MeanZero(), kern)
            set_priors!(gp.kernel, [Distributions.Normal(-1.0, 1.0) for i in 1:3])
            mcmc(gp)
        end

        @testset "With likelihood" begin
            lik = GaussLik(-1.0)
            gp = GP(X, y, MeanZero(), kern, lik)
            set_priors!(gp.kernel, [Distributions.Normal(-1.0, 1.0) for i in 1:3])
            mcmc(gp)
        end
    end
end
end
