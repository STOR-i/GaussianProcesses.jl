module TestExp
using GaussianProcesses, Distributions
using Test, Random
Random.seed!(123)

@testset "VI" begin
    n = 20
    X = collect(range(-3,stop=3,length=n));
    f = 2*cos.(2*X);
    Y = [rand(Poisson(exp.(f[i]))) for i in 1:n];

    k = Matern(3/2,0.0,0.0)   # Matern 3/2 kernel
    l = PoisLik()             # Poisson likelihood
    gp = GP(X, vec(Y), MeanZero(), k, l)
    set_priors!(gp.kernel,[Normal(-2.0,4.0),Normal(-2.0,4.0)])

    @testset "Basic" begin
        V = cov(gp.kernel, gp.x, gp.data)
        m = mean(gp.mean, gp.x)
        print(variational_expectation(l, Y, m, V))
    end
end
end
