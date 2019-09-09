module TestVI
using GaussianProcesses, Distributions
using Test, Random
Random.seed!(1234)

@testset "VI" begin
    Random.seed!(203617)
    n = 20
    X = collect(range(-3,stop=3,length=n));
    f = 2*cos.(2*X);
    Y = [rand(Poisson(exp.(f[i]))) for i in 1:n];

    k = Matern(3/2,0.0,0.0)   # Matern 3/2 kernel
    l = PoisLik()             # Poisson likelihood
    gp = GP(X, vec(Y), MeanZero(), k, l)
    set_priors!(gp.kernel,[Normal(-2.0,4.0),Normal(-2.0,4.0)])
    Q, Ω, K = initialise_Q(gp)

    # @testset "AutoDiff" begin 
    #     exact = -0.5*exp(Q.m[end] + Q.V[end, end]/2) 
    #     zyg_calc = dv_var_exp(l, Y[end], Q.m[end], Q.V[end, end])   
    #     @test exact ≈ zyg_calc # Test AutoDiff is giving the same results as exact gradient computation
    # end

    # @testset "Increasing Elbo" begin
    #     println(gp.kernel)
    #     initial_elbo = elbo(Y, mean(gp.mean, gp.x), Ω, Q.m, Q.V, l)
    #     println("Initial ELBO evaluation: ", initial_elbo)
    #     samples = mcmc(gp; nIter=30000)
    #     update_Q!(Q, gp.μ, gp.cK.mat)
    #     println(gp.kernel)
    #     mcmc_elbo = elbo(Y, mean(gp.mean, gp.x), Ω, Q.m, Q.V, l)
    #     println("Post-MCMC ELBO: ", mcmc_elbo)
    #     @test mcmc_elbo > initial_elbo
    # end

    @testset "Basic" begin
        vi(gp; nits = 100)
    end

end
end
