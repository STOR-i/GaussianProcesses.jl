module TestVI
using GaussianProcesses, Distributions, PDMats, Plots
gr()
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

#    @testset "Increasing Elbo" begin
#        println(gp.kernel)
#        initial_elbo = elbo(Y, mean(gp.mean, gp.x), Ω, Q.m, Q.V, l)
#        println("Initial ELBO evaluation: ", initial_elbo)
#        samples = mcmc(gp; nIter=5000)
#        sample_μ = mean(samples[1:20, :], dims=2)
#        sample_var = cov(transpose(samples[1:20, :]))
#        sample_θ = mean(samples[21:end, :], dims=2)
#        
#        set_params!(gp.kernel, vec(sample_θ))
#        update_Q!(Q, sample_μ, sample_var)
#        mcmc_elbo = elbo(Y, mean(gp.mean, gp.x), Ω, Q.m, Q.V, l)
#        println("Post-MCMC ELBO: ", mcmc_elbo)
#        @test mcmc_elbo > initial_elbo
#    end

    @testset "Basic" begin
        Q = vi(gp, nits=1)
        gp.μ = Q.m
        gp.cK = PDMat(Q.V)
        xtest = range(minimum(gp.x),stop=maximum(gp.x),length=50);
        nsamps = 500
        visamples = Array{Float64}(undef, nsamps, length(xtest));
        for i in 1:nsamps
                visamples[i,:] = rand(gp, xtest)
        end
        q10 = [quantile(visamples[:,i], 0.1) for i in 1:length(xtest)]
        q50 = [quantile(visamples[:,i], 0.5) for i in 1:length(xtest)]
        q90 = [quantile(visamples[:,i], 0.9) for i in 1:length(xtest)]
        plot(xtest,exp.(q50),ribbon=(exp.(q10), exp.(q90)),leg=true, fmt=:png, label="quantiles")
        plot!(xtest, mean(ymean), label="posterior mean")
        xx = range(-3,stop=3,length=1000);
        f_xx = 2*cos.(2*xx);
        plot!(xx, exp.(f_xx), label="truth")
        scatter!(X,Y, label="data")
    end
  
 end
 end
