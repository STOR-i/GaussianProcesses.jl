module TestVI
using GaussianProcesses, Distributions, PDMats, Plots, LinearAlgebra
gr()
using Test, Random
Random.seed!(203617)

@testset "VI" begin
    n = 20
    X = collect(range(-3,stop=3,length=n));
    f = 2*cos.(2*X);
    Y = [rand(Poisson(exp.(f[i]))) for i in 1:n];

    k = Matern(3/2,0.0,0.0)   # Matern 3/2 kernel
    l = PoisLik()             # Poisson likelihood
    gp = GP(X, vec(Y), MeanZero(), k, l)
    set_priors!(gp.kernel,[Normal(-2.0,4.0),Normal(-2.0,4.0)])
    Q, Ω, K = initialise_Q(gp)

    @testset "AutoDiff" begin
        exact = -0.5*exp(Q.m[end] + Q.V[end, end]/2)
        zyg_calc = dv_var_exp(l, Y[end], Q.m[end], Q.V[end, end])
        @test exact ≈ zyg_calc # Test AutoDiff is giving the same results as exact gradient computation
    end

    @testset "Poisson" begin
        Q = vi(gp, nits=10)
        println("Positive Definite: ", isposdef(Q.V))
        println("-------------------")

        xtest = range(minimum(gp.x),stop=maximum(gp.x),length=50)

        nsamps = 500
        ymean = [];
        visamples = Array{Float64}(undef, nsamps, size(xtest, 1))

        for i in 1:nsamps
            visamples[i, :] = rand(gp, xtest, Q)
            push!(ymean, predict_y(gp, xtest)[1])
        end

        q10 = [quantile(visamples[:,i], 0.1) for i in 1:length(xtest)]
        q50 = [quantile(visamples[:,i], 0.5) for i in 1:length(xtest)]
        q90 = [quantile(visamples[:,i], 0.9) for i in 1:length(xtest)]
        # plot(xtest, exp.(q50), ribbon=(exp.(q10), exp.(q90)), leg=true, fmt=:png, label="quantiles")
        plot(xtest, mean(ymean), label="posterior mean", w=2)
        xx = range(-3,stop=3,length=1000);
        f_xx = 2*cos.(2*xx);
        plot!(xx, exp.(f_xx), label="truth")
        scatter!(X,Y, label="data")
        savefig("vi_gp.png")
    end

    # @testset "Gaussian" begin
    #     n=10;                          #number of training points
    #     x = 2π * rand(n);              #predictors
    #     y = sin.(x) + 0.05*randn(n);   #regressors
    #     mZero = MeanZero()                   #Zero mean function
    #     kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
    #     l = GaussLik(0.1)             # Poisson likelihood

    #     logObsNoise = -1.0                        # log standard deviation of observation noise (this is optional)
    #     gp = GP(x, y, mZero, kern, l)       #Fit the GP

    #     Q = vi(gp)

    #     xtest = range(minimum(gp.x),stop=maximum(gp.x),length=50)

    #     nsamps = 500
    #     ymean = [];
    #     visamples = Array{Float64}(undef, nsamps, size(xtest, 1))

    #     for i in 1:nsamps
    #         visamples[i, :] = rand(gp, xtest, Q)
    #         push!(ymean, predict_y(gp, xtest)[1])
    #     end

    #     q10 = [quantile(visamples[i,:], 0.1) for i in 1:length(xtest)]
    #     q50 = [quantile(visamples[:,i], 0.5) for i in 1:length(xtest)]
    #     q90 = [quantile(visamples[:,i], 0.9) for i in 1:length(xtest)]
    #     plot(xtest, exp.(q50), ribbon=(exp.(q10), exp.(q90)), leg=true, fmt=:png, label="quantiles")
    #     plot!(xtest, mean(ymean), label="posterior mean", w=2)
    #     xx = range(-3,stop=3,length=1000);
    #     f_xx = 2*cos.(2*xx);
    #     plot!(xx, exp.(f_xx), label="truth")
    #     scatter!(X,Y, label="data")
    #     savefig("vi_gp.png")
    # end
 end
 end
