module TestHetero
using GaussianProcesses, ScikitLearnBase
using GaussianProcesses: set_params!, get_params, num_params,
                         update_cK!, update_mll!, update_target_and_dtarget!
using Test, Random

Random.seed!(1)
@testset "GPE heteroscedastic" begin
    d, n = 3, 10

    X = 2π * rand(d, n)
    y = [sum(sin, view(X, :, i)) / d for i in 1:n]
    mZero = MeanZero()
    kern = SE(0.0, 0.0)
    logNoise = range(0.0, stop=1.0, length=n)

    ntest = 5
    Xtest = randn(d, ntest)

    @testset "VectorParam" begin
        vp = GaussianProcesses.wrap_param(logNoise)
        @test all(get_params(vp) .== logNoise)
        @test num_params(vp) == n
        @test length(vp) == n
    end

    @testset "GPE constructors" begin
        gp1 = GPE(X, y, mZero, kern, logNoise)
        gp2 = GPE(X, y, mZero, kern, GaussianProcesses.wrap_param(logNoise))
    end

    gp = GP(X, y, mZero, kern, logNoise)

    @testset "update_cK!" begin
        update_cK!(gp)
    end
    @testset "update_mll" begin
        update_mll!(gp)
        @test isfinite(gp.mll)
    end

    @testset "update_dmll!" begin
        update_target_and_dtarget!(gp; noise=false)
        @test isfinite(gp.mll)
        @test isfinite(gp.target)
        @test all(isfinite.(gp.dmll))
        @test all(isfinite.(gp.dtarget))
    end

    @testset "num_params" begin
        @test num_params(gp) == 0+2+n
    end

    @testset "show" begin
        show(devnull, gp) # doesn't crash
    end

    @testset "Predictive mean at test locations" begin
        f_pred, sig = predict_f(gp, Xtest)
        f_pred, sig = predict_f(gp, Xtest; full_cov=true)
    end

    # ScikitLearn interface test
    @testset "ScikitLearn interface" begin
        gp_sk = ScikitLearnBase.fit!(GPE(), X', y)
        f_pred = ScikitLearnBase.predict(gp_sk, X')
    end

    # Modify kernel and update
    @testset "Update" begin
        gp.kernel.ℓ2 = 4.0
        X_pred = 2π * rand(d, n)

        GaussianProcesses.update_target!(gp)
        f_pred, sig = predict_f(gp, X_pred)
    end

    #Check that the rand function works
    @testset "Random GP sampling" begin
        X_test = 2π * rand(d, n)
        samples = rand(gp, X_test)
    end

    @testset "params round trip" begin
        params_1 = deepcopy(get_params(gp))
        set_params!(gp, params_1)
        params_2 = get_params(gp)
        @test params_1 ≈ params_2
    end
    
end
end
