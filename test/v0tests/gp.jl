module TestGP
using GaussianProcesses, ScikitLearnBase
using GaussianProcesses: set_params!, get_params
using Test, Random
using LinearAlgebra: diag

Random.seed!(1)

@testset "GP" begin
    d, n = 3, 10

    X = 2π * rand(d, n)
    y = [sum(sin, view(X, :, i)) / d for i in 1:n]
    mZero = MeanZero()
    kern = SE(0.0, 0.0)

    ntest = 5
    Xtest = randn(d, ntest)

    @testset "GPE constructors" begin
        gp = GP(X, y, mZero, kern)
        gp = GPE(X, y, mZero, kern)
        gp = GPE(X, y, mZero, kern, 1.2)
        gp = GPE(X, y, mZero, kern, GaussianProcesses.Scalar(1.2))
    end

    gp = GP(X, y, mZero, kern)


    # Verify that predictive mean at input observations
    # are the same as the output observations
    @testset "Predictive mean at obs locations" begin
        y_pred, σ2 = predict_y(gp, X)
        @test maximum(abs, gp.y - y_pred) ≈ 0.0 atol=0.1
        y_pred, pred_cov = predict_y(gp, X; full_cov=true)
        @test maximum(abs, gp.y - y_pred) ≈ 0.0 atol=0.1
        @test σ2 ≈ diag(pred_cov)
    end
    @testset "Predictive mean at test locations" begin
        y_pred, sig = predict_y(gp, Xtest)
        y_pred, sig = predict_y(gp, Xtest; full_cov=true)
    end

    # ScikitLearn interface test
    @testset "ScikitLearn interface" begin
        gp_sk = ScikitLearnBase.fit!(GPE(), X', y)
        y_pred = ScikitLearnBase.predict(gp_sk, X')
        @test maximum(abs, gp_sk.y - y_pred) ≈ 0.0 atol=0.1
    end

    # Modify kernel and update
    @testset "Update" begin
        gp.kernel.ℓ2 = 4.0
        X_pred = 2π * rand(d, n)

        GaussianProcesses.update_target!(gp)
        y_pred, sig = predict_y(gp, X_pred)
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
