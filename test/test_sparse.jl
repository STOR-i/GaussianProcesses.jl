module TestSparse
    using GaussianProcesses
    using GaussianProcesses: get_params, set_params!, update_mll!, update_mll_and_dmll!, init_precompute, predictMVN, FullCovariance, SoR, FITC, DTC, FSA, SubsetOfRegsStrategy, DeterminTrainCondStrat, FullyIndepStrat, FullScaleApproxStrat, getQaa, getQab, predictMVN!
    using Test, Random
    using Distributions: Beta, Normal
    import Calculus
    using LinearAlgebra
    using Statistics
    using PDMats: PDMat

    """ The true function we will be simulating from. """
    function fstar(x::Float64)
        return abs(x-5)*cos(2*x)
    end

    σy = 10.0
    n=1000
    ntest=30

    Random.seed!(1)
    Xdistr = Beta(7,7)
    ϵdistr = Normal(0,σy)
    x = rand(Xdistr, n)*10
    Y = fstar.(x) .+ rand(ϵdistr,n)
    k = SEIso(log(0.3), log(5.0))
    gp_full = GPE(x', Y, MeanConst(mean(Y)), k, log(σy))
    init_params = get_params(gp_full; noise=true, domean=true, kern=true)
    set_params!(gp_full, init_params; noise=true, domean=true, kern=true)
    update_mll!(gp_full)

    Random.seed!(1)
    Xu    = Matrix(quantile(x, [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.98])')
    xtest = Matrix(rand(Xdistr, ntest)'.*10)

    inearest = [argmin(abs.(xi.-Xu[1,:])) for xi in x]
    blockindices = [findall(isequal(i), inearest) for i in 1:size(Xu,2)]

    function test_pred(gp_sparse, cKPD::PDMat, covstrat::SubsetOfRegsStrategy)
        cK = gp_sparse.cK
        kernel = gp_sparse.kernel
        meanf = gp_sparse.mean
        mx = mean(meanf, xtest)
        xtrain = gp_sparse.x
        alpha = gp_sparse.alpha
        σ2 = exp(2*gp_sparse.logNoise.value)

        μpred, Σpred = predict_f(gp_sparse, xtest; full_cov=true)

        # see Quiñonero-Candela & Rasmussen 2005, eq. 15
        Qfx = getQab(cK, kernel, xtrain, xtest)
        # Qff = getQaa(cK, kernel, xtrain)
        Qxx = getQaa(cK, kernel, xtest)

        μ_alt, Σ_alt = predictMVN!(Qxx, cKPD, Qfx, mx, alpha)
        @test μ_alt ≈ μpred atol=1e-6
        @test Σ_alt ≈ Σpred rtol=1e-3 # should this be better?
    end
    function test_pred(gp_sparse, cKPD::PDMat, covstrat::Union{DeterminTrainCondStrat,FullyIndepStrat})
        cK = gp_sparse.cK
        kernel = gp_sparse.kernel
        meanf = gp_sparse.mean
        mx = mean(meanf, xtest)
        xtrain = gp_sparse.x
        alpha = gp_sparse.alpha
        σ2 = exp(2*gp_sparse.logNoise.value)

        μpred, Σpred = predict_f(gp_sparse, xtest; full_cov=true)

        # see Quiñonero-Candela & Rasmussen 2005, eq. 19 and 23
        Qfx = getQab(cK, kernel, xtrain, xtest)
        Kxx = cov(kernel, xtest)

        μ_alt, Σ_alt = predictMVN!(Kxx, cKPD, Qfx, mx, alpha)
        @test μ_alt ≈ μpred atol=1e-6
        @test Σ_alt ≈ Σpred rtol=1e-3 # should this be better?
    end
    function test_pred(gp_sparse, cKPD::PDMat, covstrat::FullScaleApproxStrat)
        cK = gp_sparse.cK
        kernel = gp_sparse.kernel
        meanf = gp_sparse.mean
        mx = mean(meanf, xtest)
        xtrain = gp_sparse.x
        alpha = gp_sparse.alpha
        σ2 = exp(2*gp_sparse.logNoise.value)

        iprednearest = [argmin(abs.(xi.-Xu[1,:])) for xi in vec(xtest)]
        blockindpred = [findall(isequal(i), iprednearest) 
                        for i in 1:size(Xu,2)]
        ;

        μpred, Σpred = predict_f(gp_sparse, xtest, blockindpred; full_cov=true)

        Qfx = getQab(cK, kernel, xtrain, xtest)
        blockindtrain = blockindices
        Λfx = let
            nf, nx = size(xtrain, 2), size(xtest,2)
            zeros(nf, nx)
        end
        for (predblock, trainblock) in zip(blockindpred, blockindtrain)
            Xpredblock, Xtrainblock = xtest[:,predblock], xtrain[:,trainblock]
            Kfx_block = cov(kernel, Xtrainblock, Xpredblock)
            Qfx_block = getQab(cK, kernel, Xtrainblock, Xpredblock)
            Λfx[trainblock,predblock] = Kfx_block - Qfx_block
        end

        Kxx = cov(kernel, xtest)

        μ_alt, Σ_alt = predictMVN!(Kxx, cKPD, Qfx+Λfx, mx, alpha)
        @test μ_alt ≈ μpred atol=1e-6
        @test Σ_alt ≈ Σpred rtol=1e-3 # should this be better?
    end

    function test_sparse(gp_sparse, expect_mll)
        @test gp_sparse.mll ≈ gp_full.mll atol=10 # marginal loglik shouldn't be radically different
        @test gp_sparse.mll ≈ expect_mll atol=1e-6 # from previous run, check this doesn't drift

        cK = gp_sparse.cK
        cKmat = Matrix(cK)
        cKPD = PDMat(cKmat)

        gp_fromsparse = let
            gp = GPE(x', Y, MeanConst(mean(Y)), k, log(σy))
            gp.cK = cKPD
            update_mll!(gp; noise=false, kern=false, domean=true)
            gp
        end
        @test gp_sparse.mll ≈ gp_fromsparse.mll atol=1e-6

        @test tr(cK) ≈ tr(cKmat) rtol=1e-6
        @test logdet(cK) ≈ logdet(cKmat) rtol=1e-6
        xrand = randn(gp_sparse.nobs,2)
        @test cKPD\xrand ≈ cK\xrand atol=1e-6

        buf = init_precompute(gp_sparse)
        update_mll_and_dmll!(gp_sparse, buf; noise=true, domean=true, kern=true)
        grad_analytical = copy(gp_sparse.dmll)
        grad_numerical = Calculus.gradient(init_params) do params
            set_params!(gp_sparse, params; noise=true, domean=true, kern=true)
            GaussianProcesses.update_mll!(gp_sparse)
            t = gp_sparse.mll
            set_params!(gp_sparse, init_params; noise=true, domean=true, kern=true)
            t
        end
        @test grad_numerical ≈ grad_analytical  atol=1e-3
        test_pred(gp_sparse, cKPD, gp_sparse.covstrat)
    end

    @testset "Sparse Approximations" begin
        @testset "Subset of Regressors" begin
            test_sparse(SoR(x', Xu, Y, gp_full.mean, gp_full.kernel, gp_full.logNoise.value), -3704.0847727395367)
        end
        @testset "Deterministic Training Conditionals" begin
            test_sparse(DTC(x', Xu, Y, gp_full.mean, gp_full.kernel, gp_full.logNoise.value), -3704.084703493389)
        end
        @testset "Fully Independent Training Conditionals" begin
            test_sparse(FITC(x', Xu, Y, gp_full.mean, gp_full.kernel, gp_full.logNoise.value), -3709.601737889645)
        end
        @testset "Full Scale Approximation" begin
            test_sparse(FSA(x', Xu, blockindices, Y, gp_full.mean, gp_full.kernel, gp_full.logNoise.value), -3706.2892293004734)
        end
    end
end
