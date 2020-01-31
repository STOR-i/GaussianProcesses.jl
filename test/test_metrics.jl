module TestSparse
    using GaussianProcesses
    using GaussianProcesses: ClfMetric, evaluate, proba, accuracy, precision, recall, classification

    # Simulate Data
    truth = cat(repeat([0], 100), repeat([1], 100), dims=1)
    pred_probs = cat(rand(Uniform(0, 0.5), 20), rand(Uniform(0.5, 1.0), 130), rand(Uniform(0, 0.5), 50), dims=1)
    clf = classification(pred_probs, truth)

    function test_classifier(clf::ClfMetric)
        @test clf. N == 200
        @test clf.Npos == 114.0320041500038
        @test clf.Nneg == 85.9679958499962
        @test clf.Ppos == 130.0
        @test clf.Pneg == 70.0
        @test clf.Tpos == 50.0
        @test clf.Fpos == 80.0
        @test clf.Tneg == 20.0
        @test clf.Fneg == 50.0
        
    function test_pred(gp_sparse, cKPD::PDMat, covstrat::SubsetOfRegsStrategy)
        cK = gp_sparse.cK
        kernel = gp_sparse.kernel
        meanf = gp_sparse.mean
        mx = mean(meanf, xtest)
        xtrain = gp_sparse.x
        alpha = gp_sparse.alpha
        σ2 = exp(2*gp_sparse.logNoise)

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
        σ2 = exp(2*gp_sparse.logNoise)

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
        σ2 = exp(2*gp_sparse.logNoise)

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
            test_sparse(SoR(x', Xu, Y, gp_full.mean, gp_full.kernel, gp_full.logNoise), -3704.0847727395367)
        end
        @testset "Deterministic Training Conditionals" begin
            test_sparse(DTC(x', Xu, Y, gp_full.mean, gp_full.kernel, gp_full.logNoise), -3704.084703493389)
        end
        @testset "Fully Independent Training Conditionals" begin
            test_sparse(FITC(x', Xu, Y, gp_full.mean, gp_full.kernel, gp_full.logNoise), -3709.601737889645)
        end
        @testset "Full Scale Approximation" begin
            test_sparse(FSA(x', Xu, blockindices, Y, gp_full.mean, gp_full.kernel, gp_full.logNoise), -3706.2892293004734)
        end
    end
end
