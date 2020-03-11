module TestMCMC
using GaussianProcesses, Distributions
using AdvancedHMC
using Test, Random

Random.seed!(1)

@testset "MCMC" begin
    d, n = 1, 20
    ll = rand(d)
    X = 2π * rand(d, n)
    y = randn(n) .+ 0.5

    kern = RQ(-1.0, -1.0, -1.0)

    # Just checks that it doesn't crash
    @testset "Legacy MCMC" begin
        @testset "Without likelihood" begin
            gp = GP(X, y, MeanZero(), kern)
            set_priors!(gp.kernel, [Distributions.Normal(-1.0, 1.0) for i in 1:3])
            global hmc_chain = mcmc(gp, ε=0.05)
        end

        @testset "With likelihood" begin
            lik = GaussLik(-1.0)
            gp = GP(X, y, MeanZero(), kern, lik)
            set_priors!(gp.kernel, [Distributions.Normal(-1.0, 1.0) for i in 1:3])
            mcmc(gp, ε=0.05)
        end
    end

    @testset "HMC" begin
        @testset "Without likelihood" begin
            gp = GP(X, y, MeanZero(), kern)
            set_priors!(gp.kernel, [Distributions.Normal(-1.0, 1.0) for i in 1:3])
            global hmc_chain = hmc(gp, ε=0.05)
        end

        @testset "With likelihood" begin
            lik = GaussLik(-1.0)
            gp = GP(X, y, MeanZero(), kern, lik)
            set_priors!(gp.kernel, [Distributions.Normal(-1.0, 1.0) for i in 1:3])
            hmc(gp, ε=0.05)
        end
    end

    @testset "AdvancedHMC" begin
        @testset "Without likelihood" begin
            gp = GP(X, y, MeanZero(), kern)
            set_priors!(gp.kernel, [Distributions.Normal(-1.0, 1.0) for i in 1:3])
            global hmc_chain = nuts(gp, progress=false)
        end

        @testset "With likelihood" begin
            lik = GaussLik(-1.0)
            gp = GP(X, y, MeanZero(), kern, lik)
            set_priors!(gp.kernel, [Distributions.Normal(-1.0, 1.0) for i in 1:3])
            nuts(gp, nIter=1000, burn=200, progress=true)
        end

        @testset "Use" begin
            lik = GaussLik(-1.0)
            gp = GP(X, y, MeanZero(), kern, lik)
            set_priors!(gp.kernel, [Distributions.Normal(-1.0, 1.0) for i in 1:3])
            kwargs = GaussianProcesses.get_params_kwargs(
                gp; domean=true, kern=true, noise=true, lik=true)

            metric = AdvancedHMC.DenseEuclideanMetric(
                GaussianProcesses.num_params(gp; kwargs...))
            hamiltonian = nuts_hamiltonian(gp, metric=metric)
            ε = 0.1
            integrator = AdvancedHMC.Leapfrog(ε)
            prop = AdvancedHMC.NUTS{SliceTS, ClassicNoUTurn}(integrator)
            adaptor = AdvancedHMC.NaiveHMCAdaptor(
                Preconditioner(metric), NesterovDualAveraging(0.8, integrator))
            nuts(gp, nIter=1000, burn=100, metric=metric, hamiltonian=hamiltonian,
                 ε=ε, integrator=integrator, proposals=prop, adaptor=adaptor, progress=false)
        end
    end

    @testset "ESS" begin
        gpess = GP(X, y, MeanZero(), kern)
        set_priors!(gpess.kernel, [Distributions.Normal(-1.0, 1.0) for i in 1:3])
        set_priors!(gpess.logNoise, [Distributions.Normal(-1.0, 1.0)])
        global ess_chain = ess(gpess)
    end
    
end
end
