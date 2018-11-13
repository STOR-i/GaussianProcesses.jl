module TestOptim
using GaussianProcesses, StatsFuns
using Test, Random

Random.seed!(1)

@testset "Optim" begin
    d, n = 2, 20
    X = 2π * rand(d, n)

    mean = MeanLin(zeros(d))
    kern = SE(1.0, 1.0)

    @testset "Without likelihood" begin
        y = X'rand(d) + 0.1*randn(n)
        noise = 0.0

        # Just checks that it doesn't crash
        # and that the final mll is better that the initial value
        @testset "Basic" begin
	          gp = GPE(X, y, mean, kern, -3.0)
	          init_target = gp.target
	          optimize!(gp)
	          @test gp.target > init_target
        end

        @testset "Fixed kernel" begin
            init_param = GaussianProcesses.get_params(kern)[1]
            fixed = fix(kern, GaussianProcesses.get_param_names(kern)[1])
            gp = GP(X, y, MeanZero(), fixed, -1.0)
            init_target = gp.target
            optimize!(gp)
            @test gp.target > init_target
            @test GaussianProcesses.get_params(kern)[1] == init_param
        end

        @testset "Keyword arguments" begin
            gp = GPE(X, y, mean, kern, noise)
            init_params = GaussianProcesses.get_params(gp; domean=true, kern=true,
                                                       noise=true)

            # Check mean fixed
            mean_params = GaussianProcesses.get_params(gp; domean=true, kern=false,
                                                       noise=false)
            optimize!(gp; domean=false, kern=true, noise=true)
            @test mean_params == GaussianProcesses.get_params(gp; domean=true,
                                                              kern=false, noise=false)

            set_params!(gp, init_params; domean=true, kern=true, noise=true)

            # Check kern fixed
            kern_params = GaussianProcesses.get_params(gp; domean=false, kern=true,
                                                       noise=false)
            optimize!(gp; domean=true, kern=false, noise=true)
            @test kern_params == GaussianProcesses.get_params(gp; domean=false, kern=true,
                                                              noise=false)

            set_params!(gp, init_params; domean=true, kern=true, noise=true)

            # Check noise fixed
            noise_params = GaussianProcesses.get_params(gp; domean=false, kern=false,
                                                        noise=true)
            optimize!(gp; domean=true, kern=true, noise=false)
            @test noise_params == GaussianProcesses.get_params(gp; domean=false, kern=false,
                                                               noise=true)

            set_params!(gp, init_params; domean=true, kern=true, noise=true)

            # Box
            kern_params = GaussianProcesses.get_params(gp; domean=false, kern=true,
                                                       noise=false)
            optimize!(gp, domean = false,
                      kernbounds = [kern_params .- .01, kern_params .+ .01])
            new_kern_params = GaussianProcesses.get_params(gp; domean=false, kern=true,
                                                           noise=false)
            @test (&)((@. kern_params - .01 <= new_kern_params <= kern_params + .01)...)
            @test kern_params != new_kern_params
        end
    end

    @testset "With likelihood" begin
        f = X'rand(d) + 0.1*randn(n)
        y = collect(rand(n) .< normcdf.(f)) # Binary data
        lik = BernLik()  # Bernoulli likelihood for binary data {0,1}

        # Just checks that it doesn't crash
        # and that the final mll is better that the initial value
        @testset "Basic" begin
            gp = GPMC(X, y, mean, kern, lik)
            init_target = gp.target
            optimize!(gp)
            @test gp.target > init_target
        end

        @testset "Keyword arguments" begin
            gp = GPMC(X, y, mean, kern, lik)
            init_params = GaussianProcesses.get_params(gp; domean=true, kern=true, lik=true)

            # Check mean fixed
            mean_params = GaussianProcesses.get_params(gp.mean)
            optimize!(gp; domean=false, kern=true, lik=true)
            @test mean_params == GaussianProcesses.get_params(mean)

            set_params!(gp, init_params; domean=true, kern=true, lik=true)

            # Check kern fixed
            kern_params = GaussianProcesses.get_params(gp.kernel)
            optimize!(gp; domean=true, kern=false, lik=true)
            @test kern_params == GaussianProcesses.get_params(kern)

            set_params!(gp, init_params; domean=true, kern=true, lik=true)

            # Check lik fixed
            lik_params = GaussianProcesses.get_params(gp.lik)
            optimize!(gp; domean=true, kern=true, lik=false)
            @test lik_params == GaussianProcesses.get_params(lik)

            set_params!(gp, init_params; domean=true, kern=true, lik=true)

            # Box
            kern_params = GaussianProcesses.get_params(gp; domean=false, kern=true,
                                                       lik=false)
            optimize!(gp, domean = false,
                      kernbounds = [kern_params .- .01, kern_params .+ .01])
            new_kern_params = GaussianProcesses.get_params(gp; domean=false, kern=true,
                                                           lik=false)
            @test (&)((@. kern_params - .01 <= new_kern_params <= kern_params + .01)...)
            @test kern_params != new_kern_params
        end
    end
end
end
