using GaussianProcesses, Test
using Statistics, Calculus

@testset "Means" begin
    n = 5 # number of observations
    d = 4 # dimension
    D = 3 # degree of polynomial mean function

    means = [MeanZero(),
             MeanConst(3.0),
             MeanLin(rand(d)),
             MeanPoly(rand(d, D)),
             MeanConst(3.0) * MeanLin(rand(d)),
             MeanLin(rand(d)) + MeanPoly(rand(d, D))]

    X = rand(d, n)

    @testset "Mean $(typeof(m))" for m in means
        println("\tTesting mean ", typeof(m), "...")

        @testset "Run" begin
            mean(m, X[:,1])
            mean(m, X)
            GaussianProcesses.grad_mean(m, X[:,1])
            GaussianProcesses.grad_stack(m, X)
            show(devnull, m)
            p = GaussianProcesses.get_params(m)
            set_params!(m, p)
            GaussianProcesses.get_param_names(m)
        end

        @testset "Consistency" begin
            spec = mean(m, X)
            gen = invoke(mean, Tuple{GaussianProcesses.Mean, Matrix{Float64}}, m, X)
            @test spec ≈ gen
        end

        @testset "Gradients" begin
            init_params = GaussianProcesses.get_params(m)

            @testset "Observation #$i" for i in 1:n
                x = X[:, i]
                theor_grad = GaussianProcesses.grad_mean(m, x)
                num_grad = Calculus.gradient(init_params) do params
                    set_params!(m, params)
                    mean(m, x)
                end
                @test theor_grad ≈ num_grad rtol=1e-6
            end
        end
    end
end
