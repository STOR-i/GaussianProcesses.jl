module TestMeans
using GaussianProcesses, Calculus
using Test, Statistics, Random

Random.seed!(1)

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
    x = view(X, :, 1)

    @testset "Mean $(typeof(m))" for m in means
        println("\tTesting ", nameof(typeof(m)), "...")

        means = mean(m, X)
        params = GaussianProcesses.get_params(m)

        @testset "Run" begin
            mean(m, x)
            GaussianProcesses.grad_mean(m, x)
            GaussianProcesses.grad_stack(m, X)
            show(devnull, m)
            GaussianProcesses.get_param_names(m)
            set_params!(m, params)
        end

        @testset "Consistency" begin
            @test means ≈ invoke(mean, Tuple{GaussianProcesses.Mean, Matrix{Float64}}, m, X)
        end

        @testset "Gradients" begin
            @testset "Observation #$i" for i in 1:n
                Xi = view(X, :, i)
                theor_grad = GaussianProcesses.grad_mean(m, Xi)
                num_grad = Calculus.gradient(params) do params
                    set_params!(m, params)
                    mean(m, Xi)
                end
                @test theor_grad ≈ num_grad rtol=1e-6
            end
        end
    end
end
end
