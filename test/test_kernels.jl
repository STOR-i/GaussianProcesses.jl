module TestV
using GaussianProcesses, Distributions
using Test, Random
Random.seed!(123)

@testset "V1" begin
    n = 20
    x = cat(collect(-1:0.05:1), dims=2)';
    y = 2*cos.(2*x);

    kernel = SquaredExponential(0.5, 1.0)
    mean_func = Zero()
    f = GPR(mean_func, kernel)

    @testset "Kernels" begin
        K1 = cov(kernel, x)
        K2 = cov(kernel, x, x)
        @test K1 == K2
    end
end
end


#  mean_func = Zero()
#  kernel = SquaredExponential(1.0, 1.0)
#  f = Construct_GPR(mean_func, kernel)
#  x = cat(collect(-1:0.01:1), dims=2)'

#  samples = rand(f, x; n_samples=10)
#  p = plot(vec(x), samples, color=:blue, alpha=0.5)



#  μ = mean(mean_func, x)
#  Kxx = cov(kernel, x)
#  Kxx2 = cov(kernel,x, x)
#  @assert Kxx==Kxx2



