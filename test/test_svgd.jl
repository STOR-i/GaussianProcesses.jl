module TestVI
using GaussianProcesses, Distributions, PDMats, Plots, LinearAlgebra
gr()
using Test, Random
Random.seed!(203617)

@testset "VI" begin
Random.seed!(13579)               # Set the seed using the 'Random' package
n = 20;                           # number of training points
x = 2π * rand(n);                 # predictors
y = sin.(x) + 0.05*randn(n);      # regressors

# Select mean and covariance function
mZero = MeanZero()                  # Zero mean function
kern = SE(0.0,0.0)                  # Sqaured exponential kernel
logObsNoise = -1.0                  # log standard deviation of observation noise
gp = GP(x,y,mZero,kern,logObsNoise) # Fit the GP

xtest = 0:0.1:2π                # a sequence between 0 and 2π with 0.1 spacing
μ, Σ = predict_y(gp,xtest);
optimize!(gp) #Optimise the parameters
set_priors!(kern, [Normal(0,1), Normal(0,1)])

    @testset "MCMC comparison" begin
        particles = svgd(gp;nIter=5000,nParticles=100, ε=0.1)
        chain = mcmc(gp);

        svgd_mean = mean(particles;dims=2)
        mcmc_mean = mean(chain;dims=2)
        svgd_var = var(particles;dims=2)
        mcmc_var = var(chain;dims=2)

        # Check that MCMC and SVGD roughly agree
        @test isapprox(svgd_mean, mcmc_mean, atol=10)
        @test isapprox(svgd_var, mcmc_var, atol=10)
    end
end
end
