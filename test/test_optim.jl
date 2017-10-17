using Base: Test
using GaussianProcesses
using GaussianProcesses: get_params, set_params!, get_param_names, Mean, Kernel, Likelihood, update_target_and_dtarget!, update_ll_and_dll!, prior_gradlogpdf
using StatsFuns


""" Not much of a test really... just checks that it doesn't crash
    and that the final mll is better that the initial value
"""
function test_gpe_optim(mean::Mean, kern::Kernel, X::Matrix{Float64}, y::Vector{Float64})
	gp = GPE(X, y, mean, kern, -3.0)
	init_target = gp.target
	optimize!(gp)
	@test gp.target > init_target
end

function test_gpmc_optim(mean::Mean, kern::Kernel, lik::Likelihood, X::Matrix{Float64}, y::Vector{<:Real})
	gp = GPMC(X, y, mean, kern, lik)
	init_target = gp.target
	optimize!(gp)
	@test gp.target > init_target
end

function test_gpe_optim_params_options(mean::Mean, kern::Kernel, noise::Float64, X::Matrix{Float64}, y::Vector{Float64})
    gp = GPE(X, y, mean, kern, noise)
    init_params = get_params(gp; mean=true, kern=true, noise=true)
    
    # Check mean fixed
    mean_params = get_params(gp; mean=true, kern=false, noise=false)
    optimize!(gp; mean=false, kern=true, noise=true)
    @test mean_params == get_params(gp; mean=true, kern=false, noise=false)

    set_params!(gp, init_params; mean=true, kern=true, noise=true)
    
    # Check kern fixed
    kern_params = get_params(gp; mean=false, kern=true, noise=false)
    optimize!(gp; mean=true, kern=false, noise=true)
    @test kern_params == get_params(gp; mean=false, kern=true, noise=false)

    set_params!(gp, init_params; mean=true, kern=true, noise=true)

    # Check noise fixed
    noise_params = get_params(gp; mean=false, kern=false, noise=true)
    optimize!(gp; mean=true, kern=true, noise=false)
    @test noise_params == get_params(gp; mean=false, kern=false, noise=true)
end

function test_gpmc_optim_params_options(mean::Mean, kern::Kernel, lik::Likelihood, X::Matrix{Float64}, y::Vector{<:Real})
    gp = GPMC(X, y, mean, kern, lik)
    init_params = get_params(gp; mean=true, kern=true, lik=true)
    
    # Check mean fixed
    mean_params = get_params(gp.m)
    optimize!(gp; mean=false, kern=true, lik=true)
    @test mean_params == get_params(mean)

    set_params!(gp, init_params; mean=true, kern=true, lik=true)
    
    # Check kern fixed
    kern_params = get_params(gp.k)
    optimize!(gp; mean=true, kern=false, lik=true)
    @test kern_params == get_params(kern)
    
    set_params!(gp, init_params; mean=true, kern=true, lik=true)

    # Check lik fixed
    lik_params = get_params(gp.lik)
    optimize!(gp; mean=true, kern=true, lik=false)
    @test lik_params == get_params(lik)
end


function test_fixed_kernel(kern::Kernel, X::Matrix{Float64}, y::Vector{Float64})
    init_param = get_params(kern)[1]
    fixed=fix(kern, get_param_names(kern)[1])
	gp = GP(X,y,MeanZero(), fixed, -1.0)
	init_target = gp.target
	optimize!(gp)
	@test gp.target > init_target
	@test get_params(kern)[1] == init_param
end

d, n = 2, 20

# GPE tests

X = 2π * rand(d, n)
y = X'rand(d) + 0.1*randn(n)

mean = MeanLin(zeros(d))
kern = SE(1.0, 1.0)
noise = 0.0

test_gpe_optim(mean, kern, X, y)
test_gpe_optim_params_options(mean, kern, noise, X, y)

# GPMC tests
X = 2π * randn(d, n)
f = X'rand(d) + 0.1*randn(n)
y = collect(rand(n) .< normcdf.(f)) # Binary data
lik = BernLik();  # Bernoulli likelihood for binary data {0,1}
test_gpmc_optim(mean, kern, lik, X, y)
test_gpmc_optim_params_options(mean, kern, lik, X, y)


# update_target_and_dtarget!(gp; lik=true, mean=false, kern=true)

# showall(get_params(gp; mean=true, kern=true, lik=true))
