using Random

Random.seed!(1)

const TESTS = ["utils.jl",
               "means.jl",
               "kernels.jl",
               "gp.jl",
               "optim.jl",
               "mcmc.jl",
               "gpmc.jl"]

for test in TESTS
    @time include(test)
end
