using GaussianProcesses
using Base.Test

# Add tests here...
println("Running test_utils.jl...")
include("test_utils.jl")

println("Running test_GP.jl...")
include("test_GP.jl")

println("Running test_optim.jl...")
include("test_optim.jl")

println("Running test_mcmc.jl...")
include("test_mcmc.jl")

println("Running test_kernels.jl...")
include("test_kernels.jl")

println("Running test_gpmc.jl...")
include("test_gpmc.jl")
