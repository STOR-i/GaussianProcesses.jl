using GaussianProcesses
using Base.Test

# Add tests here...
println("Running test_utils.jl...")
include("test_utils.jl")

println("Running test_GP.jl...")
include("test_GP.jl")

println("Running test_kernels.jl...")
include("test_kernels.jl")
