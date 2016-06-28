using GaussianProcesses
using BenchmarkLite

# Define Benchmark test

type GP_UpdateBenchmark <: Proc
    d::Int
    op::Function
    GP_UpdateBenchmark(d::Int, op::Function) = new(d, op)
end

AbstractString(proc::GP_UpdateBenchmark) = "Dim: $(proc.d)"
Base.length(proc::GP_UpdateBenchmark, cfg) = cfg
Base.isvalid(proc::GP_UpdateBenchmark, cfg) = (isa(cfg, Int) && cfg > 0)
function Base.start(proc::GP_UpdateBenchmark, cfg)
    n = cfg
    x = 2π * rand(proc.d,n)                              # Training set
    y = Float64[sum(sin(x[:,i])) for i in 1:n]/proc.d    # y = 1/d Σᵢ sin(xᵢ)
    GP(x, y, MeanZero(), SEIso(0.0,0.0))
end

function Base.run(proc::GP_UpdateBenchmark, cfg, s)
    proc.op(s)
end
Base.done(proc::GP_UpdateBenchmark, cfg, s) = nothing

######################

# Inputs parameters
dims = [10, 30, 50, 100] # Dimensions
num_train = [20, 50, 100, 1000]  # Num training points

println("Benchmarking update_mll! function....")
mll_procs = vec(Proc[ GP_UpdateBenchmark(d, GaussianProcesses.update_mll_and_dmll!) for d in dims ])
mll_table = run(mll_procs, num_train)
show(mll_table)
