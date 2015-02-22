using gaussianprocesses
using BenchmarkLite


# Define Benchmark test

type GP_Benchmark <: Proc
    gp::GP
    op::Function
    function GP_Benchmark(d::Int, n::Int, op::Function)
        x = 2π * rand(d,n)                             # Training set
        y = Float64[sum(sin(x[:,i])) for i in 1:n]/d  # y = 1/d Σᵢ sin(xᵢ)
        new(GP(x, y, meanZero, mat32), op)
    end
end

function Base.string(proc::GP_Benchmark)
    dim = proc.gp.dim
    n = proc.gp.nobvs 
    "Dim: $(dim), Nobvs: $(n)"
end

Base.length(proc::GP_Benchmark, cfg) = cfg
Base.isvalid(proc::GP_Benchmark, cfg) = (isa(cfg, Int) && cfg > 0)
function Base.start(proc::GP_Benchmark, cfg)
    2π * rand(proc.gp.dim, cfg)
end

function Base.run(proc::GP_Benchmark, cfg, s)
    proc.op(proc.gp, s)
end

Base.done(proc::GP_Benchmark, cfg, s) = nothing

# Inputs parameters
dims = [10, 30] # Dimensions
num_train = [20, 50]  # Num training points
num_pred = [30, 60, 100, 200]  # Number of prediction points

# Setup GP and benchmark functions

println("Benchmarking predict function....")
predict_procs = vec(Proc[ GP_Benchmark(d, n, predict) for n in num_train, d in dims ])
predict_table = run(predict_procs, num_pred)
show(predict_table)


println("\nBenchmarking EI function...")
ei_procs = vec(Proc[ GP_Benchmark(d,n, EI) for n in num_train, d in dims ])
ei_table = run(ei_procs, num_pred)
show(ei_table)
