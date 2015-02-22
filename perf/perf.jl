using gaussianprocesses
using BenchmarkLite


# Define Benchmark test

type GP_Benchmark <: Proc
    gp::GP
    function GP_Benchmark(d::Int, n::Int)
        x = 2π * rand(d,n)                             # Training set
        y = Float64[sum(sin(x[:,i])) for i in 1:n]/d  # y = 1/d Σᵢ sin(xᵢ)
        new(GP(x, y, meanZero, mat32))
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
    predict(proc.gp, s)
end

Base.done(proc::GP_Benchmark, cfg, s) = nothing


# Inputs parameters
dims = [10, 20, 30] # Dimensions
num_train = [20, 50]  # Num training points
num_pred = [30, 60, 100, 200]  # Number of prediction points

# Setup GP and benchmark functions

procs = vec(Proc[ GP_Benchmark(d, n) for n in num_train, d in dims])

rtable = run(procs, num_pred)
show(rtable)
