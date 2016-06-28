using GaussianProcesses
using GaussianProcesses: KernelData
using BenchmarkLite

# Define Benchmark test

type KernelTest <: Proc
    k::Kernel
    d::Int
    op::Function
    KernelTest(k::Kernel, d::Int, op::Function) = new(k, d, op)
end

Base.string(proc::KernelTest) = "$(typeof(proc.k)), d=$(proc.d)"
Base.length(proc::KernelTest, cfg) = cfg
Base.isvalid(proc::KernelTest, cfg) = (isa(cfg, Int) && cfg > 0)

function Base.start(proc::KernelTest, cfg)
    n = cfg
    X = 2π * rand(proc.d, n)
    data = KernelData(proc.k, X)
    return X, data
end

function Base.run(proc::KernelTest, cfg, s)
    X, data = s
    proc.op(proc.k, X, data)
end

Base.done(proc::KernelTest, cfg, s) = nothing
