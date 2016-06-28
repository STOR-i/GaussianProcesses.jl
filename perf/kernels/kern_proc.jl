using GaussianProcesses
using BenchmarkLite

# Define Benchmark test

type KernelTest <: Proc
    k::Kernel
    d::Int
    op::Function
    KernelTest(k::Kernel, d::Int, op::Function) = new(k, d, op)
end

AbstractString(proc::KernelTest) = "$(typeof(proc.k)), d=$(proc.d)"
Base.length(proc::KernelTest, cfg) = cfg
Base.isvalid(proc::KernelTest, cfg) = (isa(cfg, Int) && cfg > 0)

function Base.start(proc::KernelTest, cfg)
    n = cfg
    2π * rand(proc.d,n)
end

function Base.run(proc::KernelTest, cfg, s)
    proc.op(proc.k, s)
end

Base.done(proc::KernelTest, cfg, s) = nothing
