using GaussianProcesses
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
    2π * rand(proc.d,n)
end

function Base.run(proc::KernelTest, cfg, s)
    proc.op(s, proc.k)
end

Base.done(proc::KernelTest, cfg, s) = nothing

d = 10
n = [50, 100, 1000, 2000]

cross_kern_procs = Array(Proc, 0)
push!(cross_kern_procs, KernelTest(SEIso(1.0, 1.0), d, GaussianProcesses.crossKern))
push!(cross_kern_procs, KernelTest(Mat12Iso(1.0, 1.0), d, GaussianProcesses.crossKern))
push!(cross_kern_procs, KernelTest(RQIso(1.0, 1.0, 1.0), d, GaussianProcesses.crossKern))
push!(cross_kern_procs, KernelTest(Periodic(1.0, 1.0, 1.0), d, GaussianProcesses.crossKern))
cross_kern_table = run(cross_kern_procs, n)

grad_stack_procs = Array(Proc, 0)
push!(grad_stack_procs, KernelTest(SEIso(1.0, 1.0), d, GaussianProcesses.grad_stack))
push!(grad_stack_procs, KernelTest(Mat12Iso(1.0, 1.0), d, GaussianProcesses.grad_stack))
push!(grad_stack_procs, KernelTest(RQIso(1.0, 1.0, 1.0), d, GaussianProcesses.grad_stack))
push!(grad_stack_procs, KernelTest(Periodic(1.0, 1.0, 1.0), d, GaussianProcesses.grad_stack))
grad_stack_table = run(grad_stack_procs, n)

println("\ncrossKern calculations")
show(cross_kern_table)

println("\grad_stack calculations")
show(grad_stack_table)
