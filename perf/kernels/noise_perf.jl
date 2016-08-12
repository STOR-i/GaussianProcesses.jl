include("kern_proc.jl")

d = 10
n = [50, 100, 500, 1000]

srand(1)

cov_procs = Array(Proc, 0)
push!(cov_procs, KernelTest(Noise(0.0), d, GaussianProcesses.cov))
push!(cov_procs, KernelTest(SEIso(1.0, 1.0), d, GaussianProcesses.grad_stack))
push!(cov_procs, KernelTest(SEIso(1.0, 1.0) + Noise(0.0), d, GaussianProcesses.cov))
push!(cov_procs, KernelTest(SEArd(rand(d), 1.0), d, GaussianProcesses.cov))
push!(cov_procs, KernelTest(SEArd(rand(d), 1.0) + Noise(0.0), d, GaussianProcesses.cov))
cov_table = run(cov_procs, n)

grad_stack_procs = Array(Proc, 0)
push!(grad_stack_procs, KernelTest(Noise(0.0), d, GaussianProcesses.grad_stack))
push!(grad_stack_procs, KernelTest(SEIso(1.0, 1.0), d, GaussianProcesses.grad_stack))
push!(grad_stack_procs, KernelTest(SEIso(1.0, 1.0) + Noise(0.0), d, GaussianProcesses.grad_stack))
push!(grad_stack_procs, KernelTest(SEArd(rand(d), 1.0), d, GaussianProcesses.grad_stack))
push!(grad_stack_procs, KernelTest(SEArd(rand(d), 1.0) + Noise(0.0), d, GaussianProcesses.grad_stack))
grad_stack_table = run(grad_stack_procs, n)

println("\ncov calculations")
show(cov_table)
println("\grad_stack calculations")
show(grad_stack_table)
