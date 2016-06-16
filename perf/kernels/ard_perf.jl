include("kern_proc.jl")

d = 10
n = [50, 100, 500]

λ = rand(d)

cov_procs = Array(Proc, 0)
push!(cov_procs, KernelTest(SEArd(λ, 1.0), d, GaussianProcesses.cov))
push!(cov_procs, KernelTest(Mat12Ard(λ, 1.0), d, GaussianProcesses.cov))
push!(cov_procs, KernelTest(Mat32Ard(λ, 1.0), d, GaussianProcesses.cov))
push!(cov_procs, KernelTest(Mat52Ard(λ, 1.0), d, GaussianProcesses.cov))
push!(cov_procs, KernelTest(RQArd(λ, 1.0, 1.0), d, GaussianProcesses.cov))
cov_table = run(cov_procs, n)

grad_stack_procs = Array(Proc, 0)
push!(grad_stack_procs, KernelTest(SEArd(λ, 1.0), d, GaussianProcesses.grad_stack))
push!(grad_stack_procs, KernelTest(Mat12Ard(λ, 1.0), d, GaussianProcesses.grad_stack))
push!(grad_stack_procs, KernelTest(Mat32Ard(λ, 1.0), d, GaussianProcesses.grad_stack))
push!(grad_stack_procs, KernelTest(Mat52Ard(λ, 1.0), d, GaussianProcesses.grad_stack))
push!(grad_stack_procs, KernelTest(RQArd(λ, 1.0, 1.0), d, GaussianProcesses.grad_stack))
grad_stack_table = run(grad_stack_procs, n)

println("\ncov calculations")
show(cov_table)

println("\grad_stack calculations")
show(grad_stack_table)
