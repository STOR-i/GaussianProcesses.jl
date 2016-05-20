using GaussianProcesses, Base.Test
using GaussianProcesses: get_params, get_param_names, num_params

function test_crossKern(kern::Kernel, x::Matrix{Float64})
    spec = GaussianProcesses.crossKern(x, kern)
    gen = invoke(GaussianProcesses.crossKern, (Matrix{Float64}, Kernel), x, kern)
    @test_approx_eq spec gen
end

function test_grad_stack(kern::Kernel, x::Matrix{Float64})
    n = GaussianProcesses.num_params(kern)
    d, nobsv = size(x)
    stack1 = Array(Float64, nobsv, nobsv, n)
    stack2 = Array(Float64, nobsv, nobsv, n)
    
    GaussianProcesses.grad_stack!(stack1, x, kern)
    invoke(GaussianProcesses.grad_stack!, (AbstractArray, Matrix{Float64}, Kernel), stack2, x, kern)
    @test_approx_eq stack1 stack2
end

function test_Kernel(kern::Kernel, x::Matrix{Float64})
    t = typeof(kern)
    println("\tTesting $(t)...")
    @assert length(get_param_names(kern)) == length(get_params(kern)) == num_params(kern)
    test_crossKern(kern, x)
    test_grad_stack(kern, x)
end
    
d, n = 5, 4
ll = rand(d)
x = 2π * rand(d, n)

# Isotropic kernels

se = SEIso(1.0, 1.0)
test_Kernel(se, x)

mat12 = Mat12Iso(1.0,1.0)
test_Kernel(mat12, x)

mat32 = Mat32Iso(1.0,1.0)
test_Kernel(mat32, x)

mat52 = Mat52Iso(1.0,1.0)
test_Kernel(mat52, x)

rq = RQIso(1.0, 1.0, 1.0)
test_Kernel(rq, x)

peri = Periodic(1.0, 1.0, 2π)
test_Kernel(peri, x)

# Non-isotropic

lin = Lin(1.0)
test_Kernel(lin, x)

poly = Poly(0.0, 0.0, 2)
test_Kernel(poly, x)

noise = Noise(1.0)
test_Kernel(noise, x)

# ARD kernels

se_ard = SEArd(ll, 1.0)
test_Kernel(se_ard, x)

mat12_ard = Mat12Ard(ll, 1.0)
test_Kernel(mat12_ard, x)

mat32_ard = Mat32Ard(ll, 1.0)
test_Kernel(mat32_ard, x)

mat52_ard = Mat52Ard(ll, 1.0)
test_Kernel(mat52_ard, x)

rq_ard=RQArd(ll, 0.0, 2.0)
test_Kernel(rq_ard, x)

lin_ard = LinArd(ll)
test_Kernel(lin_ard, x)

# Composite kernels

sum_kern = se + mat12
test_Kernel(sum_kern, x)

prod_kern = se * mat12
test_Kernel(prod_kern, x)
