using GaussianProcesses, Base.Test

function test_crossKern(kern::Kernel, x::Matrix{Float64})
    spec = GaussianProcesses.crossKern(x, kern)
    gen = invoke(GaussianProcesses.crossKern, (Matrix{Float64}, Kernel), x, kern)
    @test_approx_eq spec gen
end

function test_grad_stack(kern::Kernel, x::Matrix{Float64})
    n = GaussianProcesses.num_params(kern)
    d, nobsv = size(x)
    stack = Array(Float64, nobsv, nobsv, n)
    
    spec = GaussianProcesses.grad_stack!(stack, x, kern)
    gen = invoke(GaussianProcesses.grad_stack!, (AbstractArray, Matrix{Float64}, Kernel), stack, x, kern)
    @test_approx_eq spec gen
end

function test_Kernel(kern::Kernel, x::Matrix{Float64})
    t = typeof(kern)
    println("\tTesting $(t)...")
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

# ARD kernels

se_ard = SEArd(ll, 1.0)
test_Kernel(se_ard, x)

mat12_ard = Mat12Ard(ll, 1.0)
test_Kernel(mat12_ard, x)

# Composite kernels

sum = se + mat12
test_Kernel(sum, x)

prod = se + mat12
test_Kernel(prod, x)
