using GaussianProcesses, Base.Test
using GaussianProcesses: get_params, get_param_names, KernelData, EmptyData
using GaussianProcesses: set_params!, num_params, grad_stack!
using GaussianProcesses: GP, MeanConst, update_mll!, update_mll_and_dmll!
import Calculus

function test_cov(kern::Kernel, X::Matrix{Float64})
    spec = GaussianProcesses.cov(kern, X)
    gen = invoke(GaussianProcesses.cov, (Kernel, Matrix{Float64}), kern, X)
    @test_approx_eq spec gen
    dim,n=size(X)
    i = rand(1:n)
    j = rand(1:n)
    cK_ij = GaussianProcesses.cov(kern, X[:,i], X[:,j])
    @test_approx_eq cK_ij spec[i,j]
    cK_added = zeros(n,n)
    GaussianProcesses.addcov!(cK_added, kern, X)
    @test_approx_eq spec cK_added
    cK_added[:,:] = 0.0
    GaussianProcesses.addcov!(cK_added, kern, X, KernelData(kern,X))
    @test_approx_eq spec cK_added
    cK_prod = ones(n,n)
    GaussianProcesses.multcov!(cK_prod, kern, X)
    @test_approx_eq spec cK_prod
    cK_prod[:,:] = 1.0
    GaussianProcesses.multcov!(cK_prod, kern, X, KernelData(kern,X))
    @test_approx_eq spec cK_prod
end

function test_grad_stack(kern::Kernel, X::Matrix{Float64})
    n = num_params(kern)
    data = KernelData(kern, X)
    d, nobsv = size(X)
    stack1 = Array(Float64, nobsv, nobsv, n)
    stack2 = Array(Float64, nobsv, nobsv, n)
    
    grad_stack!(stack1, kern, X, data)
    invoke(grad_stack!, (AbstractArray, Kernel, Matrix{Float64}, EmptyData), stack2, kern, X, EmptyData())
    @test_approx_eq stack1 stack2
end

function test_gradient(kern::Kernel, X::Matrix{Float64})
    init_params = get_params(kern)
    data = KernelData(kern, X)
    nobsv = size(X,2)
    function f(hyp)
        set_params!(kern, hyp)
        s = sum(cov(kern, X))
        return s
    end
    stack = Array(Float64, nobsv, nobsv, num_params(kern))
    grad_stack!(stack, kern, X, data)
    theor_grad = vec(sum(stack, [1,2]))
    numer_grad = Calculus.gradient(f, init_params)
    for i in 1:length(theor_grad)
        @assert isapprox(theor_grad[i], numer_grad[i], rtol=1e-1, atol=1e-2) string(
            theor_grad, " != ", numer_grad, " at index ", i
            )
    end
end

function test_dmLL(kern::Kernel, X::Matrix{Float64}, y::Vector{Float64})
    gp = GP(X,y,MeanConst(0.0), kern, -3.0)
    init_params = get_params(gp)
    nobsv = size(X,2)
    function f(hyp)
        set_params!(gp, hyp)
        update_mll!(gp)
        return gp.mLL
    end
    update_mll_and_dmll!(gp)
    theor_grad = copy(gp.dmLL)
    numer_grad = Calculus.gradient(f, init_params)
    set_params!(gp, init_params)
    for i in 1:length(theor_grad)
        @assert isapprox(theor_grad[i], numer_grad[i], rtol=1e-3, atol=1e-3) string(
            theor_grad, " != ", numer_grad, " at index ", i
            )
    end
end

function test_Kernel(kern::Kernel, x::Matrix{Float64}, y::Vector{Float64})
    t = typeof(kern)
    println("\tTesting $(t)...")
    @assert length(get_param_names(kern)) == length(get_params(kern)) == num_params(kern)
    test_cov(kern, x)
    test_grad_stack(kern, x)
    test_gradient(kern, x)
    test_dmLL(kern, x, y)
end
    
d, n = 2, 4
ll = rand(d)
x = 2π * rand(d, n)
y = randn(n)

# Isotropic kernels

se = SEIso(1.0, 1.0)
test_Kernel(se, x, y)

mat12 = Mat12Iso(1.0,1.0)
test_Kernel(mat12, x, y)

mat32 = Mat32Iso(1.0,1.0)
test_Kernel(mat32, x, y)

mat52 = Mat52Iso(1.0,1.0)
test_Kernel(mat52, x, y)

rq = RQIso(1.0, 1.0, 1.0)
test_Kernel(rq, x, y)

peri = Periodic(1.0, 1.0, 2π)
test_Kernel(peri, x, y)

# Non-isotropic

lin = Lin(1.0)
test_Kernel(lin, x, y)

poly = Poly(0.0, 0.0, 2)
test_Kernel(poly, x, y)

noise = Noise(1.0)
test_Kernel(noise, x, y)

# ARD kernels

se_ard = SEArd(ll, 1.0)
test_Kernel(se_ard, x, y)

mat12_ard = Mat12Ard(ll, 1.0)
test_Kernel(mat12_ard, x, y)

mat32_ard = Mat32Ard(ll, 1.0)
test_Kernel(mat32_ard, x, y)

mat52_ard = Mat52Ard(ll, 1.0)
test_Kernel(mat52_ard, x, y)

rq_ard=RQArd(ll, 0.0, 2.0)
test_Kernel(rq_ard, x, y)

lin_ard = LinArd(ll)
test_Kernel(lin_ard, x, y)

# Composite kernels

sum_kern = se + mat12
test_Kernel(sum_kern, x, y)

sum_kern_3 = sum_kern + lin
test_Kernel(sum_kern_3, x, y)

prod_kern = se * mat12
test_Kernel(prod_kern, x, y)

prod_kern_3 = prod_kern * lin
test_Kernel(prod_kern_3, x, y)
