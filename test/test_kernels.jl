using GaussianProcesses, Base.Test
using GaussianProcesses: get_params, get_param_names
using GaussianProcesses: kernel_data_key, KernelData, EmptyData
using GaussianProcesses: set_params!, num_params, grad_stack!
using GaussianProcesses: GP, MeanConst, update_target!, update_target_and_dtarget!
using GaussianProcesses: StationaryARD
import Calculus

function test_cov_x1x2(kern::Kernel, X1::Matrix{Float64}, X2::Matrix{Float64})
    spec = GaussianProcesses.cov(kern, X1, X2)
    dim,n1=size(X1)
    dim,n2=size(X2)
    i = rand(1:n1)
    j = rand(1:n2)
    cK_ij = GaussianProcesses.cov(kern, X1[:,i], X2[:,j])
    @test cK_ij ≈ spec[i,j]
    cK_added = zeros(n1,n2)
    GaussianProcesses.addcov!(cK_added, kern, X1, X2)
    @test spec ≈ cK_added
    cK_prod = ones(n1,n2)
    GaussianProcesses.multcov!(cK_prod, kern, X1, X2)
    @test spec ≈ cK_prod
end
function test_cov(kern::Kernel, X::Matrix{Float64})
    spec = GaussianProcesses.cov(kern, X)
    gen = invoke(GaussianProcesses.cov, Tuple{Kernel, Matrix{Float64}}, kern, X)
    @test spec ≈ gen
    dim,n=size(X)
    i = rand(1:n)
    j = rand(1:n)
    cK_ij = GaussianProcesses.cov(kern, X[:,i], X[:,j])
    @test cK_ij ≈ spec[i,j]
    cK_added = zeros(n,n)
    GaussianProcesses.addcov!(cK_added, kern, X)
    @test spec ≈ cK_added
    cK_added[:,:] = 0.0
    kdata = KernelData(kern,X)
    GaussianProcesses.addcov!(cK_added, kern, X, kdata)
    @test spec ≈ cK_added
    cK_prod = ones(n,n)
    GaussianProcesses.multcov!(cK_prod, kern, X)
    @test spec ≈ cK_prod
    cK_prod[:,:] = 1.0
    GaussianProcesses.multcov!(cK_prod, kern, X, KernelData(kern,X))
    @test spec ≈ cK_prod
    key = kernel_data_key(kern, X)
    @test typeof(key) == String
    # check we've overwritten the default if necessary
    if typeof(kdata) != EmptyData
        @test key != "EmptyData"
    end
end

function test_grad_stack(kern::Kernel, X::Matrix{Float64})
    n = num_params(kern)
    data = KernelData(kern, X)
    d, nobsv = size(X)
    stack1 = Array{Float64}( nobsv, nobsv, n)
    stack2 = Array{Float64}( nobsv, nobsv, n)
    
    grad_stack!(stack1, kern, X, data)
    invoke(grad_stack!, Tuple{AbstractArray, Kernel, Matrix{Float64}, EmptyData}, stack2, kern, X, EmptyData())
    @test stack1 ≈ stack2
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
    stack = Array{Float64}(nobsv, nobsv, num_params(kern))
    grad_stack!(stack, kern, X, data)
    theor_grad = vec(sum(stack, [1,2]))
    numer_grad = Calculus.gradient(f, init_params)
    for i in 1:length(theor_grad)
        @assert isapprox(theor_grad[i], numer_grad[i], rtol=1e-1, atol=1e-2) string(
            theor_grad, " != ", numer_grad, " at index ", i
            )
    end
end

function test_dtarget(kern::Kernel, X::Matrix{Float64}, y::Vector{Float64})
    gp = GPE(X,y,MeanConst(0.0), kern, -3.0)
    init_params = get_params(gp)
    nobsv = size(X,2)
    function f(hyp)
        set_params!(gp, hyp)
        update_target!(gp)
        return gp.target
    end
    update_target_and_dtarget!(gp)
    theor_grad = copy(gp.dtarget)
    numer_grad = Calculus.gradient(f, init_params)
    set_params!(gp, init_params)
    for i in 1:length(theor_grad)
        @assert isapprox(theor_grad[i], numer_grad[i], rtol=1e-3, atol=1e-3) string(
            "(thereotical) ", theor_grad, " != (numerical) ", numer_grad, " at index ", i
            )
    end
end

function test_masked(kern::Kernel, x::Matrix, x2::Matrix, y::Vector)
    masked = Masked(kern, [1])
    test_cov(masked, x)
    test_cov_x1x2(masked, x, x2)
    test_grad_stack(masked, x)
    test_gradient(masked, x)
    test_dtarget(masked, x, y)
end

# This is a bit of a hack, to remove one parameter
# from an ARD kernel to make it have the right number
# of parameters when masked
function _test_masked_ARD(kern::Kernel, x::Matrix, x2::Matrix, y::Vector)
    par = get_params(kern)[2:end]
    k_masked = typeof(kern)([par[1]], par[2:end]...)
    masked = Masked(k_masked, [1])
    test_cov(masked, x)
    test_cov_x1x2(masked, x, x2)
    test_grad_stack(masked, x)
    test_gradient(masked, x)
    test_dtarget(masked, x, y)
end

function test_masked(kern::StationaryARD, x::Matrix, x2::Matrix, y::Vector)
    _test_masked_ARD(kern, x, x2, y)
end
function test_masked(kern::LinArd, x::Matrix, x2::Matrix, y::Vector)
    _test_masked_ARD(kern, x, x2, y)
end

function test_Kernel(kern::Kernel, x::Matrix, x2::Matrix, y::Vector)
    t = typeof(kern)
    println("\tTesting $(t)...")
    @assert length(get_param_names(kern)) == length(get_params(kern)) == num_params(kern)
    test_cov(kern, x)
    test_cov_x1x2(kern, x, x2)
    test_grad_stack(kern, x)
    test_gradient(kern, x)
    test_dtarget(kern, x, y)
    test_masked(kern, x, x2, y)
end
    
d, n, n2 = 2, 10, 5
ll = rand(d)
x = randn(d, n)
x2 = randn(d, n2)
y = randn(n)

# Isotropic kernels

se = SEIso(1.0, 1.0)
test_Kernel(se, x, x2, y)

expk = ExpIso(1.0, 1.0)
test_Kernel(expk, x, x2, y)

mat12 = Mat12Iso(1.0,1.0)
test_Kernel(mat12, x, x2, y)

mat32 = Mat32Iso(1.0,1.0)
test_Kernel(mat32, x, x2, y)

mat52 = Mat52Iso(1.0,1.0)
test_Kernel(mat52, x, x2, y)

rq = RQIso(1.0, 1.0, 1.0)
test_Kernel(rq, x, x2, y)

peri = Periodic(1.0, 1.0, 2π)
test_Kernel(peri, x, x2, y)


# Non-isotropic

lin = Lin(1.0)
test_Kernel(lin, x, x2, y)

poly = Poly(0.0, 0.0, 2)
test_Kernel(poly, x, x2, y)

noise = Noise(1.0)
test_Kernel(noise, x, x2, y)

# Constant kernel

cons = Const(1.0)
test_Kernel(cons, x, x2, y)

# ARD kernels

se_ard = SEArd(ll, 1.0)
test_Kernel(se_ard, x, x2, y)

expk_ard = ExpArd(ll, 1.0)
test_Kernel(expk_ard, x, x2, y)

mat12_ard = Mat12Ard(ll, 1.0)
test_Kernel(mat12_ard, x, x2, y)

mat32_ard = Mat32Ard(ll, 1.0)
test_Kernel(mat32_ard, x, x2, y)

mat52_ard = Mat52Ard(ll, 1.0)
test_Kernel(mat52_ard, x, x2, y)

rq_ard=RQArd(ll, 0.0, 2.0)
test_Kernel(rq_ard, x, x2, y)

lin_ard = LinArd(ll)
test_Kernel(lin_ard, x, x2, y)

# Composite kernels

sum_kern = se + mat12
test_Kernel(sum_kern, x, x2, y)

sum_kern_3 = sum_kern + lin
test_Kernel(sum_kern_3, x, x2, y)

prod_kern = se * mat12
test_Kernel(prod_kern, x, x2, y)

prod_kern_3 = prod_kern * lin
test_Kernel(prod_kern_3, x, x2, y)


# Fixed Kernel

rq = RQIso(1.0, 1.0, 1.0)
test_Kernel(fix(rq, :lσ), x, x2, y)

test_Kernel(fix(rq), x, x2, y)

# Sum and Product and Fix
sum_prod_kern = se * mat12 + lin * fix(rq, :lσ)
test_Kernel(sum_prod_kern, x, x2, y)
