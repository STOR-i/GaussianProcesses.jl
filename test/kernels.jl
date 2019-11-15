module TestKernels
using GaussianProcesses, Calculus
using Test, LinearAlgebra, Statistics, Random
using ForwardDiff
using GaussianProcesses: EmptyData, update_target_and_dtarget!, 
      cov_ij, dKij_dθp, dKij_dθ!,
      get_params, set_params!, num_params, StationaryARD, WeightedEuclidean
import Calculus: gradient

Random.seed!(1)
const d, n, n2 = 3, 6, 3
function testkernel(kern::Kernel)
    X = randn(d, n)
    X2 = randn(d, n2)
    y = randn(n)

    # Random columns
    i, j = rand(1:n), rand(1:n2)
    Xi = view(X, :, i)
    Xj = view(X, :, j) # works since n2 < n
    X2j = view(X2, :, j)

    # Preallocate some matrices
    cK = zeros(n, n)
    cK2 = zeros(n, n2)
    @test length(GaussianProcesses.get_param_names(kern)) ==
        length(get_params(kern)) ==
        num_params(kern)


    @testset "get and set params" begin
        kcopy = deepcopy(kern)
        params_1 = randn(num_params(kcopy))
        set_params!(kcopy, params_1)
        params_2 = get_params(kcopy)
        @test params_1 ≈ params_2
    end
    @testset "Variance" begin
        spec = cov(kern, X)
        @test spec ≈ invoke(cov, Tuple{Kernel, Matrix{Float64}}, kern, X)
        @test spec[i, j] ≈ cov(kern, Xi, Xj)

        key = GaussianProcesses.kernel_data_key(kern, X, X)
        @test typeof(key) == String
        # check we've overwritten the default if necessary
        kdata = GaussianProcesses.KernelData(kern, X, X)
        if typeof(kdata) != EmptyData
            @test key != "EmptyData"
        end
    end

    data = GaussianProcesses.KernelData(kern, X, X)
    data12 = GaussianProcesses.KernelData(kern, X, X2)

    @testset "Covariance" begin
        spec = cov(kern, X, X2)
        @test spec[i,j] ≈ cov(kern, Xi, X2j)
        spec = cov(kern, X, X2, data12)
        @test spec[i,j] ≈ cov(kern, Xi, X2j)
    end


    @testset "Gradient" begin
        nparams = num_params(kern)
        init_params = Vector(get_params(kern))
        dK = zeros(nparams)
        i, j = 3, 5
        dKij_dθ!(dK, kern, X, X, data, i, j, d, nparams)
        dK1 = copy(dK)
        dKij_dθ!(dK, kern, X, X, EmptyData(), i, j, d, nparams)
        dK2 = copy(dK)
        @test dK1 ≈ dK2
        for p in 1:nparams
            @test dK[p] ≈ dKij_dθp(kern, X, X, data,        i, j, p, d)
            @test dK[p] ≈ dKij_dθp(kern, X, X, EmptyData(), i, j, p, d)
            try
                dkp = dKij_dθp(kern, X, X, i, j, p, d)
                @test dkp == dK[p]
            catch
                # that's OK too
                continue
            end
        end
        if nparams > 0
            numer_grad = Calculus.gradient(init_params) do params
                set_params!(kern, params)
                t = cov_ij(kern, X, X, data, i, j, d)
                set_params!(kern, init_params)
                t
            end
            theor_grad = dK
            @test numer_grad ≈ theor_grad rtol=1e-3 atol=1e-3
        end
    end
    @testset "Gradient stack X1 ≠ X2" begin
        nparams = num_params(kern)
        init_params = Vector(get_params(kern))
        stack1 = Array{Float64}(undef, n, n2, nparams)
        stack2 = Array{Float64}(undef, n, n2, nparams)

        GaussianProcesses.grad_stack!(stack1, kern, X, X2, data12)

        theor_grad = vec(sum(stack1; dims=[1,2]))
        if nparams > 0
            numer_grad = Calculus.gradient(init_params) do params
                set_params!(kern, params)
                t = sum(cov(kern, X, X2, data12))
                set_params!(kern, init_params)
                t
            end
            @test theor_grad ≈ numer_grad rtol=1e-2 atol=1e-2
        end

        GaussianProcesses.grad_stack!(stack2, kern, X, X2, EmptyData())
        # invoke(GaussianProcesses.grad_stack!,
               # Tuple{AbstractArray, Kernel, Matrix{Float64}, Matrix{Float64},
                     # EmptyData},
               # stack2, kern, X, X2, EmptyData())
        @test stack1 ≈ stack2 rtol=1e-3 atol=1e-3
    end


    @testset "Gradient stack" begin
        nparams = num_params(kern)
        init_params = Vector(get_params(kern))
        stack1 = Array{Float64}(undef, n, n, nparams)
        stack2 = Array{Float64}(undef, n, n, nparams)

        GaussianProcesses.grad_stack!(stack1, kern, X, X, data)
        invoke(GaussianProcesses.grad_stack!,
               Tuple{AbstractArray, Kernel, Matrix{Float64}, Matrix{Float64},
                     EmptyData},
               stack2, kern, X, X, EmptyData())
        @test stack1 ≈ stack2

        theor_grad = vec(sum(stack1; dims=[1,2]))
        if nparams > 0
            numer_grad = Calculus.gradient(init_params) do params
                set_params!(kern, params)
                t = sum(cov(kern, X))
                set_params!(kern, init_params)
                t
            end
            @test theor_grad ≈ numer_grad rtol=1e-1 atol=1e-2
        end
    end

    @testset "dtarget" begin
        nparams = num_params(kern)
        gp = GPE(X, y, MeanConst(0.0), kern, -3.0)
        init_params = Vector(get_params(gp))
        update_target_and_dtarget!(gp)
        theor_grad = copy(gp.dtarget)
        if nparams > 0
            numer_grad = Calculus.gradient(init_params) do params
                set_params!(gp, params)
                update_target!(gp)
                t = gp.target
                set_params!(gp, init_params)
                t
            end
            @test theor_grad ≈ numer_grad rtol=1e-3 atol=1e-3
        end
    end

    @testset "EmptyData" begin
        gp = GPE(X, y, MeanConst(0.0), kern, -3.0)
        gp_empty = GPE(X, y, MeanConst(0.0), kern, -3.0, EmptyData())
        update_target_and_dtarget!(gp_empty)
        update_target_and_dtarget!(gp)
        @test gp.dmll ≈ gp_empty.dmll rtol=1e-6 atol=1e-6
    end
    
    @testset "predict gradient" begin
        gp = GPE(X, y, MeanConst(0.0), kern, -3.0)
        f = x -> sum(predict_y(gp, reshape(x, :, 1)))[1]
        z = rand(d)
        autodiff_grad = ForwardDiff.gradient(f, z)
        numer_grad = Calculus.gradient(f, z)
        @test autodiff_grad ≈ numer_grad rtol = 1e-3 atol = 1e-3
    end
end

@testset "kernel shortcuts" begin
    x1 = [0.1, 0.2]
    x2 = [1.1, 1.0]

    for pairs in [
                  (SEIso(1.0, 1.2), SE(1.0, 1.2)),
                  (SEArd([1.0, 1.5], 1.3), SE([1.0, 1.5], 1.3)),
                  (RQIso(1.0, 1.2, 0.5), RQ(1.0, 1.2, 0.5)),
                  (RQArd([1.0, 1.5], 1.3, 0.5), RQ([1.0, 1.5], 1.3, 0.5)),
                  (Matern(1/2, 1.0, 1.2), Mat12Iso(1.0, 1.2)),
                  (Matern(3/2, 1.0, 1.2), Mat32Iso(1.0, 1.2)),
                  (Matern(5/2, 1.0, 1.2), Mat52Iso(1.0, 1.2)),
                  (Matern(1/2, [1.0, 1.5], 1.2), Mat12Ard([1.0, 1.5], 1.2)),
                  (Matern(3/2, [1.0, 1.5], 1.2), Mat32Ard([1.0, 1.5], 1.2)),
                  (Matern(5/2, [1.0, 1.5], 1.2), Mat52Ard([1.0, 1.5], 1.2)),
                  (Lin(1.0), LinIso(1.0)),
                  (Lin([1.0, 1.5]), LinArd([1.0, 1.5])),
                 ]
        @test cov(pairs[1], x1, x2) == cov(pairs[2], x1, x2)
    end

end

@testset "Kernels" begin
    ll = rand(d)
    kernels = [# Isotropic kernels
               SEIso(1.0, 1.0), Mat12Iso(1.0,1.0), Mat32Iso(1.0,1.0), Mat52Iso(1.0,1.0),
               RQIso(1.0, 1.0, 1.0), Periodic(1.0, 1.0, 2π),
               # Non-isotropic
               Lin(1.0), Poly(0.0, 0.0, 2), Noise(1.0),
               # Constant kernel
               Const(1.0),
               # ARD kernels
               SEArd(ll, 1.0), Mat12Ard(ll, 1.0), Mat32Ard(ll, 1.0), Mat52Ard(ll, 1.0),
               RQArd(ll, 0.0, 2.0), LinArd(ll),
               # Composite kernels
               SEIso(1.0, 1.0) + Mat12Iso(1.0, 1.0),
               (SEIso(1.0, 1.0) + Mat12Iso(1.0, 1.0)) + Lin(1.0),
               SEIso(1.0, 1.0) * Mat12Iso(1.0, 1.0),
               (SEIso(1.0, 1.0) * Mat12Iso(1.0, 1.0)) * Lin(1.0),
               # Fixed Kernel
               fix(RQIso(1.0, 1.0, 1.0), :lσ), 
               fix(RQIso(1.0, 1.0, 1.0)),
               # Sum and Product and Fix
               SEIso(1.0, 1.0) * Mat12Iso(1.0, 1.0) +
               Lin(1.0) * fix(RQIso(1.0, 1.0, 1.0), :lσ)]
    @testset for kern in kernels
        println("\tTesting ", nameof(typeof(kern)), "...")
        testkernel(kern)
    end
    @testset "Masked" for kernel in kernels
        println("\tTesting masked", nameof(typeof(kernel)), "...")
        if isa(kernel, LinArd) || isa(kernel, GaussianProcesses.StationaryARD)
            par = get_params(kernel)
            k_masked = typeof(kernel).name.wrapper([par[1]], par[d+1:end]...)
            kern = Masked(k_masked, [1])
        else
            kern = Masked(kernel, [1])
        end
        println("\tTesting masked ", nameof(typeof(kern)), "...")
        testkernel(kern)
    end
    @testset "autodiff" for kernel in kernels[1:end-1]
        println("\tTesting autodiff ", nameof(typeof(kernel)), "...")
        testkernel(autodiff(kernel))
    end

end # testset
end # module
