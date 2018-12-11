module TestKernels
using GaussianProcesses, Calculus
using Test, LinearAlgebra, Statistics, Random
using ForwardDiff
using GaussianProcesses: EmptyData, update_target_and_dtarget!, 
      cov_ij, dKij_dθp, dKij_dθ!, 
      get_params, set_params!, StationaryARD, WeightedEuclidean
import Calculus: gradient

Random.seed!(1)
const d, n, n2 = 3, 10, 5
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
        length(GaussianProcesses.get_params(kern)) ==
        GaussianProcesses.num_params(kern)


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

    @testset "Covariance" begin
        spec = cov(kern, X, X2)
        @test spec[i,j] ≈ cov(kern, Xi, X2j)
    end

    data = GaussianProcesses.KernelData(kern, X, X)

    @testset "Gradient" begin
        nparams = GaussianProcesses.num_params(kern)
        init_params = Vector(GaussianProcesses.get_params(kern))
        dK = zeros(nparams)
        i, j = 3, 5
        dKij_dθ!(dK, kern, X, i, j, d, nparams)
        dK1 = copy(dK)
        dKij_dθ!(dK, kern, X, data, i, j, d, nparams)
        dK2 = copy(dK)
        dKij_dθ!(dK, kern, X, EmptyData(), i, j, d, nparams)
        dK3 = copy(dK)
        @test dK1 ≈ dK2
        @test dK1 ≈ dK3
        for p in 1:nparams
            @test dK[p] ≈ dKij_dθp(kern, X, i, j, p, d)
            @test dK[p] ≈ dKij_dθp(kern, X, data, i, j, p, d)
            @test dK[p] ≈ dKij_dθp(kern, X, EmptyData(), i, j, p, d)
        end
        # if nparams > 0
            # numer_grad = Calculus.gradient(init_params) do params
                # set_params!(kern, params)
                # t = cov_ij(kern, X, X, i, j, d)
                # set_params!(kern, init_params)
                # t
            # end
            # theor_grad = dK
            # @test numer_grad ≈ theor_grad rtol=1e-3 atol=1e-3
            # end
        # end
    end

    @testset "Gradient stack" begin
        nparams = GaussianProcesses.num_params(kern)
        init_params = Vector(GaussianProcesses.get_params(kern))
        stack1 = Array{Float64}(undef, n, n, nparams)
        stack2 = Array{Float64}(undef, n, n, nparams)

        GaussianProcesses.grad_stack!(stack1, kern, X, data)
        invoke(GaussianProcesses.grad_stack!,
               Tuple{AbstractArray, Kernel, Matrix{Float64},
                     EmptyData},
               stack2, kern, X, EmptyData())
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
        nparams = GaussianProcesses.num_params(kern)
        gp = GPE(X, y, MeanConst(0.0), kern, -3.0)
        init_params = Vector(GaussianProcesses.get_params(gp))
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
            par = GaussianProcesses.get_params(kernel)
            k_masked = typeof(kernel).name.wrapper([par[1]], par[d+1:end]...)
            kern = Masked(k_masked, [1])
        else
            kern = Masked(kernel, [1])
        end
        println("\tTesting masked ", nameof(typeof(kern)), "...")
        testkernel(kern)
    end
    @testset "autodiff" for kernel in kernels[1:end-1]
        if typeof(kernel) <: FixedKernel
            # TODO: autodiff for FixedKernel
            continue
        end
        if typeof(kernel) <: StationaryARD{WeightedEuclidean}
            # causes trouble
            continue
        end
        println("\tTesting autodiff ", nameof(typeof(kernel)), "...")
        testkernel(autodiff(kernel))
    end

end # testset
end # module
