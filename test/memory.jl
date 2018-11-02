module TestMemory
using Test, GaussianProcesses
using GaussianProcesses: EmptyData

@testset "Memory Allocation" begin
    @testset "simple" begin
        k = SEIso(0.0, 0.0)
        dim = 5
        nobs = 1000
        X = randn(dim, nobs)
        y = randn(nobs)
        logNoise = 0.1
        m = MeanZero()
        gp = GP(X, y, m, k, logNoise)
        mem = @allocated GP(X, y, m, k, logNoise)
        # there should be 1 allocation of an n×x matrix for the covariance
        # one for its Cholesky decomposition, and one for KernelData
        matrix_bytes = (nobs^2)*64/8
        @test mem/matrix_bytes < 3.1
        buf = Array{Float64}(undef, nobs, nobs)
        GaussianProcesses.update_mll_and_dmll!(gp, buf)
        mem = @allocated GaussianProcesses.update_mll_and_dmll!(gp, buf)
        @test mem/matrix_bytes < 0.1
    end

    @testset "sum kernel" begin
        k = SEIso(0.0, 0.0) + RQIso(1.0, 1.0, 1.0) + SEIso(1.0, 1.0)
        dim = 5
        nobs = 1000
        X = randn(dim, nobs)
        y = randn(nobs)
        logNoise = 0.1
        m = MeanZero()
        gp = GP(X, y, m, k, logNoise)
        mem = @allocated GP(X, y, m, k, logNoise)
        matrix_bytes = (nobs^2)*64/8
        @test mem/matrix_bytes < 3.1
        buf = Array{Float64}(undef, nobs, nobs)
        GaussianProcesses.update_mll_and_dmll!(gp, buf)
        mem = @allocated GaussianProcesses.update_mll_and_dmll!(gp, buf)
        @test mem/matrix_bytes < 0.1
    end

    @testset "prod kernel" begin
        k = Mat12Iso(0.0, 0.0) * SEIso(0.0, 0.0) + RQIso(1.0, 1.0, 1.0) * Mat32Iso(1.0, 1.0)
        dim = 5
        nobs = 1000
        X = randn(dim, nobs)
        y = randn(nobs)
        logNoise = 0.1
        m = MeanZero()
        gp = GP(X, y, m, k, logNoise)
        mem = @allocated GP(X, y, m, k, logNoise)
        matrix_bytes = (nobs^2)*64/8
        @test mem/matrix_bytes < 4.1
        buf = Array{Float64}(undef, nobs, nobs)
        GaussianProcesses.update_mll_and_dmll!(gp, buf)
        mem = @allocated GaussianProcesses.update_mll_and_dmll!(gp, buf)
        @test mem/matrix_bytes < 0.1
    end

    @testset "EmptyData" begin
        k = SEIso(0.0, 0.0) + RQIso(1.0, 1.0, 1.0)
        dim = 5
        nobs = 1000
        X = randn(dim, nobs)
        y = randn(nobs)
        logNoise = 0.1
        m = MeanZero()
        gp = GPE(X, y, m, k, logNoise, EmptyData())
        mem = @allocated GPE(X, y, m, k, logNoise, EmptyData())
        matrix_bytes = (nobs^2)*64/8
        @test mem/matrix_bytes < 2.1
        buf = Array{Float64}(undef, nobs, nobs)
        GaussianProcesses.update_mll_and_dmll!(gp, buf)
        mem = @allocated GaussianProcesses.update_mll_and_dmll!(gp, buf)
        @test mem/matrix_bytes < 0.1
    end

    @testset "EmptyData prod kernel" begin
        k = (SEIso(0.0, 0.0) + RQIso(1.0, 1.0, 1.0)) * SEIso(1.0, 1.0)
        dim = 5
        nobs = 1000
        X = randn(dim, nobs)
        y = randn(nobs)
        logNoise = 0.1
        m = MeanZero()
        gp = GPE(X, y, m, k, logNoise, EmptyData())
        mem = @allocated GPE(X, y, m, k, logNoise, EmptyData())
        matrix_bytes = (nobs^2)*64/8
        @test mem/matrix_bytes < 2.1
        buf = Array{Float64}(undef, nobs, nobs)
        GaussianProcesses.update_mll_and_dmll!(gp, buf)
        mem = @allocated GaussianProcesses.update_mll_and_dmll!(gp, buf)
        @test mem/matrix_bytes < 0.1
    end
end
end
