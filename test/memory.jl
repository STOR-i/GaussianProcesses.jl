using Test, GaussianProcesses
using GaussianProcesses: EmptyData

@testset "simple memory allocations" begin
    k = SEIso(0.0, 0.0)
    dim = 5
    nobs = 5000
    X = randn(dim, nobs)
    y = randn(nobs)
    logNoise = 0.1
    m = MeanZero()
    gp = GP(X, y, m, k, logNoise)
    mem = @allocated GP(X, y, m, k, logNoise)
    # there should be 1 allocation of an n×x matrix for the covariance
    # one for its Cholesky decomposition, and one for KernelData
    matrix_bytes = (nobs^2)*64/8
    @test mem/matrix_bytes < 3.5
    GaussianProcesses.update_target_and_dtarget!(gp)
    mem = @allocated GaussianProcesses.update_target_and_dtarget!(gp)
    @test mem/matrix_bytes < 0.5
end

@testset "sum kernel memory allocation" begin
    k = SEIso(0.0, 0.0) + RQIso(1.0, 1.0, 1.0)
    dim = 5
    nobs = 5000
    X = randn(dim, nobs)
    y = randn(nobs)
    logNoise = 0.1
    m = MeanZero()
    gp = GP(X, y, m, k, logNoise)
    mem = @allocated GP(X, y, m, k, logNoise)
    matrix_bytes = (nobs^2)*64/8
    @test mem/matrix_bytes < 3.5
    mem = @allocated GaussianProcesses.update_target_and_dtarget!(gp)
    @test mem/matrix_bytes < 0.5
end

@testset "memory alloc with EmptyData" begin
    k = SEIso(0.0, 0.0) + RQIso(1.0, 1.0, 1.0)
    dim = 5
    nobs = 5000
    X = randn(dim, nobs)
    y = randn(nobs)
    logNoise = 0.1
    m = MeanZero()
    gp = GPE(X, y, m, k, logNoise, EmptyData())
    mem = @allocated GPE(X, y, m, k, logNoise)
    matrix_bytes = (nobs^2)*64/8
    @test mem/matrix_bytes < 2.5
    mem = @allocated GaussianProcesses.update_target_and_dtarget!(gp)
    @test mem/matrix_bytes < 0.5
end
