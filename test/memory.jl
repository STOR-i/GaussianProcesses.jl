using Test, GaussianProcesses

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
    @test mem < 3.9*nobs^2*64
    GaussianProcesses.update_target_and_dtarget!(gp)
    mem = @allocated GaussianProcesses.update_target_and_dtarget!(gp)
    @test mem < 0.5*nobs^2*64

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
    @test mem < 3.9*nobs^2*64
    mem = @allocated GaussianProcesses.update_target_and_dtarget!(gp)
    @test mem < 0.5*nobs^2*64
end
