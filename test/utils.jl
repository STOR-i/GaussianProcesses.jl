using GaussianProcesses, Test

@testset "Utils" begin
    d, n = 5, 10
    logℓ, logσ = rand(d), rand()
    X = rand(d,n)
    kern = SEArd(logℓ, logσ)
    data = GaussianProcesses.KernelData(kern, X)

    @test GaussianProcesses.distance(kern, X, data) ≈ GaussianProcesses.distance(kern, X)
end

