using Base.Test, GaussianProcesses
using GaussianProcesses: distance, KernelData, StationaryARDData, SEArd

srand(1)

d, n = 5, 10
logℓ, logσ = rand(d), rand()
X = rand(d,n)
kern = SEArd(logℓ, logσ)
data = KernelData(kern, X)

distance(kern, data)
distance(kern, X)

@test_approx_eq distance(kern, data) distance(kern, X)
