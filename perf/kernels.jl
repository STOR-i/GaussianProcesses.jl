using GaussianProcesses
using GaussianProcesses: KernelData, num_params, grad_stack!, cov!
using BenchmarkTools
using DataFrames
using JLD

srand(1)

const d = 10       # Input observation dimension
const nt = 100     # Number of training points
const λ = rand(d)

simple_kerns = Dict("isotropic"=> [SEIso(1.0, 1.0), Mat12Iso(1.0, 1.0),
                            Mat32Iso(1.0, 1.0), Mat52Iso(1.0, 1.0),
                            RQIso(1.0, 1.0, 1.0), Periodic(1.0, 1.0, 1.0)],
             "ARD"=> [SEArd(λ, 1.0), Mat12Ard(λ, 1.0), Mat32Ard(λ, 1.0),
                      Mat52Ard(λ, 1.0), RQArd(λ, 1.0, 1.0)],
             "non-stationary" => [LinIso(0.0), LinArd(λ)])

composite_kerns = Dict("isotropic + isotropic" =>
                       SEIso(1.0, 1.0) + RQIso(1.0, 1.0, 1.0),
                       "isotropic + non-stationary"=>
                       SEIso(1.0, 1.0) + LinIso(1.0),
                       "isotropic * isotropic" =>
                       Mat32Iso(1.0, 1.0) * Periodic(1.0, 1.0, 1.0),
                       "ARD * non-stationary" =>
                       RQArd(λ, 1.0, 1.0) * LinIso(1.0))
                       # "isotropic + Noise" =>
                       # SEIso(1.0, 1.0) + Noise(1.0))

function benchmark_kernel(group, kern)
    X = 2π * rand(d, nt) # Training data
    data = KernelData(kern, X)
    np = num_params(kern)
    cK = Array{Float64}( nt, nt)
    stack = Array{Float64}( nt, nt, np)
    group["data"] = @benchmarkable KernelData($(kern), $(X))
    group["cov"] =  @benchmarkable cov!($(cK), $(kern), $(X), $(data))
    group["grad"] = @benchmarkable grad_stack!($(stack), $(kern), $(X), $(data))
end

SUITE = BenchmarkGroup()
for (cls, kerns) in simple_kerns
    for k in kerns
        k_str = split(string(typeof(k)), '.')[2]
        SUITE[k_str] = BenchmarkGroup([cls])
        benchmark_kernel(SUITE[k_str], k)
    end
end

for (lab, k) in composite_kerns
    SUITE[lab] = BenchmarkGroup(["composite"])
    benchmark_kernel(SUITE[lab], k)
end

showall(SUITE)

const paramspath = joinpath(dirname(@__FILE__), "params.jld")
if !isfile(paramspath)
    println("Tuning benchmarks...")
    tune!(SUITE, verbose=true)
    JLD.save(paramspath, "SUITE", params(SUITE))
end
loadparams!(SUITE, JLD.load(paramspath, "SUITE"), :evals, :samples)

# Use @tagged to select more specific tests

# test_group = SUITE[@tagged "isotropic"]
# results = run(test_group, verbose = true, seconds = 1)
results = run(SUITE, verbose = true, seconds = 1)
showall(results)

names = @data(collect(keys(results)))
data_time = @data([time(results[k]["data"])/1000 for k in keys(results)])
cov_time = @data([time(results[k]["cov"])/1000 for k in keys(results)])
grad_time = @data([time(results[k]["grad"])/1000 for k in keys(results)])
df = DataFrame([names, data_time, cov_time, grad_time],
               [:kernel, :data, :cov, :gradient])
print(df)
