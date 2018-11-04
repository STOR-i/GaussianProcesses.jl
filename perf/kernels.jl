using GaussianProcesses, BenchmarkTools
using Random, Printf

Random.seed!(1)

const d = 10       # Input observation dimension
const nt = 100     # Number of training points
const λ = rand(d)
const X = 2π * rand(d, nt) # Training data
const cK = Array{Float64}(undef, nt, nt)

const kernelswithtags = Dict((k => ["simple", "isotropic"] for k in
                                (SEIso(1.0, 1.0), Mat12Iso(1.0, 1.0), Mat32Iso(1.0, 1.0),
                                 Mat52Iso(1.0, 1.0), RQIso(1.0, 1.0, 1.0),
                                 Periodic(1.0, 1.0, 1.0)))...,
                             (k => ["simple", "ARD"] for k in
                                (SEArd(λ, 1.0), Mat12Ard(λ, 1.0), Mat32Ard(λ, 1.0),
                                 Mat52Ard(λ, 1.0), RQArd(λ, 1.0, 1.0)))...,
                             (k => ["simple", "non-stationary"] for k in
                                (LinIso(0.0), LinArd(λ)))...,
                             SEIso(1.0, 1.0) + RQIso(1.0, 1.0, 1.0) =>
                                ["composite", "isotropic + isotropic"],
                             SEIso(1.0, 1.0) + LinIso(1.0) =>
                                ["composite", "isotropic + non-stationary"],
                             Mat32Iso(1.0, 1.0) * Periodic(1.0, 1.0, 1.0) =>
                                ["composite", "isotropic * isotropic"],
                             RQArd(λ, 1.0, 1.0) * LinIso(1.0) =>
                                ["composite", "ARD * non-stationary"])
                       # SEIso(1.0, 1.0) + Noise(1.0) => ["composite", "isotropic + Noise"])

# Define benchmarks
const suite = BenchmarkGroup(["kernels"])

for (kernel, tags) in kernelswithtags
    data = GaussianProcesses.KernelData(kernel, X, X)
    np = GaussianProcesses.num_params(kernel)
    stack = Array{Float64}(undef, nt, nt, np)
    group = suite[string(typeof(kernel))] = BenchmarkGroup(tags)
    group["data"] = @benchmarkable GaussianProcesses.KernelData($kernel, $X, $X)
    group["cov"] =  @benchmarkable GaussianProcesses.cov!($cK, $kernel, $X, $data)
    group["grad"] = @benchmarkable GaussianProcesses.grad_stack!($stack, $kernel, $X, $data)
end

println("Benchmark suite:")
# TODO: Workaround until https://github.com/JuliaCI/BenchmarkTools.jl/issues/96 is fixed
show(stdout, MIME"text/plain"(), suite; verbose = true, limit = max(3, length(suite)))
println()

# If a cache of tuned parameters already exists, use it, otherwise, tune and cache
# the benchmark parameters. Reusing cached parameters is faster and more reliable
# than re-tuning `suite` every time the file is included.
const paramspath = joinpath(dirname(@__FILE__), "params.json")

if isfile(paramspath)
    loadparams!(suite, BenchmarkTools.load(paramspath)[1], :evals, :samples)
else
    println("\nTuning benchmarks...")
    tune!(suite, verbose=true)
    BenchmarkTools.save(paramspath, params(suite))
    println()
end

println("\nRunning benchmarks...")
# Use @tagged to select more specific tests:
#const test_group = suite[@tagged "non-stationary"]
#const results = run(test_group, verbose = true, seconds = 1)
const results = run(suite, verbose = true, seconds = 1)

# Save results
println("\nSaving results...")
BenchmarkTools.save(joinpath(dirname(@__FILE__), "results.json"), results)

# Print table of results
function printpadright(text, total)
    print(text)
    for _ in (length(text)+1):total
        print(" ")
    end
end

function printresults(results)
    # Compute max number of chars in the first column with kernel names
    maxchar = max(6, maximum(length(k) for k in keys(results)))

    # Print header
    printpadright("kernel", maxchar)
    println(" | data [μs] | cov [μs]  | grad [μs]")
    foreach(x -> print("-"), 1:maxchar)
    println("-+-----------+-----------+----------")

    # Print results for every kernel
    foreach(results) do (key, result)
        printpadright(key, maxchar)
        @printf(" | %9.4f | %9.4f | %9.4f\n", time(result["data"]) / 1000,
                time(result["cov"]) / 1000, time(result["grad"]) / 1000)
    end
end

println("\nResults:")
printresults(results)
