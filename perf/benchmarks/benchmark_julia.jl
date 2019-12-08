using GaussianProcesses
using GaussianProcesses: update_mll_and_dmll!, update_cK!, FullCovariancePrecompute
using BenchmarkTools
using DataFrames
using Random
using CSV
using LinearAlgebra

kerns = Dict(
    "se" => SEIso(0.3,0.3),
    "mat12" => Mat12Iso(0.3,0.3),
    "rq" => RQIso(0.3,0.3,0.3),
    "se+rq" => SEIso(0.3,0.3) + RQIso(0.3,0.3,0.3),
    "se+mat12" => SEIso(0.3,0.3) + Mat12Iso(0.3,0.3),
    "se*rq" => SEIso(0.3,0.3) * RQIso(0.3,0.3,0.3),
    "se*mat12" => SEIso(0.3,0.3) * Mat12Iso(0.3,0.3),
    "se+mat12+rq" => SEIso(0.3,0.3) + Mat12Iso(0.3,0.3) + RQIso(0.3,0.3,0.3),
    "(se+mat12)*rq" => (SEIso(0.3,0.3) + Mat12Iso(0.3,0.3)) * RQIso(0.3,0.3,0.3),
    # "(se+se2)*rq" => (SEIso(0.3,0.3) + SEIso(0.5,0.5)) * RQIso(0.3,0.3,0.3),
    "mask(se, [1])" => Masked(SEIso(0.3,0.3), [1]),
    "mask(se, [1])+mask(rq, [2:10])" =>  Masked(SEIso(0.3,0.3), [1]) +  Masked(RQIso(0.3,0.3,0.3), collect(2:10)),
    "fix(se, σ)" => fix(SEIso(0.3,0.3), :lσ)
    )
    
function benchmark_kernel(group, kern)
    XY_df = CSV.read("simdata.csv")
    Xt = XY_df[!,[:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :x10]]
    X = Matrix(transpose(Matrix(Xt)))
    Y = XY_df[!,:Y]
    @assert length(Y) == nt
    buffer = FullCovariancePrecompute(nt)
    gp = GP(X, Y, MeanConst(0.0), kern, 0.3)
    group["cK"] = @benchmarkable update_cK!($gp)
    group["mll_and_dmll"] = @benchmarkable update_mll_and_dmll!($gp, $buffer)
end

const d = 10        # Input observation dimension
const nt = 3000     # Number of training points
Random.seed!(1)
X = randn(d, nt) # Training data
Y = randn(nt)
XY_df = DataFrame(X')
XY_df[!,:Y] = Y
XY_df
CSV.write("simdata.csv", XY_df; writeheader=true)

SUITE = BenchmarkGroup()

for (label, k) in kerns
    SUITE[label] = BenchmarkGroup([label])
    benchmark_kernel(SUITE[label], k)
end
;
 
results = run(SUITE, verbose=false, seconds=10000, samples=20, evals=1)
;

knames = sort(collect(keys(kerns)))
times = [time(results[k]["mll_and_dmll"])/10^6 for k in knames]

df = DataFrame(kernel = knames, times=times)
print(df)

CSV.write("bench_results/GaussianProcesses_jl.csv", df; writeheader=false)
