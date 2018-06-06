
using GaussianProcesses
using GaussianProcesses: update_mll_and_dmll!, update_cK!
using BenchmarkTools
using DataFrames
using JLD

const d = 10        # Input observation dimension
const nt = 3000     # Number of training points

kerns = Dict(
    "se" => SEIso(0.0,0.0),
    "mat12" => Mat12Iso(0.0,0.0),
    "rq" => RQIso(0.0,0.0,0.0),
    "se+rq" => SEIso(0.0,0.0) + RQIso(0.0,0.0,0.0),
    "se*rq" => SEIso(0.0,0.0) * RQIso(0.0,0.0,0.0),
    "se+se2+rq" => SEIso(0.0,0.0) + SEIso(0.5,0.5) + RQIso(0.0,0.0,0.0),
    "(se+se2)*rq" => (SEIso(0.0,0.0) + SEIso(0.5,0.5)) * RQIso(0.0,0.0,0.0),
    "mask(se, [1])" => Masked(SEIso(0.0,0.0), [1]),
    "mask(se, [1])+mask(rq, [2:10])" =>  Masked(SEIso(0.0,0.0), [1]) +  Masked(RQIso(0.0,0.0,0.0), collect(2:10)),
    "fix(se, σ)" => fix(SEIso(0.0,0.0), :lσ)
    )
    
function benchmark_kernel(group, kern)
    srand(1)
    X = randn(d, nt) # Training data
    Y = randn(nt)
    buf1=Array{Float64}(nt,nt)
    buf2=Array{Float64}(nt,nt)
    gp = GP(X, Y, MeanConst(0.0), kern, log(1.0))
    group["cK"] = @benchmarkable update_cK!($gp)
    group["mll_and_dmll"] = @benchmarkable update_mll_and_dmll!($gp, $buf1, $buf2)
end

SUITE = BenchmarkGroup()

for (label, k) in kerns
    SUITE[label] = BenchmarkGroup([label])
    benchmark_kernel(SUITE[label], k)
end
;

results = run(SUITE, verbose=false, seconds=1000, samples=10, evals=1)
;

knames = sort(collect(keys(kerns)))
times = [time(results[k]["mll_and_dmll"])/10^6 for k in knames]

df = DataFrame(kernel = knames, times=times)
print(df)

writetable("bench_results/GaussianProcesses_jl.csv", df, header=false)

srand(1)
X = randn(d, nt) # Training data
Y = randn(nt)
XY_df = DataFrame(X')
XY_df[:Y] = Y
XY_df
writetable("simdata.csv", XY_df, header=true)

buf1=Array{Float64}(nt,nt)
buf2=Array{Float64}(nt,nt)
gp = GPE(X, Y, MeanConst(0.0), kerns["se"], log(1.0))
update_mll_and_dmll!(gp, buf1, buf2)
gp.mll, gp.dmll # SE kernel

buf1=Array{Float64}(nt,nt)
buf2=Array{Float64}(nt,nt)
gp = GPE(X, Y, MeanConst(0.0), kerns["rq"], log(1.0))
Profile.clear()
update_mll_and_dmll!(gp, buf1, buf2)
@profile update_mll_and_dmll!(gp, buf1, buf2)
@profile update_mll_and_dmll!(gp, buf1, buf2)

Profile.print(mincount=10)
