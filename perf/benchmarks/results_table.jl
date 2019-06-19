using CSV
using DataFrames
using Markdown
using Printf

julia_results = CSV.read("bench_results/GaussianProcesses_jl.csv", header=["label", "GaussianProcesses.jl"])
GPy_results = CSV.read("bench_results/GPy.csv", header=["label", "GPy"])
gpml_results = CSV.read("bench_results/gpml.csv", header=["label", "gpml"])

comparison = join(julia_results, GPy_results, kind=:outer, on=:label, makeunique=false)
comparison = join(comparison, gpml_results,   kind=:outer, on=:label, makeunique=false)
sort!(comparison, Symbol("GaussianProcesses.jl"))

CSV.write("bench_results/compare.csv", comparison);
comparison

""" Return the column index of the fastest execution for row i """
function rowindmin(i)
    row = [comparison[i, col] for col in 2:4]
    row_nomiss = [(ismissing(x) ? Inf : x) for x in row]
    imin = argmin(row_nomiss)
    return imin +1
end

imin = rowindmin.(1:nrow(comparison))
print(imin)

fmt(::Missing) = ""
fmt(x::Int64) = @sprintf("%d", x)
fmt(x::Float64) = @sprintf("%.0f", x)
fmt(x) = String(x)

rows = [String.(names(comparison))] # start with just header

for irow in 1:nrow(comparison)
    row = [fmt(comparison[irow, col]) for col in names(comparison)]
    row[imin[irow]] = @sprintf("**%s**", row[imin[irow]])
    push!(rows, row)
end

t=Markdown.Table(rows, [:r, :r, :r, :r, :r])

b = IOBuffer()
Markdown.plain(b, t)
print(String(take!(b)))

fmt(::Missing) = ""
fmt(x::Int64) = @sprintf("%d", x)
fmt(x::Float64) = @sprintf("%.0f", x)
fmt(x) = String(x)

rows = [String.(names(comparison))] # start with just header

for irow in 1:nrow(comparison)
    row = [fmt(comparison[irow, col]) for col in names(comparison)]
    row[imin[irow]] = @sprintf("**%s**", row[imin[irow]])
    push!(rows, row)
end

t=Markdown.Table(rows, [:r, :r, :r, :r, :r])

b = IOBuffer()
Markdown.plain(b, t)
print(String(take!(b)))
