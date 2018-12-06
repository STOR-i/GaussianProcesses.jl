using Documenter, GaussianProcesses

makedocs(
    modules = [GaussianProcesses],
    format = :html,
    sitename = "GaussianProcesses.jl",
    pages = Any["Introduction" => "index.md",
                "Usage" => "usage.md",
                "Reference" => ["gp.md", "kernels.md", "mean.md","lik.md"]
               ]
)

deploydocs(
    repo = "github.com/STOR-i/GaussianProcesses.jl.git",
    target = "build",
    julia = "1.0",
    deps = nothing,
    make = nothing,
)
