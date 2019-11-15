using Documenter, GaussianProcesses

makedocs(
    modules = [GaussianProcesses],
    format = Documenter.HTML(),
    sitename = "GaussianProcesses.jl",
    pages = Any["Introduction" => "index.md",
                "Basic usage" => ["usage.md", "sparse_example.md", "classification_example.md", "mauna_loa.md", "plotting_gps.md", "poisson_regression.md"],
                "Reference" => ["gp.md", "kernels.md", "mean.md","lik.md","sparse.md","crossvalidation.md"]
               ]
)

deploydocs(
    repo = "github.com/STOR-i/GaussianProcesses.jl.git",
    target = "build",
    julia = "1.0",
    deps = nothing,
    make = nothing,
)
