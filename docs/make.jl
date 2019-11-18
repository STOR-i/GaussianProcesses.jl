using Documenter, GaussianProcesses

makedocs(
    modules = [GaussianProcesses],
    format = Documenter.HTML(),
    sitename = "GaussianProcesses.jl",
         pages = Any["Introduction" => "index.md",
                     "Basic usage" => ["usage.md", "plotting_gps.md"],
                     "Tutorials" => ["classification_example.md", "sparse_example.md", "mauna_loa.md", "poisson_regression.md"],
                     "Reference" => ["gp.md", "kernels.md", "mean.md","lik.md","sparse.md","crossvalidation.md"]
               ]
)

deploydocs(repo = "github.com/STOR-i/GaussianProcesses.jl.git")
