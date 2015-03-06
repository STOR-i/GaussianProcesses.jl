import Distributions.params

# See Chapter 4 Page 90 of Rasumussen and Williams Gaussian Processes for Machine Learning

abstract Kernel

include("se.jl")                # Squared exponential covariance function
include("mat32.jl")             # Matern 3/2 covariance function
include("mat52.jl")             # Matern 5/2 covariance function
include("exf.jl")               # Exponential covariance function
include("peri.jl")              # Periodic covariance function
include("poly.jl")              # Polnomial covariance function
include("rq.jl")                # Rational quadratic covariance function






