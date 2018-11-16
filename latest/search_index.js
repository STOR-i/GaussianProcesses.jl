var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#GaussianProcesses-1",
    "page": "Introduction",
    "title": "GaussianProcesses",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Introduction-1",
    "page": "Introduction",
    "title": "Introduction",
    "category": "section",
    "text": "Gaussian processes are a family of stochastic processes which provide a flexible nonparametric tool for modelling data. A Gaussian Process places a prior over functions, and can be described as an infinite dimensional generalisation of a multivariate Normal distribution. Moreover, the joint distribution of any finite collection of points is a multivariate Normal. This process can be fully characterised by its mean and covariance functions, where the mean of any point in the process is described by the mean function and the covariance between any two observations is specified by the kernel. Given a set of observed real-valued points over a space, the Gaussian Process is used to make inference on the values at the remaining points in the space.For an extensive review of Gaussian Processes there is an excellent book Gaussian Processes for Machine Learning by Rasmussen and Williams, (2006)"
},

{
    "location": "index.html#Installation-1",
    "page": "Introduction",
    "title": "Installation",
    "category": "section",
    "text": "GaussianProcesses.jl requires Julia version 0.5 or above. To install GaussianProcesses.jl run the following command inside the Julia package REPLpkg> add GaussianProcessesor in the standard REPLjulia> using Pkg\njulia> Pkg.add(\"GaussianProcesses\")"
},

{
    "location": "usage.html#",
    "page": "Usage",
    "title": "Usage",
    "category": "page",
    "text": ""
},

{
    "location": "usage.html#Usage-1",
    "page": "Usage",
    "title": "Usage",
    "category": "section",
    "text": ""
},

{
    "location": "gp.html#",
    "page": "Gaussian Processes",
    "title": "Gaussian Processes",
    "category": "page",
    "text": ""
},

{
    "location": "gp.html#GaussianProcesses.predict_f-Tuple{GPBase,AbstractArray{T,2} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.predict_f",
    "category": "method",
    "text": "predict_f(gp::GPBase, X::Matrix{Float64}[; full_cov::Bool = false])\n\nReturn posterior mean and variance of the Gaussian Process gp at specfic points which are given as columns of matrix X. If full_cov is true, the full covariance matrix is returned instead of only variances.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.GPE-Tuple{AbstractArray{T,2} where T,AbstractArray{T,1} where T,GaussianProcesses.Mean,Kernel,Float64,GaussianProcesses.KernelData,PDMats.AbstractPDMat}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.GPE",
    "category": "method",
    "text": "GPE(x, y, mean, kernel[, logNoise])\n\nFit a Gaussian process to a set of training points. The Gaussian process is defined in terms of its user-defined mean and covariance (kernel) functions. As a default it is assumed that the observations are noise free.\n\nArguments:\n\nx::AbstractVecOrMat{Float64}: Input observations\ny::AbstractVector{Float64}: Output observations\nmean::Mean: Mean function\nkernel::Kernel: Covariance function\nlogNoise::Float64: Natural logarithm of the standard deviation for the observation noise. The default is -2.0, which is equivalent to assuming no observation noise.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.GPE-Tuple{}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.GPE",
    "category": "method",
    "text": "GPE(; mean::Mean = MeanZero(), kernel::Kernel = SE(0.0, 0.0), logNoise::Float64 = -2.0)\n\nConstruct a GPE object without observations.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.GP",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.GP",
    "category": "function",
    "text": "GP(x, y, mean::Mean, kernel::Kernel[, logNoise::Float64=-2.0])\n\nFit a Gaussian process that is defined by its mean, its kernel, and the logarithm logNoise of the standard deviation of its observation noise to a set of training points x and y.\n\nSee also: GPE\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.predict_y-Tuple{GPE,AbstractArray{T,2} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.predict_y",
    "category": "method",
    "text": "predict_y(gp::GPE, x::Union{Vector{Float64},Matrix{Float64}}[; full_cov::Bool=false])\n\nReturn the predictive mean and variance of Gaussian Process gp at specfic points which are given as columns of matrix x. If full_cov is true, the full covariance matrix is returned instead of only variances.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.update_target!-Tuple{GPE}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.update_target!",
    "category": "method",
    "text": "update_target!(gp::GPE, ...)\n\nUpdate the log-posterior\n\nlog p(θ  y)  log p(y  θ) +  log p(θ)\n\nof a Gaussian process gp.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.GPMC-Tuple{AbstractArray{T,2} where T,AbstractArray{#s206,1} where #s206<:Real,GaussianProcesses.Mean,Kernel,Likelihood}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.GPMC",
    "category": "method",
    "text": "GPMC(x, y, mean, kernel, lik)\n\nFit a Gaussian process to a set of training points. The Gaussian process with non-Gaussian observations is defined in terms of its user-defined likelihood function, mean and covaiance (kernel) functions.\n\nThe non-Gaussian likelihood is handled by a Monte Carlo method. The latent function values are represented by centered (whitened) variables f(x) = m(x) + Lv where v  N(0 I) and LLᵀ = K_θ.\n\nArguments:\n\nx::AbstractVecOrMat{Float64}: Input observations\ny::AbstractVector{<:Real}: Output observations\nmean::Mean: Mean function\nkernel::Kernel: Covariance function\nlik::Likelihood: Likelihood function\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.GP-Tuple{Union{AbstractArray{Float64,1}, AbstractArray{Float64,2}},AbstractArray{#s206,1} where #s206<:Real,GaussianProcesses.Mean,Kernel,Likelihood}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.GP",
    "category": "method",
    "text": "GP(x, y, mean::Mean, kernel::Kernel, lik::Likelihood)\n\nFit a Gaussian process that is defined by its mean, its kernel, and its likelihood function lik to a set of training points x and y.\n\nSee also: GPMC\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.predict_y-Tuple{GPMC,AbstractArray{T,2} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.predict_y",
    "category": "method",
    "text": "predict_y(gp::GPMC, x::Union{Vector{Float64},Matrix{Float64}}[; full_cov::Bool=false])\n\nReturn the predictive mean and variance of Gaussian Process gp at specfic points which are given as columns of matrix x. If full_cov is true, the full covariance matrix is returned instead of only variances.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.update_target!-Tuple{GPMC}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.update_target!",
    "category": "method",
    "text": "update_target!(gp::GPMC, ...)\n\nUpdate the log-posterior\n\nlog p(θ v  y)  log p(y  v θ) + log p(v) + log p(θ)\n\nof a Gaussian process gp.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.mcmc-Tuple{GPBase}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.mcmc",
    "category": "method",
    "text": "mcmc(gp::GPBase; kwargs...)\n\nRun MCMC algorithms provided by the Klara package for estimating the hyperparameters of Gaussian process gp.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.make_posdef!-Tuple{AbstractArray{T,2} where T,AbstractArray{T,2} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.make_posdef!",
    "category": "method",
    "text": "make_posdef!(m::Matrix{Float64}, chol_factors::Matrix{Float64})\n\nTry to encode covariance matrix m as a positive definite matrix. The chol_factors matrix is recycled to store the cholesky decomposition, so as to reduce the number of memory allocations.\n\nSometimes covariance matrices of Gaussian processes are positive definite mathematically but have negative eigenvalues numerically. To resolve this issue, small weights are added to the diagonal (and hereby all eigenvalues are raised by that amount mathematically) until all eigenvalues are positive numerically.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.fit!-Union{Tuple{Y}, Tuple{X}, Tuple{GPE{X,Y,M,K,P,D} where D<:KernelData where P<:AbstractPDMat where K<:Kernel where M<:Mean,X,Y}} where Y where X",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.fit!",
    "category": "method",
    "text": "fit!(gp::GPE{X,Y}, x::X, y::Y)\n\nFit Gaussian process GPE to a training data set consisting of input observations x and output observations y.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.get_ααinvcKI!-Tuple{AbstractArray{T,2} where T,PDMats.AbstractPDMat,Array{T,1} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.get_ααinvcKI!",
    "category": "method",
    "text": "get_ααinvcKI!(ααinvcKI::Matrix{Float64}, cK::AbstractPDMat, α::Vector)\n\nWrite ααᵀ - cK⁻¹ to ααinvcKI avoiding any memory allocation, where cK and α are the covariance matrix and the alpha vector of a Gaussian process, respectively. Hereby α is defined as cK⁻¹ (Y - μ).\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.initialise_target!-Tuple{GPE}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.initialise_target!",
    "category": "method",
    "text": "initialise_target!(gp::GPE)\n\nInitialise the log-posterior\n\nlog p(θ  y)  log p(y  θ) +  log p(θ)\n\nof a Gaussian process gp.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.update_cK!-Tuple{GPE}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.update_cK!",
    "category": "method",
    "text": "update_cK!(gp::GPE)\n\nUpdate the covariance matrix and its Cholesky decomposition of Gaussian process gp.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.update_dmll!-Tuple{GPE,AbstractArray{T,2} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.update_dmll!",
    "category": "method",
    "text": " update_dmll!(gp::GPE, ...)\n\nUpdate the gradient of the marginal log-likelihood of Gaussian process gp.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.update_mll_and_dmll!-Tuple{GPE,AbstractArray{T,2} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.update_mll_and_dmll!",
    "category": "method",
    "text": "update_mll_and_dmll!(gp::GPE, ...)\n\nUpdate the gradient of the marginal log-likelihood of a Gaussian process gp.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.update_target_and_dtarget!-Tuple{GPE,AbstractArray{T,2} where T,AbstractArray{T,2} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.update_target_and_dtarget!",
    "category": "method",
    "text": "update_target_and_dtarget!(gp::GPE, ...)\n\nUpdate the log-posterior\n\nlog p(θ  y)  log p(y  θ) +  log p(θ)\n\nof a Gaussian process gp and its derivative.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.initialise_ll!-Tuple{GPMC}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.initialise_ll!",
    "category": "method",
    "text": "initialise_ll!(gp::GPMC)\n\nInitialise the log-likelihood of Gaussian process gp.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.initialise_target!-Tuple{GPMC}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.initialise_target!",
    "category": "method",
    "text": "initialise_target!(gp::GPMC)\n\nInitialise the log-posterior\n\nlog p(θ v  y)  log p(y  v θ) + log p(v) + log p(θ)\n\nof a Gaussian process gp.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.update_cK!-Tuple{GPMC}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.update_cK!",
    "category": "method",
    "text": "update_cK!(gp::GPMC)\n\nUpdate the covariance matrix and its Cholesky decomposition of Gaussian process gp.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.update_dll!-Tuple{GPMC,AbstractArray{T,2} where T,AbstractArray{T,2} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.update_dll!",
    "category": "method",
    "text": " update_dll!(gp::GPMC, ...)\n\nUpdate the gradient of the log-likelihood of Gaussian process gp.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.update_target_and_dtarget!-Tuple{GPMC,AbstractArray{T,2} where T,AbstractArray{T,2} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.update_target_and_dtarget!",
    "category": "method",
    "text": "update_target_and_dtarget!(gp::GPMC, ...)\n\nUpdate the log-posterior\n\nlog p(θ v  y)  log p(y  v θ) + log p(v) + log p(θ)\n\nof a Gaussian process gp and its derivative.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.composite_param_names-Tuple{Any,Any}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.composite_param_names",
    "category": "method",
    "text": "composite_param_names(objects, prefix)\n\nCall get_param_names on each element of objects and prefix the returned name of the element at index i with prefix * i * \'_\'.\n\nExamples\n\njulia> GaussianProcesses.get_param_names(ProdKernel(Mat12Iso(1/2, 1/2), SEArd([0.0, 1.0], 0.0)))\n5-element Array{Symbol,1}:\n :pk1_ll\n :pk1_lσ\n :pk2_ll_1\n :pk2_ll_2\n :pk2_lσ\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.map_column_pairs!-Tuple{AbstractArray{T,2} where T,Any,AbstractArray{T,2} where T,AbstractArray{T,2} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.map_column_pairs!",
    "category": "method",
    "text": "map_column_pairs!(D::Matrix{Float64}, f, X::Matrix{Float64}[, Y::Matrix{Float64} = X])\n\nLike map_column_pairs, but stores the result in D rather than a new matrix.\n\n\n\n\n\n"
},

{
    "location": "gp.html#GaussianProcesses.map_column_pairs-Tuple{Any,AbstractArray{T,2} where T,AbstractArray{T,2} where T}",
    "page": "Gaussian Processes",
    "title": "GaussianProcesses.map_column_pairs",
    "category": "method",
    "text": "map_column_pairs(f, X::Matrix{Float64}[, Y::Matrix{Float64} = X])\n\nCreate a matrix by applying function f to each pair of columns of input matrices X and Y.\n\n\n\n\n\n"
},

{
    "location": "gp.html#Gaussian-Processes-1",
    "page": "Gaussian Processes",
    "title": "Gaussian Processes",
    "category": "section",
    "text": "Modules = [GaussianProcesses]\nPages = [\"GP.jl\", \"GPE.jl\", \"GPMC.jl\", \"GPEelastic.jl\", \"common.jl\", \"mcmc.jl\", \"utils.jl\"]"
},

{
    "location": "kernels.html#",
    "page": "Kernels",
    "title": "Kernels",
    "category": "page",
    "text": ""
},

{
    "location": "kernels.html#GaussianProcesses.Const",
    "page": "Kernels",
    "title": "GaussianProcesses.Const",
    "category": "type",
    "text": "Const <: Kernel\n\nConstant kernel\n\nk(xx) = σ²\n\nwith signal standard deviation σ.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Const-Union{Tuple{T}, Tuple{T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.Const",
    "category": "method",
    "text": "Const(lσ::T)\n\nCreate Const with signal standard deviation exp(lσ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Lin-Tuple{Real}",
    "page": "Kernels",
    "title": "GaussianProcesses.Lin",
    "category": "method",
    "text": "Lin(ll::Union{Real,Vector{Real}})\n\nCreate linear kernel with length scale exp.(ll).\n\nSee also LinIso and LinArd.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.LinArd",
    "page": "Kernels",
    "title": "GaussianProcesses.LinArd",
    "category": "type",
    "text": "LinArd <: Kernel\n\nARD linear kernel (covariance)\n\nk(xx) = xᵀL²x\n\nwith length scale ℓ = (ℓ₁ ℓ₂ ) and L = diag(ℓ₁ ℓ₂ ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.LinArd-Union{Tuple{Array{T,1}}, Tuple{T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.LinArd",
    "category": "method",
    "text": "LinArd(ll::Vector{T})\n\nCreate LinArd with length scale exp.(ll).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.LinIso",
    "page": "Kernels",
    "title": "GaussianProcesses.LinIso",
    "category": "type",
    "text": "LinIso <: Kernel\n\nIsotropic linear kernel (covariance)\n\nk(x x) = xᵀxℓ²\n\nwith length scale ℓ.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.LinIso-Union{Tuple{T}, Tuple{T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.LinIso",
    "category": "method",
    "text": "LinIso(ll::T)\n\nCreate LinIso with length scale exp(ll).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Masked",
    "page": "Kernels",
    "title": "GaussianProcesses.Masked",
    "category": "type",
    "text": "Masked{K<:Kernel} <: Kernel\n\nA wrapper for kernels so that they are only applied along certain dimensions.\n\nThis is similar to the active_dims kernel attribute in the python GPy package and to the covMask function in the matlab gpml package.\n\nThe implementation is very simple: any function of the kernel that takes an X::Matrix input is delegated to the wrapped kernel along with a view of X that only includes the active dimensions.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Matern-Tuple{Real,Real,Real}",
    "page": "Kernels",
    "title": "GaussianProcesses.Matern",
    "category": "method",
    "text": "Matern(ν::Real, ll::Union{Real,Vector{Real}}, lσ::Real)\n\nCreate Matérn kernel of type ν (i.e. ν = 1/2, ν = 3/2, or ν = 5/2) with length scale exp.(ll) and signal standard deviation exp(σ).\n\nSee also Mat12Iso, Mat12Ard, Mat32Iso, Mat32Ard, Mat52Iso, and Mat52Ard.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat12Ard",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat12Ard",
    "category": "type",
    "text": "Mat12Ard <: MaternARD\n\nARD Matern 1/2 kernel (covariance)\n\nk(xx) = σ² exp(-x-xL)\n\nwith length scale ℓ = (ℓ₁ ℓ₂ ) and signal standard deviation σ where L = diag(ℓ₁ ℓ₂ ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat12Ard-Union{Tuple{T}, Tuple{Array{T,1},T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat12Ard",
    "category": "method",
    "text": "Mat12Ard(ll::Vector{T}, lσ::T)\n\nCreate Mat12Ard with length scale exp.(ll) and signal standard deviation exp(σ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat12Iso",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat12Iso",
    "category": "type",
    "text": "Mat12Iso <: MaternISO\n\nIsotropic Matern 1/2 kernel (covariance)\n\nk(xx) = σ^2 exp(-x-yℓ)\n\nwith length scale ℓ and signal standard deviation σ.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat12Iso-Union{Tuple{T}, Tuple{T,T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat12Iso",
    "category": "method",
    "text": "Mat12Iso(ll::T, lσ::T)\n\nCreate Mat12Iso with length scale exp(ll) and signal standard deviation exp(σ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat32Ard",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat32Ard",
    "category": "type",
    "text": "Mat32Ard <: MaternARD\n\nARD Matern 3/2 kernel (covariance)\n\nk(xx) = σ²(1 + 3x-xL)exp(- 3x-xL)\n\nwith length scale ℓ = (ℓ₁ ℓ₂ ) and signal standard deviation σ where L = diag(ℓ₁ ℓ₂ ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat32Ard-Union{Tuple{T}, Tuple{Array{T,1},T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat32Ard",
    "category": "method",
    "text": "Mat32Ard(ll::Vector{T}, lσ::T)\n\nCreate Mat32Ard with length scale exp.(ll) and signal standard deviation exp(lσ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat32Iso",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat32Iso",
    "category": "type",
    "text": "Mat32Iso <: MaternIso\n\nIsotropic Matern 3/2 kernel (covariance)\n\nk(xx) = σ²(1 + 3x-xℓ)exp(-3x-xℓ)\n\nwith length scale ℓ and signal standard deviation σ.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat32Iso-Union{Tuple{T}, Tuple{T,T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat32Iso",
    "category": "method",
    "text": "Mat32Iso(ll::T, lσ::T)\n\nCreate Mat32Iso with length scale exp(ll) and signal standard deviation exp(lσ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat52Ard",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat52Ard",
    "category": "type",
    "text": "Mat52Ard <: MaternARD\n\nARD Matern 5/2 kernel (covariance)\n\nk(xx) = σ²(1 + 5x-xL + 5x-x²(3L²))exp(- 5x-xL)\n\nwith length scale ℓ = (ℓ₁ ℓ₂ ) and signal standard deviation σ where L = diag(ℓ₁ ℓ₂ ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat52Ard-Union{Tuple{T}, Tuple{Array{T,1},T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat52Ard",
    "category": "method",
    "text": "Mat52Ard(ll::Vector{Real}, lσ::Real)\n\nCreate Mat52Ard with length scale exp.(ll) and signal standard deviation exp(lσ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat52Iso",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat52Iso",
    "category": "type",
    "text": "Mat52Iso <: MaternIso\n\nIsotropic Matern 5/2 kernel (covariance)\n\nk(xx) = σ²(1+5x-xℓ + 5x-x²(3ℓ²))exp(- 5x-xℓ)\n\nwith length scale ℓ and signal standard deviation σ.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Mat52Iso-Union{Tuple{T}, Tuple{T,T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.Mat52Iso",
    "category": "method",
    "text": "Mat52Iso(ll::Real, lσ::Real)\n\nCreate Mat52Iso with length scale exp(ll) and signal standard deviation exp(lσ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Noise",
    "page": "Kernels",
    "title": "GaussianProcesses.Noise",
    "category": "type",
    "text": "Noise <: Kernel\n\nNoise kernel (covariance)\n\nk(xx) = σ²δ(x-x)\n\nwhere δ is the Kronecker delta function and σ is the signal standard deviation.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Noise-Union{Tuple{T}, Tuple{T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.Noise",
    "category": "method",
    "text": "Noise(lσ::Real)\n\nCreate Noise with signal standard deviation exp(lσ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Periodic",
    "page": "Kernels",
    "title": "GaussianProcesses.Periodic",
    "category": "type",
    "text": "Periodic <: Isotropic{Euclidean}\n\nPeriodic kernel (covariance)\n\nk(xx) = σ²exp(-2sin²(πx-xp)ℓ²)\n\nwith length scale ℓ, signal standard deviation σ, and period p.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Periodic-Union{Tuple{T}, Tuple{T,T,T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.Periodic",
    "category": "method",
    "text": "Periodic(ll::Real, lσ::Real, lp::Real)\n\nCreate Periodic with length scale exp(ll), signal standard deviation exp(lσ), and period exp(lp).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Poly",
    "page": "Kernels",
    "title": "GaussianProcesses.Poly",
    "category": "type",
    "text": "Poly <: Kernel\n\nPolynomial kernel (covariance)\n\nk(xx) = σ²(xᵀx + c)ᵈ\n\nwith signal standard deviation σ, additive constant c, and degree d.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.Poly-Union{Tuple{T}, Tuple{T,T,Int64}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.Poly",
    "category": "method",
    "text": "Poly(lc::Real, lσ::Real, deg::Int)\n\nCreate Poly with signal standard deviation exp(lσ), additive constant exp(lc), and degree deg.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.RQ-Tuple{Real,Real,Real}",
    "page": "Kernels",
    "title": "GaussianProcesses.RQ",
    "category": "method",
    "text": "RQ(ll::Union{Real,Vector{Real}}, lσ::Real, lα::Real)\n\nCreate Rational Quadratic kernel with length scale exp.(ll), signal standard deviation exp(lσ), and shape parameter exp(lα).\n\nSee also RQIso and RQArd.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.RQArd",
    "page": "Kernels",
    "title": "GaussianProcesses.RQArd",
    "category": "type",
    "text": "RQArd <: StationaryARD{WeightedSqEuclidean}\n\nARD Rational Quadratic kernel (covariance)\n\nk(xx) = σ²(1 + (x - x)ᵀL²(x - x)(2α))^-α\n\nwith length scale ℓ = (ℓ₁ ℓ₂ ), signal standard deviation σ, and shape parameter α where L = diag(ℓ₁ ℓ₂ ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.RQArd-Union{Tuple{T}, Tuple{Array{T,1},T,T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.RQArd",
    "category": "method",
    "text": "RQArd(ll::Vector{Real}, lσ::Real, lα::Real)\n\nCreate RQArd with length scale exp.(ll), signal standard deviation exp(lσ), and shape parameter exp(lα).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.RQIso",
    "page": "Kernels",
    "title": "GaussianProcesses.RQIso",
    "category": "type",
    "text": "RQIso <: Isotropic{SqEuclidean}\n\nIsotropic Rational Quadratic kernel (covariance)\n\nk(xx) = σ²(1 + (x - x)ᵀ(x - x)(2αℓ²))^-α\n\nwith length scale ℓ, signal standard deviation σ, and shape parameter α.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.RQIso-Union{Tuple{T}, Tuple{T,T,T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.RQIso",
    "category": "method",
    "text": "RQIso(ll:T, lσ::T, lα::T)\n\nCreate RQIso with length scale exp(ll), signal standard deviation exp(lσ), and shape parameter exp(lα).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.SE-Tuple{Real,Real}",
    "page": "Kernels",
    "title": "GaussianProcesses.SE",
    "category": "method",
    "text": "SE(ll::Union{Real,Vector{Real}}, lσ::Real)\n\nCreate squared exponential kernel with length scale exp.(ll) and signal standard deviation exp(lσ).\n\nSee also SEIso and SEArd.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.SEArd",
    "page": "Kernels",
    "title": "GaussianProcesses.SEArd",
    "category": "type",
    "text": "SEArd <: StationaryARD{WeightedSqEuclidean}\n\nARD Squared Exponential kernel (covariance)\n\nk(xx) = σ²exp(- (x - x)ᵀL²(x - y)2)\n\nwith length scale ℓ = (ℓ₁ ℓ₂ ) and signal standard deviation σ where L = diag(ℓ₁ ℓ₂ ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.SEArd-Union{Tuple{T}, Tuple{Array{T,1},T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.SEArd",
    "category": "method",
    "text": "SEArd(ll::Vector{Real}, lσ::Real)\n\nCreate SEArd with length scale exp.(ll) and signal standard deviation exp(lσ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.SEIso",
    "page": "Kernels",
    "title": "GaussianProcesses.SEIso",
    "category": "type",
    "text": "SEIso <: Isotropic{SqEuclidean}\n\nIsotropic Squared Exponential kernel (covariance)\n\nk(xx) = σ²exp(- (x - x)ᵀ(x - x)(2ℓ²))\n\nwith length scale ℓ and signal standard deviation σ.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.SEIso-Union{Tuple{T}, Tuple{T,T}} where T",
    "page": "Kernels",
    "title": "GaussianProcesses.SEIso",
    "category": "method",
    "text": "SEIso(ll::T, lσ::T)\n\nCreate SEIso with length scale exp(ll) and signal standard deviation exp(lσ).\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.EmptyData",
    "page": "Kernels",
    "title": "GaussianProcesses.EmptyData",
    "category": "type",
    "text": "EmptyData <: KernelData\n\nDefault empty KernelData.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.KernelData",
    "page": "Kernels",
    "title": "GaussianProcesses.KernelData",
    "category": "type",
    "text": "KernelData\n\nData to be used with a kernel object to calculate a covariance matrix, which is independent of kernel hyperparameters.\n\nSee also EmptyData.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#GaussianProcesses.cov!",
    "page": "Kernels",
    "title": "GaussianProcesses.cov!",
    "category": "function",
    "text": "cov!(cK::AbstractMatrix, k::Kernel, X₁::AbstractMatrix, X₂::AbstractMatrix)\n\nLike cov(k, X₁, X₂), but stores the result in cK rather than a new matrix.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#Statistics.cov",
    "page": "Kernels",
    "title": "Statistics.cov",
    "category": "function",
    "text": "cov(k::Kernel, X₁::AbstractMatrix, X₂::AbstractMatrix)\n\nCreate covariance matrix from kernel k and matrices of observations X₁ and X₂, where each column is an observation.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#Statistics.cov-Tuple{Kernel,AbstractArray{T,2} where T,GaussianProcesses.EmptyData}",
    "page": "Kernels",
    "title": "Statistics.cov",
    "category": "method",
    "text": "cov(k::Kernel, X::AbstractMatrix[, data::KernelData = KernelData(k, X, X)])\n\nCreate covariance function from kernel k, matrix of observations X, where each column is an observation, and kernel data data constructed from input observations.\n\n\n\n\n\n"
},

{
    "location": "kernels.html#Kernels-1",
    "page": "Kernels",
    "title": "Kernels",
    "category": "section",
    "text": "Modules = [GaussianProcesses]\nPages = readdir(joinpath(\"..\", \"src\", \"kernels\"))"
},

{
    "location": "mean.html#",
    "page": "Means",
    "title": "Means",
    "category": "page",
    "text": ""
},

{
    "location": "mean.html#GaussianProcesses.MeanConst",
    "page": "Means",
    "title": "GaussianProcesses.MeanConst",
    "category": "type",
    "text": "MeanConst <: Mean\n\nConstant mean function\n\nm(x) = β\n\nwith constant β.\n\n\n\n\n\n"
},

{
    "location": "mean.html#GaussianProcesses.MeanLin",
    "page": "Means",
    "title": "GaussianProcesses.MeanLin",
    "category": "type",
    "text": "MeanLin <: Mean\n\nLinear mean function\n\nm(x) = xᵀβ\n\nwith linear coefficients β.\n\n\n\n\n\n"
},

{
    "location": "mean.html#GaussianProcesses.MeanPoly",
    "page": "Means",
    "title": "GaussianProcesses.MeanPoly",
    "category": "type",
    "text": "MeanPoly <: Mean\n\nPolynomial mean function\n\nm(x) = ᵢⱼ βᵢⱼxᵢʲ\n\nwith polynomial coefficients βᵢⱼ of shape d  D where d is the dimension of observations and D is the degree of the polynomial.\n\n\n\n\n\n"
},

{
    "location": "mean.html#GaussianProcesses.MeanZero",
    "page": "Means",
    "title": "GaussianProcesses.MeanZero",
    "category": "type",
    "text": "MeanZero <: Mean\n\nZero mean function\n\nm(x) = 0\n\n\n\n\n\n"
},

{
    "location": "mean.html#Means-1",
    "page": "Means",
    "title": "Means",
    "category": "section",
    "text": "Modules = [GaussianProcesses]\nPages = readdir(joinpath(\"..\", \"src\", \"means\"))"
},

]}
