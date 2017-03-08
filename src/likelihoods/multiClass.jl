"""
    # Description
    Constructor for the Multiclass likelihood for classification, where the link function is the Softmax function

    # Arguments:
    * `nClass::Int64`: number of classes
    """
type MultiClassLik <: Likelihood
    nClass::Int64                #number of classes
    MultiClassLik(nClass::Int64) = new(nClass)
end

# function log_dens(multiclass::MultiClassLik, f::Vector{Float64}, y::Vector{Int64})
#     return 
# end

# function dlog_dens_df(multiclass::MultiClassLik, f::Vector{Float64}, y::Vector{Int64})
#     return 
# end                   
 
#mean and variance under the likelihood
# mean_lik(multiclass::MultiClassLik, f::Vector{Float64}) = Float64[Φ(fi) for fi in f]
# var_lik(multiclass::MultiClassLik, f::Vector{Float64}) = Float64[Φ(fi)*(1-Φ(fi)) for fi in f]

# get_params(multiclass::MultiClassLik) = []
# num_params(multiclass::MultiClassLik) = 0
