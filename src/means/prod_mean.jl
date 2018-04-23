type ProdMean <: CompositeMean
    means::Vector{Mean}
    ProdMean(args::Vararg{Mean}) = new(collect(args))
end

submeans(pm::ProdMean) = pm.means

mean(pm::ProdMean, x::VecF64) = prod(mean(m,x) for m in submeans(pm))
function mean(pm::ProdMean, X::MatF64)
    n = size(X, 2)
    p = ones(n)
    for m in submeans(pm)
        broadcast!(*, p, p, mean(m, X))
    end
    return p
end

get_param_names(pm::ProdMean) = composite_param_names(pm.means, :pm)

function grad_mean(pm::ProdMean, x::VecF64)
    np = num_params(pm)
    dm = Array{Float64}(np)
    means = submeans(pm)
    v = 1
    for (i,m) in enumerate(means)
        p = prod(mean(m2, x) for (j,m2) in enumerate(means) if i!=j)
        np_m = num_params(m)
        dm[v:v+np_m-1] = p * grad_mean(m, x)
        v+=np_m
    end
    return dm
end

# Multiplication operators
*(m1::ProdMean, m2::Mean) = ProdMean(m1.means..., m2)
*(m1::ProdMean, m2::ProdMean) = ProdMean(m1.means..., m2.means...)
*(m1::Mean, m2::Mean) = ProdMean(m1,m2)
*(m1::Mean, m2::ProdMean) = ProdMean(m1, m2.means...)
