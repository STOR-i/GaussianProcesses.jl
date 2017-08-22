type ProdMean <: CompositeMean
    means::Vector{Mean}
    ProdMean(args::Vararg{Mean}) = new(collect(args))
end

submeans(pm::ProdMean) = pm.means
mean(pd::ProdMean, x::MatF64) = prod(mean(m,x) for m in submeans(pm))
get_param_names(prodmean::ProdMean) = composite_param_names(prodmean.means, :pm)

function grad_mean(prodmean::ProdMean, x::Vector{Float64})
     dm = Array{Float64}( 0)
      for m in prodmean.means
          p = 1.0
          for j in prodkern.means[find(k.!=prodkern.means)]
              p = p.*mean(j, x)
          end
        append!(dm,grad_mean(m, x).*p)
      end
    dm
end

# Multiplication operators
*(m1::ProdMean, m2::Mean) = ProdMean(m1.means..., m2)
*(m1::ProdMean, m2::ProdMean) = ProdMean(m1.means..., m2.means...)
*(m1::Mean, m2::Mean) = ProdMean(m1,m2)
*(m1::Mean, m2::ProdMean) = ProdMean(m1, m2.means...)
