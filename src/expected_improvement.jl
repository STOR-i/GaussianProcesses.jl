#Expected improvement function

function EI(gp::GP,x::Matrix{Float64},maxY::Float64)
    
#Fit the GP
  mu, Sigma = predict(gp, x)

#Calculate useful terms
s = sqrt(max(diag(Sigma),0.0))
y = mu - maxY
ynorm = y./s

    
#Calculate the expected improvement
ei = y .* cdf(Normal(),ynorm) + s .* pdf(Normal(),ynorm)
ei = max(0,ei)
  return(ei)    
end
