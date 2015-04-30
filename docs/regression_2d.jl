#Plot basic Gaussian Process

using Gadfly, GaP

#Training data
d, n = 2, 50

x = 2π * rand(d, n)
y = vec(sin(x[1,:]).*sin(x[2,:])) + 0.05*rand(n) 

mZero = MeanZero()
kern = Mat(5/2,[0.0,0.0],0.0)
gp = GP(x,y,mZero,kern,-2.0)
optimize!(gp)

plot(gp; clim=(-10.0, 10.0,-10.0,10.0))
     


