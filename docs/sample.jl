# Sample from the GP

using Gadfly, GaussianProcesses

srand(13579)

# Select mean and covariance function
mZero = MeanZero()
kern = SE(0.0,0.0)

# First sample from GP prior
gp = GP(m=mZero,k=kern)     

x_path = collect(linspace(-5,5));
prior=rand(gp,x_path, 10)

colors = [colorant"black", colorant"red", colorant"green",
          colorant"blue", colorant"yellow"]
layers = []
for i in 1:5
    push!(layers, layer(x=x_path,y=prior[:,i],Geom.line,
                        Theme(default_color=colors[i])))
end

p1 = plot(layers...)

# Training data
x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];

# Fit data to GP object
GaussianProcesses.fit!(gp, x, y)
post=rand(gp,x_path, 10)

layers = []
push!(layers, layer(x=x,y=y,Geom.point, Theme(default_color=colorant"black",default_point_size=4pt)))
for i in 1:5
    push!(layers, layer(x=x_path,y=post[:,i], Geom.line,
                        Theme(default_color=colors[i])))
end

p2 = plot(layers...)

# Write plots to PNGs
draw(PNG("prior_samples.png", 800px, 600px), p1)
draw(PNG("posterior_samples.png", 800px, 600px), p2)
