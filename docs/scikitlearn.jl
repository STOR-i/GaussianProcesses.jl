using ScikitLearn
using ScikitLearn.GridSearch
using PyPlot
using GaussianProcesses: GP, MeanZero, SE

srand(42)
# Training data
n = 10
x = 2π * rand(n, 1)
y = sin(x[:, 1]) + 0.05*randn(n)


# Select mean and covariance function
mZero = MeanZero()                   # Zero mean function
kern = SE(0.0,0.0)                   # Squared exponential kernel with parameters
                                     # log(ℓ) = 0.0, log(σ) = 0.0
gp = fit!(GP(m=mZero,k=kern, logNoise=-1.0), x,y);

gp_cv = fit!(GridSearchCV(GP(m=mZero,k=SE(0.0,0.0)), Dict(:logNoise=>collect(-10:0.3:10), :k_lσ=>collect(0:0.1:5))), x, y);
best_gp = gp_cv.best_estimator_;
@show get_params(best_gp)[:logNoise] get_params(best_gp)[:k_lσ]


xx = -5:0.1:10
plot(xx, predict(gp, reshape(collect(xx), length(xx), 1)), label="hand-specified")
plot(xx, predict(best_gp, reshape(collect(xx), length(xx), 1)), label="gridsearch-optimized")
plot(x, y, "bo")
legend();

get_params(gp)

#Cross-validation

using ScikitLearn, PyPlot
using ScikitLearn.CrossValidation: cross_val_score

srand(2)

n_samples = 30
degrees = [1, 4, 15]

true_fun(X) = cos(1.5 * pi * X)
X = rand(n_samples)
y = true_fun(X) + randn(n_samples) * 0.1

gp = fit!(GP(logNoise=-10.0), X'', y)

X_test = linspace(0, 1, 100)
plot(X_test, predict(gp, X_test''), label="Model")
plot(X_test, true_fun(X_test), label="True function")
scatter(X, y, label="Samples")
xlabel("x")
ylabel("y")
xlim((0, 1))
ylim((-2, +2))
legend(loc="best");


mean(cross_val_score(gp, X'', y, cv=10))
