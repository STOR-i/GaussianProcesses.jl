using Optim, Distributions, Random, RDatasets, PyCall
using Revise
using GaussianProcesses
Random.seed!(203617)

datasets = pyimport("sklearn.datasets")
n = 100
d = 2
X, y = datasets.make_regression(n_samples = n, n_features = d, random_state=123)

function test_ssgp(X, y)
    n = 100
    d = 2
    # Partition into test/train
    train_prop = 0.7
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, train_prop*n))
    test_idx = view(idx, (floor(Int, train_prop*n)+1):n)
    Xtr = X[train_idx, :]
    Xte = X[test_idx, :]
    ytr = y[train_idx]
    yte = y[test_idx]

    mZero = MeanZero();                # Zero mean function
    kern = SE(zeros(2), 0.0);   # Matern 3/2 ARD kernel (note that hyperparameters are on the log scale)
    m = 20 # Number of features
    F = RFF(m, Xtr)
    gp = SSGP(Xtr, ytr, F, mZero, kern)
    fit!(gp, Xtr, ytr)
    μ, Σ = predict_y(gp, Xte)
    return μ, Σ
end

# @run test_ssgp(X, y)

# μ, Σ = test_ssgp(X, y)


n = 100
d = 2
# Partition into test/train
train_prop = 0.7
idx = shuffle(1:n)
train_idx = view(idx, 1:floor(Int, train_prop*n))
test_idx = view(idx, (floor(Int, train_prop*n)+1):n)
Xtr = X[train_idx, :]
Xte = X[test_idx, :]
ytr = y[train_idx]
yte = y[test_idx]

mZero = MeanZero();                # Zero mean function
kern = SE(zeros(d), 0.0);   # Matern 3/2 ARD kernel (note that hyperparameters are on the log scale)
m = 20 # Number of features
F = RFF(m, Xtr)
gp = GP(Xtr', ytr, F, mZero, kern)
fit!(gp, Xtr, ytr)
μ, Σ = predict_y(gp, Xte)
