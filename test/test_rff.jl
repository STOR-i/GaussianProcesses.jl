using Optim, GaussianProcesses, Distributions, Random, RDatasets, PyCall
Random.seed!(203617)

# function
datasets = pyimport("sklearn.datasets")
n = 100
d = 2
X, y = datasets.make_classification(n_samples = n, n_features = d, n_informative=2, n_redundant=0, random_state=123)
y = convert(Array, Bool.(y));
println("Proportion of positives: ", 100*sum(y)/length(y), "%.")

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
m = 30 # Number of features

gp = SSGPR(Xtr, ytr, kern, M=m)
ϕ = build_design_mat(gp.fourier, X)

# gpe = GP(X', y, mZero, kern)       #Fit the GP
# optimize!(gpe; method=ConjugateGradient())   # Optimise the hyperparameters
# contour(gpe)
