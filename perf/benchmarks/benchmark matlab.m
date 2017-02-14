
run('gpml-matlab-v4.0-2016-10-19/startup.m')

nobsv=3000;
X = randn(nobsv,2);
Y = randn(nobsv);

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', 0);

gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, X, Y);

f = @() gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, X, Y);;
timeit(f)

covfunc = @covRQiso;
hyp = struct('mean', [], 'cov', [0 0 0], 'lik', 0);
f = @() gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, X, Y);;
timeit(f)

covfunc = { 'covSum', { 'covSEiso', 'covRQiso' } };
hyp = struct('mean', [], 'cov', [0 0 0 0 0], 'lik', 0);
f = @() gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, X, Y);;
timeit(f)

covfunc = { 'covProd', { 'covSEiso', 'covRQiso' } };
hyp = struct('mean', [], 'cov', [0 0 0 0 0], 'lik', 0);
f = @() gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, X, Y);;
timeit(f)

covfunc = { 'covSum', { 'covSEiso', 'covSEiso', 'covRQiso' } };
hyp = struct('mean', [], 'cov', [0 0 0 0 0 0 0], 'lik', 0);
f = @() gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, X, Y);;
timeit(f)

covfunc = {'covProd', {
                        { 'covSum', { 'covSEiso', 'covRQiso' } }, 
                        'covSEiso'}
                       };
hyp = struct('mean', [], 'cov', [0 0 0 0 0 0 0], 'lik', 0);
f = @() gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, X, Y);;
timeit(f)

covfunc = {@covMask, {1,@covSEiso}};
hyp = struct('mean', [], 'cov', [0 0], 'lik', 0);
f = @() gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, X, Y);;
timeit(f)

covfunc = { @covSum, { 
    {@covMask, {1, @covSEiso}}, 
    {@covMask, {2, @covRQiso}} 
        } 
    };
hyp = struct('mean', [], 'cov', [0 0 0 0 0], 'lik', 0);
f = @() gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, X, Y);;
timeit(f)
