# Benchmarks

This contains a set of benchmarks comparing different packages for fitting Gaussian processes.
All benchmarks use the same simulated dataset with 10 covariates and 3000 observations, 
and compute the marginal likelihood and its derivative under various covariance functions (kernels).
We allow any pre-computation that does not depend on the kernel parameters; for example `GaussianProcesses.jl` precomputes the distance matrix for isotropic kernels.
For each kernel, we run the benchmark function 20 times, and report the minimum time.

## Running benchmarks

For julia `GaussianProcesses.jl`:

```sh
cd benchmarks
julia benchmark_julia.jl
```

For python `GPy`:

```sh
cd benchmarks
python benchmark_GPy.py
```

For matlab `gpml`, first download `gpml` from [gaussianprocesses.org](http://www.gaussianprocess.org/gpml/code/matlab/doc/) into the benchmarks directory. The first line of the `benchmark_gpml.m` script loads the `gpml` library from `gpml-matlab-v4.2-2018-06-11/startup.m`, change this path if you are using a different version.

```sh
cd benchmarks
matlab -nodisplay -nojvm -nosplash -nodesktop -r "try, run('benchmark_gpml.m'), catch, exit(1), end, exit(0);"
```

## Results

All results are in milliseconds, and report the best of 20 trials.
For each kernel, the fastest time is highlighted in bold.

|                          label | GaussianProcesses.jl |  GPy |     gpml |
| ------------------------------:| --------------------:| ----:| --------:|
|                     fix(se, σ) |              **670** | 1160 |          |
|                  mask(se, [1]) |              **773** | 1153 |      826 |
|                             se |              **781** | 1111 |      913 |
|                          mat12 |              **807** | 1154 |      988 |
|                             rq |                 1164 | 1556 |  **905** |
|                          se+rq |             **1311** | 1632 |     4145 |
| mask(se, [1])+mask(rq, [2:10]) |                 1360 | 1621 | **1269** |
|                          se*rq |             **1484** | 1651 |     4534 |
|                      se+se2+rq |             **1540** | 1682 |     5177 |
|                    (se+se2)*rq |                 1803 | 1784 | **1594** |

## Additional comparisons

We are open to adding other Gaussian process packages to this comparison.
Please submit a pull request with some benchmark code against other packages.
