using Statistics
using CairoMakie

S0 = 8e7
# 95% confidence intervals from Zitzmann et al. 2024 (supp Table A)
param_cis = Dict(:β => [3.63e-7, 5.13e-7] * S0, :δ => [1.20, 1.35], :πv => [2.76, 3.41])
param_means = Dict(:β => 4.27e-7, :δ => 1.28, :πv => 3.07)
# These assume symmetric 95% CIs meaning the standard deviation
param_sigma = Dict(k => (v[2] - v[1]) / (2 * 1.96) for (k, v) in param_cis)

using DataFrames, CSV
df = DataFrame(param_sigma)
CSV.write("data/param_sigma.csv", df)
