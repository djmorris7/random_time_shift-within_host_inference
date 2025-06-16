"""
This file stores the priors. This includes the hyper priors, the parameter priors, and the fixed parameters.
Priors are parameterised as functions, everything else is a known dist.
"""

using CSV
using DataFrames
using Distributions

priors = Dict(
    :z_R₀ => Normal(0, 1),
    :z_δ => Normal(0, 1),
    :z_πv => Normal(0, 1),
    :infection_time => Gumbel(-7, 3)
)

const HalfNormal = Truncated(Normal(0, 1), 0, Inf)
const BigHalfNormal = Truncated(Normal(0, 3), 0, Inf)
# Truncated(Normal(0, 3), 0, Inf)

hyper_priors = Dict(
    :μ_R₀ => Gamma(10 / 3, 3),
    :σ_R₀ => BigHalfNormal,
    :μ_δ => Gamma(1.3 / 0.05, 0.05),
    :σ_δ => HalfNormal,
    :μ_πv => Gamma(3 / 0.3, 0.3),
    :μ_πv => Gamma(4, 1),
    # :σ_πv => HalfNormal,
    :σ_πv => BigHalfNormal,
    :κ => HalfNormal
)

# Dictionary of fixed parameters. This is used to determine what we're actually sampling.
infer_hyper_params = true

fixed_individual_params = Dict(
    :z_R₀ => false,
    :z_δ => false,
    :z_πv => false,
    :infection_time => false,
    # These aren't actually sampled but some are recon
    :R₀ => true,
    :δ => true,
    :πv => true,
    :k => true,
    :c => true
)

fixed_shared_params = Dict(
    :μ_R₀ => false,
    :σ_R₀ => false,
    :μ_δ => false,
    :σ_δ => false,
    :μ_πv => false,
    :σ_πv => false,
    :κ => false,
    :μ_k => true,
    :σ_k => true,
    :μ_c => true,
    :σ_c => true
)

fixed_params = Dict(x for x in Iterators.flatten((fixed_individual_params, fixed_shared_params)))
