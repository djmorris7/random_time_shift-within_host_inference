"""
test_mcmc.jl

    This script tests most of the MCMC functionality. It needs to be run from the root directory of the project
    (i.e. the directory containing the Project.toml file) and opening normally in VSCode will work fine.

    The script will test the proposal functions and the fill functions for the parameters. It will print out the
    before and after values for each parameter. These need to be checked manually to ensure that the functions are
    working as expected.

    TODO: Maybe add actual test suite but this is a good start.
"""

# include("../src/inference_centered/within_host_inference.jl")
# include("../src/inference_centered/mcmc.jl")
include("../src/inference/within_host_inference.jl")
include("../src/inference/mcmc.jl")

##

Random.seed!(2023)

data, true_infection_times, vl, obs_t = load_sim_data()
N = length(data)

df_true_hyper_pars = CSV.read(data_dir("sims/hyper_parameters.csv"), DataFrame)

##

# Individual parameters (means) pulled from Ke et al. (2021)
μ_β = df_true_hyper_pars[1, "μ_R₀"]
μ_k = df_true_hyper_pars[1, "μ_k"]
μ_δ = df_true_hyper_pars[1, "μ_δ"]
μ_πv = df_true_hyper_pars[1, "μ_πv"]
μ_c = df_true_hyper_pars[1, "μ_c"]
mean_pars = [μ_β, μ_k, μ_δ, μ_πv, μ_c]

κ = df_true_hyper_pars[1, "κ"]

##

pars0 = deepcopy(mean_pars)
LOD = 2.0

##

# Fill in the feasible infection times
infection_time_ranges = [zeros(2) for _ in eachindex(data)]
for (i, dat) in enumerate(data)
    obs_peak_timing = dat.obs_times[findmax(dat.vl)[2]]

    earliest_timing = obs_peak_timing - 20
    latest_timing = obs_peak_timing + 10
    a = obs_peak_timing - 20
    b = obs_peak_timing + 10

    # a = floor(true_infection_times[i])
    # b = ceil(true_infection_times[i])

    infection_time_ranges[i] = [a, b]
end

all_params = [
    Params(0, μ_β, μ_k, 0, μ_δ, 0, μ_πv, μ_c, x, y) for
    (x, y) in zip(true_infection_times, infection_time_ranges)
]

σ_R₀, σ_k, σ_δ, σ_πv, σ_c = df_true_hyper_pars[1, ["σ_R₀", "σ_k", "σ_δ", "σ_πv", "σ_c"]]
σ_R₀ = 0.1

shared_params = SharedParams(μ_R₀, σ_R₀, μ_k, σ_k, μ_δ, σ_δ, μ_πv, σ_πv, μ_c, σ_c, κ)

## ============== Check the proposal functions ==============

# Initialise starting values and covariance matrices for MCMC
θ₀ = deepcopy(all_params)
ϕ₀ = deepcopy(shared_params)

N_indiv_pars = sum(1 - v for (k, v) in fixed_individual_params)
N_shared_pars = sum(1 - v for (k, v) in fixed_shared_params)

Σ_individuals_diag = 0.5 * ones(N_indiv_pars)
Σ_shared_diag = 1e-3 * ones(N_shared_pars)
Σ_shared_diag[1] = 1e-3
Σs = intialise_Σs(N, Σ_individuals_diag, Σ_shared_diag)

##

ϕ_old = deepcopy(ϕ₀)
ϕ_new = deepcopy(ϕ₀)
θ_old = deepcopy(θ₀[1])
θ_new = deepcopy(θ_old)

θ_proposal!(θ_new, θ_old, ϕ_old, Σs[1])

for k in fieldnames(Params)
    if k != :infection_time_range
        println("$k: $(getfield(θ_old, k)) --> $(round(getfield(θ_new, k), digits = 2))")
    end
    # println("$(k): $(θ_old.k)")
end

ϕ_proposal!(ϕ_new, ϕ_old, Σs[end])

for k in fieldnames(SharedParams)
    println("$k: $(getfield(ϕ_old, k)) --> $(round(getfield(ϕ_new, k), digits = 2))")
end

ϕ_new_samps = Dict(
    :μ_β => [],
    :σ_β => [],
    :μ_k => [],
    :σ_k => [],
    :μ_δ => [],
    :σ_δ => [],
    :μ_πv => [],
    :σ_πv => [],
    :μ_c => [],
    :σ_c => [],
    :κ => []
)

for _ in 1:10000
    ϕ_proposal!(ϕ_new, ϕ_old, Σs[end])
    for k in fieldnames(SharedParams)
        push!(ϕ_new_samps[k], getfield(ϕ_new, k))
    end
end

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1])
# hist!(ax, log.(ϕ_new_samps[:μ_β]), bins = 50, color = :blue, label = "β")
# hist!(ax, ϕ_new_samps[:μ_β], bins = 50, color = :blue, label = "β")
hist!(ax, ϕ_new_samps[:σ_β], bins = 50, color = :blue, label = "β")
display(fig)

## ============== Check the fill functions ==============

θ_from = deepcopy(θ_new)
θ_to = deepcopy(θ_old)

fill_θ!(θ_to, θ_from)

for k in fieldnames(Params)
    if k != :infection_time_range
        println(
            "$k: $(round(getfield(θ_from, k), digits = 2)) --> $(round(getfield(θ_to, k), digits = 2))"
        )
    end
end

ϕ_from = deepcopy(ϕ_new)
ϕ_to = deepcopy(ϕ_old)

fill_ϕ!(ϕ_to, ϕ_from)

for k in fieldnames(SharedParams)
    println(
        "$k: $(round(getfield(ϕ_from, k), digits = 2)) --> $(round(getfield(ϕ_to, k), digits = 2))"
    )
end

params_to_vec(θ_new)
vec_to_params(params_to_vec(θ_new), [1, 2])

shared_params_to_vec(ϕ_new)
vec_to_shared_params(shared_params_to_vec(ϕ_new))

## ============== Check the likelihood calculation ==============

## Reading in data

Random.seed!(2023)
data, true_infection_times, vl, obs_t = load_sim_data()
data = drop_lod_data(data; lod = 0.0)
N = length(data)
df_true_pars = CSV.read(data_dir("sims/parameters.csv"), DataFrame)
df_true_hyper_pars = CSV.read(data_dir("sims/hyper_parameters.csv"), DataFrame)

## Checking data looks ok

# Now check to see that extinction prob calculation is equal no matter the way we get it
S0 = Int(8e7)
# Individual parameters (means) pulled from Ke et al. (2021)
μ_R₀ = df_true_hyper_pars[1, "μ_R₀"]
μ_k = df_true_hyper_pars[1, "μ_k"]
μ_δ = df_true_hyper_pars[1, "μ_δ"]
μ_πv = df_true_hyper_pars[1, "μ_πv"]
μ_c = df_true_hyper_pars[1, "μ_c"]
mean_pars = [μ_R₀, μ_k, μ_δ, μ_πv, μ_c]

κ = df_true_hyper_pars[1, "κ"]

E0 = 1
I0 = 0
V0 = 0

Z0 = [S0 - (E0 + I0), E0, I0, V0]

##

vl_model_pars = [μ_β / S0, μ_k, μ_δ, μ_πv, μ_c]
obs_t = 70
tspan = (0, 20)

function tcl_deterministic(x, pars, t)
    """
    The ODE system for the within-host model.
    """
    β, k, δ, πv, c = pars
    s, e, i, v = x

    ds = -β * v * s
    de = β * v * s - k * e
    di = k * e - δ * i
    dv = πv * i - c * v

    return SA[ds, de, di, dv]
end

Z0_static = SA[S0 - (E0 + I0), E0, I0, V0]

prob = ODEProblem(tcl_deterministic, Z0_static, tspan, vl_model_pars)

##

pars0 = deepcopy(mean_pars)
LOD = 2.0

##

nn = load_nn()
M = ModelInternals(Z0 = Z0, prob = prob, nn = nn, LOD = LOD)
M_threads = [deepcopy(M) for _ in 1:Threads.nthreads()]

##

# Fill in the feasible infection times
infection_time_ranges = [zeros(2) for _ in eachindex(true_infection_times)]
for (i, dat) in enumerate(data)
    obs_peak_timing = dat.obs_times[findmax(dat.vl)[2]]

    earliest_timing = obs_peak_timing - 20
    latest_timing = obs_peak_timing + 10

    a = floor(true_infection_times[i])
    b = ceil(true_infection_times[i])

    infection_time_ranges[i] = [a, b]
end

all_params = [
    Params(μ_β, μ_k, μ_δ, μ_πv, μ_c, x, y) for
    (x, y) in zip(true_infection_times, infection_time_ranges)
]

σ_β, σ_k, σ_δ, σ_πv, σ_c = df_true_hyper_pars[1, ["σ_β", "σ_k", "σ_δ", "σ_πv", "σ_c"]]

σ_β = 1.0

shared_params = SharedParams(μ_β, σ_β, μ_k, σ_k, μ_δ, σ_δ, μ_πv, σ_πv, μ_c, σ_c, κ)

##

# Initialise starting values and covariance matrices for MCMC
θ₀ = deepcopy(all_params)
ϕ₀ = deepcopy(shared_params)

# Test single likelihood
likelihood(θ₀[1], data[1], ϕ₀, M_threads[1])

# Test multiple likelihoods

sum(likelihood(θ, dat, ϕ₀, M) for (θ, dat) in zip(θ₀, data))

# Test priors

individual_prior(θ₀[1], ϕ₀)

shared_prior(ϕ₀)
