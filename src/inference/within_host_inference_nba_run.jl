"""
This script runs the within-host inference for the simulated data.
"""

include("within_host_inference.jl")
include("mcmc.jl")
include("../plotting.jl")
include("../io.jl")

##

Random.seed!(2023)

(data, id_mapping) = get_cleaned_data("data/nba/nba_data_clean.csv")

##

N = length(data)

order = sortperm([length(data[i].vl) for i in eachindex(data)])

idx = 24

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, data[idx].obs_times, data[idx].vl)
display(fig)

##

fig = Figure(size = (1600, 1200))
axs = [Axis(fig[i, j]) for i in 1:12, j in 1:12]
for (i, id) in enumerate(eachindex(data))
    i > length(axs) && break
    scatter!(axs[i], data[i].obs_times, data[i].vl)
end
display(fig)

save("figures/zitzmann_data.pdf", fig, pt_per_unit = 1)

timeseries_lengths = [length(d.vl) for d in data]
unique_lengths = unique(timeseries_lengths)
counts = [sum(timeseries_lengths .== l) for l in unique_lengths]

fig = Figure()
ax = Axis(fig[1, 1])
barplot!(ax, unique_lengths, counts)
ax.xlabel = "Timeseries length"
ax.ylabel = "Count"
display(fig)

obs_before_peaks = [sum(d.obs_times .< 0) for d in data]
unique_lengths = unique(obs_before_peaks)
counts = [sum(obs_before_peaks .== l) for l in unique_lengths]

fig = Figure()
ax = Axis(fig[1, 1])
barplot!(ax, unique_lengths, counts)
ax.xlabel = "Number of pre-peak observations"
ax.ylabel = "Count"
display(fig)

##

df_true_hyper_pars = CSV.read(data_dir("sims/hyper_parameters.csv"), DataFrame)
df_true_pars = CSV.read(data_dir("sims/parameters.csv"), DataFrame)

# Now check to see that extinction prob calculation is equal no matter the way we get it
S0 = Int(8e7)
# Individual parameters (means) pulled from Ke et al. (2021)
μ_R₀ = df_true_hyper_pars[1, "μ_R₀"]
μ_k = df_true_hyper_pars[1, "μ_k"]
μ_δ = df_true_hyper_pars[1, "μ_δ"]
μ_πv = df_true_hyper_pars[1, "μ_πv"]
μ_c = df_true_hyper_pars[1, "μ_c"]
mean_pars = [8, μ_k, μ_δ, μ_πv, μ_c]

κ = df_true_hyper_pars[1, "κ"]

E0 = 1
I0 = 0
V0 = 0
Z0 = [S0 - (E0 + I0), E0, I0, V0]

##

vl_model_pars = [8, μ_k, μ_δ, μ_πv, μ_c]
obs_t = 70
tspan = (0, 70)

Z0_static = SA[Z0...]
prob = ODEProblem(tcl_deterministic, Z0_static, tspan, vl_model_pars)
sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, save_idxs = 4)

##

pars0 = deepcopy(mean_pars)

LOD = 2.658

nn = load_nn()
M = ModelInternals(Z0 = Z0, prob = prob, nn = nn, LOD = LOD)
M_threads = [deepcopy(M) for _ in 1:Threads.nthreads()]

##

# Fill in the feasible infection times
infection_time_ranges = [zeros(2) for _ in 1:N]
infection_time0 = init_infection_times(data)

all_params = [
    Params(0.0, μ_R₀, μ_k, 0.0, μ_δ, 0.0, μ_πv, μ_c, x, y) for
    (x, y) in zip(infection_time0, infection_time_ranges)
]

σ_R₀, σ_k, σ_δ, σ_πv, σ_c = df_true_hyper_pars[1, ["σ_R₀", "σ_k", "σ_δ", "σ_πv", "σ_c"]]

shared_params = SharedParams(8, σ_R₀, μ_k, σ_k, μ_δ, σ_δ, μ_πv, σ_πv, μ_c, σ_c, κ)
# shared_params = SharedParams(μ_β, 1.0, μ_k, σ_k, μ_δ, 0.1, μ_πv, 1.0, μ_c, σ_c, κ)

##

# Initialise starting values and covariance matrices for MCMC
θ₀ = deepcopy(all_params)
ϕ₀ = deepcopy(shared_params)

##

N_indiv_pars = sum(1 - v for (k, v) in fixed_individual_params)
N_shared_pars = sum(1 - v for (k, v) in fixed_shared_params)

Σ_individuals_diag = 1.0 * ones(N_indiv_pars)
Σ_shared_diag = 1e-3 * ones(N_shared_pars)
# Σs = intialise_Σs(N, Σ_individuals_diag, Σ_shared_diag)
Σs = intialise_Σs(N, Σ_individuals_diag, Σ_shared_diag)

# Σs[end][1, 1] = 1.0

df_samples = CSV.read(results_dir("samples_nba.csv"), DataFrame)

##

Random.seed!(100)

# data = data[1:50]
# θ₀ = θ₀[1:50]
# Σs = Σs[[1:50; N + 1]]

samples = metropolis_within_gibbs(
    θ₀, ϕ₀, data, Σs, 100_000, fixed_params = fixed_params, save_every = 1
)

df = create_sampling_df(N, samples)

##

fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, df.μ_πv, bins = 100, normalization = :pdf)
plot!(ax, hyper_priors[:μ_πv], color = :red)
# plot!(ax, Truncated(Normal(0, 1), 0, Inf), color = :red)
display(fig)

fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, df.σ_πv, bins = 100, normalization = :pdf)
plot!(ax, hyper_priors[:σ_πv], color = :red)
# plot!(ax, Truncated(Normal(0, 1), 0, Inf), color = :red)
display(fig)

##

# Look at a single individuals traceplots to make sure things are happening
fig = Figure()
axs = [Axis(fig[i, j]) for i in 1:2, j in 1:2]
plot_inds = [1, 3, 4, 6] .+ (31 * 6)
for (i, ind) in enumerate(plot_inds)
    lines!(axs[i], samples[:, ind], color = :black, alpha = 0.5)
    # hist!(axs[i], samples[:, ind])
end
display(fig)

##

fig = Figure(size = (600, 600))
axs = [Axis(fig[i, 1]) for i in 1:8]
for (i, j) in enumerate([1, 2, 5, 6, 7, 8, 11])
    # lines!(axs[i], samples[50000:end, end - 11 + j], color = :black, alpha = 0.5)
    lines!(axs[i], samples[:, end - 11 + j], color = :black, alpha = 0.5)
end
display(fig)

##

df = create_sampling_df(N, samples)

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, hyper_priors[:σ_πv], color = :red)
hist!(ax, df.σ_πv, bins = 100, normalization = :pdf)
display(fig)

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, hyper_priors[:σ_R₀], color = :red)
hist!(ax, df.σ_R₀, bins = 100, normalization = :pdf)
display(fig)

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, hyper_priors[:μ_R₀], color = :red)x
hist!(ax, df.μ_R₀, bins = 100, normalization = :pdf)
display(fig)

##

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, df.R₀_1, color = (:black, 0.2))
lines!(ax, df.R₀_2, color = (:black, 0.2))
lines!(ax, df.μ_R₀, color = :red)
display(fig)

##

R0_means = [mean(samples[:, 1 + 9 * (i - 1)]) for i in 1:N]

plot(R0_means)

##

labels = ["μ_β", "σ_β", "μ_δ", "σ_δ", "μ_πv", "σ_πv", "κ"]

fig = Figure(size = (700, 450))
axs = [Axis(fig[i, j]) for i in 1:3, j in 1:3]
for (i, j) in enumerate([1, 2, 5, 6, 7, 8, 11])
    hist!(axs[i], samples[:, end - 11 + j])
    axs[i].xlabel = labels[i]
end
display(fig)

##

# c = 2.4 / sqrt(d)

df = create_sampling_df(N, samples)
# scalings = 2.0 * ones(N + 1)
# scalings[end] = 2.0
scalings = [2.38^2 / size(Σs[i], 1) for i in 1:(N + 1)]
# scalings = [1.1 for i in 1:(N + 1)]
scalings[end] = 0.1
# scalings[1:N] .= 1.0
# scalings[N + 1] = 0.0001
burnin = 10000
for i in 1:N
    df_tmp = Matrix(df[:, ["z_R₀_$i", "z_δ_$i", "z_πv_$i", "infection_time_$i"]])
    tmp = Matrix(df[:, ["z_R₀_$i", "z_δ_$i", "z_πv_$i", "infection_time_$i"]])
    Σs[i] = 1.0 * cov(tmp)
    θ₀[i].z_R₀ = mean(df_tmp[:, 1])
    θ₀[i].z_δ = mean(df_tmp[:, 2])
    θ₀[i].z_πv = mean(df_tmp[:, 3])
    θ₀[i].infection_time = mean(df_tmp[:, 4])
    θ₀[i].R₀ = θ₀[i].z_R₀ * mean(df.σ_R₀) + mean(df.μ_R₀)
    θ₀[i].δ = θ₀[i].z_δ * mean(df.σ_δ) + mean(df.μ_δ)
    θ₀[i].πv = θ₀[i].z_πv * mean(df.σ_πv) + mean(df.μ_πv)
end

scale = (2.38 / sqrt(N_shared_pars))^2

Σs[end] = 0.15 * cov(Matrix(df[:, ["μ_R₀", "σ_R₀", "μ_δ", "σ_δ", "μ_πv", "σ_πv", "κ"]]))
ϕ₀.μ_R₀ = mean(df[:, "μ_R₀"])
ϕ₀.σ_R₀ = mean(df[:, "σ_R₀"])
ϕ₀.μ_δ = mean(df[:, "μ_δ"])
ϕ₀.σ_δ = mean(df[:, "σ_δ"])
ϕ₀.μ_πv = mean(df[:, "μ_πv"])
ϕ₀.σ_πv = mean(df[:, "σ_πv"])
ϕ₀.κ = mean(df[:, "κ"])

if !isdir(results_dir("nba_tuned_covariances"))
    mkdir(results_dir("nba_tuned_covariances"))
end
# mkdir(results_dir("nba_tuned_covariances"))
# save_Σs(Σs; path = "nba_tuned_covariances/")

##

function sample_starting_points(all_params, shared_params)
    θ₀ = deepcopy(all_params)
    ϕ₀ = deepcopy(shared_params)

    ϕ₀.μ_R₀ = ϕ₀.μ_R₀ + randn() * 1.5
    ϕ₀.σ_R₀ = ϕ₀.σ_R₀ + randn()
    ϕ₀.μ_δ = ϕ₀.μ_δ + randn()
    ϕ₀.σ_δ = ϕ₀.σ_δ + randn()
    ϕ₀.μ_πv = ϕ₀.μ_πv + randn()
    ϕ₀.σ_πv = ϕ₀.σ_πv + randn()
    ϕ₀.κ = ϕ₀.κ + randn()

    if isinf(shared_prior(ϕ₀))
        return false
    end

    for i in 1:N
        θ₀[i].z_R₀ = θ₀[i].z_R₀ + randn() * 0.1
        θ₀[i].z_δ = θ₀[i].z_δ + randn() * 0.1
        θ₀[i].z_πv = θ₀[i].z_πv + randn() * 0.1
        θ₀[i].R₀ = θ₀[i].z_R₀ * ϕ₀.σ_R₀ + ϕ₀.μ_R₀
        θ₀[i].δ = θ₀[i].z_δ * ϕ₀.σ_δ + ϕ₀.μ_δ
        θ₀[i].πv = θ₀[i].z_πv * ϕ₀.σ_πv + ϕ₀.μ_πv

        if isinf(individual_prior(θ₀[i], ϕ₀)) || !is_in_θ_support!(θ₀[i])
            return false
        end

        if isinf(likelihood(θ₀[i], data[i], ϕ₀, M_threads[1]))
            return false
        end
    end

    return (θ₀, ϕ₀)
end

##

Random.seed!(2024)

initial_parameter_sets = []

for i in 1:4
    while true
        tmp = sample_starting_points(all_params, shared_params)
        if tmp == false
            continue
        end
        θ₀, ϕ₀ = tmp
        push!(initial_parameter_sets, (θ₀, ϕ₀))
        break
    end
end

##

for (i, (θ₀, ϕ₀)) in enumerate(initial_parameter_sets)
    println("Starting chain $i...")
    samples = metropolis_within_gibbs(
        θ₀, ϕ₀, data, Σs, 100_000; fixed_params = fixed_params, save_every = 1
    )
    df = create_sampling_df(N, samples)
    CSV.write(results_dir("samples_nba_$i.csv"), df)
    println("Finished chain $i.")
end
