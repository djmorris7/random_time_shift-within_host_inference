"""
This script runs the within-host inference for the simulated data.
"""

include("within_host_inference.jl")
include("mcmc.jl")
include("../plotting.jl")
include("../io.jl")
include("../helpers.jl")

##

PILOT_RUN = true

Random.seed!(2023)

##

(data, ids) = get_cleaned_data("data/sims/sim_data_clean.csv")

N = length(data)

df_true_pars = CSV.read(data_dir("sims/parameters.csv"), DataFrame)
df_true_hyper_pars = CSV.read(data_dir("sims/hyper_parameters.csv"), DataFrame)

## Checking data looks ok

S0 = Int(8e7)
# Individual parameters (means) pulled from Ke et al. (2021)
μ_R₀ = 8.0
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

vl_model_pars = [μ_R₀, μ_k, μ_δ, 2, μ_c]

Z0_static = SA[S0 - (E0 + I0), E0, I0, V0]

tspan = (0.0, 40.0)
prob = ODEProblem(tcl_deterministic, Z0_static, tspan, vl_model_pars)

##

pars0 = deepcopy(mean_pars)
LOD = 2.6576090679593496

##

nn = load_nn()
M = ModelInternals(Z0 = Z0, prob = prob, nn = nn, LOD = LOD)
M_threads = [deepcopy(M) for _ in 1:Threads.nthreads()]

##

function intialise_params()
    df_true_pars = CSV.read(data_dir("sims/parameters.csv"), DataFrame)

    σ_R₀, σ_k, σ_δ, σ_πv, σ_c = df_true_hyper_pars[1, ["σ_R₀", "σ_k", "σ_δ", "σ_πv", "σ_c"]]
    ϕ₀ = SharedParams(μ_R₀, σ_R₀, μ_k, σ_k, μ_δ, σ_δ, μ_πv, σ_πv, μ_c, σ_c, 0.5)

    θ₀ = Vector{Params}()

    for i in 1:N
        R₀, k, δ, πv, c, t₀ = df_true_pars[i, ["R₀", "k", "δ", "πv", "c", "infection_time"]]
        z_R₀ = (R₀ - μ_R₀) / σ_R₀
        z_δ = (δ - μ_δ) / σ_δ
        z_πv = (πv - μ_πv) / σ_πv
        push!(θ₀, Params(z_R₀, R₀, k, z_δ, δ, z_πv, πv, c, t₀, infection_time_ranges[i]))
    end

    return (θ₀, ϕ₀)
end

# Fill in the feasible infection times
infection_time_ranges = [zeros(2) for _ in 1:N]
for (i, dat) in enumerate(data)
    earliest_observation = dat.obs_times[1]
    obs_peak_timing = dat.obs_times[argmax(dat.vl)]

    earliest_timing = earliest_observation - 5.0
    latest_timing = obs_peak_timing + 5.0

    # a = floor(true_infection_times[i]) - 0.5
    # b = ceil(true_infection_times[i]) + 0.5

    infection_time_ranges[i] = [earliest_timing, latest_timing]
end

df_true_pars = CSV.read(data_dir("sims/parameters.csv"), DataFrame)

σ_R₀, σ_k, σ_δ, σ_πv, σ_c = df_true_hyper_pars[1, ["σ_R₀", "σ_k", "σ_δ", "σ_πv", "σ_c"]]
ϕ₀ = SharedParams(μ_R₀, σ_R₀, μ_k, σ_k, μ_δ, σ_δ, μ_πv, σ_πv, μ_c, σ_c, 0.5)

θ₀ = Vector{Params}()

for i in 1:N
    R₀, k, δ, πv, c, t₀ = df_true_pars[i, ["R₀", "k", "δ", "πv", "c", "infection_time"]]
    z_R₀ = (R₀ - μ_R₀) / σ_R₀
    z_δ = (δ - μ_δ) / σ_δ
    z_πv = (πv - μ_πv) / σ_πv
    push!(θ₀, Params(z_R₀, R₀, k, z_δ, δ, z_πv, πv, c, t₀, infection_time_ranges[i]))
end

##

id = 1
laplace_approx_full_likelihood(θ₀[id], data[id], ϕ₀, M)
laplace_approx_only_likelihood(θ₀[id], data[id], ϕ₀, M)
exact_likelihood(θ₀[id], data[id], ϕ₀, M)

##

app = [laplace_approx_full_likelihood(θ₀[i], data[i], ϕ₀, M) for i in 1:N]
sum(app)
exact = [exact_likelihood(θ₀[i], data[i], ϕ₀, M) for i in 1:N]
sum(exact)
only_like = [laplace_approx_only_likelihood(θ₀[i], data[i], ϕ₀, M) for i in 1:N]
sum(only_like)

@profview for i in 1:N
    laplace_approx_only_likelihood(θ₀[i], data[i], ϕ₀, M)
end

@benchmark laplace_approx_full_likelihood(θ₀[1], data[1], ϕ₀, M)
@benchmark exact_likelihood(θ₀[1], data[1], ϕ₀, M)
@benchmark laplace_approx_only_likelihood(θ₀[1], data[1], ϕ₀, M)

sum(app)
sum(exact)
sum(only_like)

##

θ₀, ϕ₀ = intialise_params()

p_vals_exact = zeros(100)
p_vals_approx_all = zeros(100)
p_vals_approx = zeros(100)

πv_grid = range(0.5, 5, length = 100)

for (i, πv) in enumerate(πv_grid)
    θ₀[1].πv = πv
    p_vals_exact[i] = exp(exact_likelihood(θ₀[1], data[1], ϕ₀, M))
    p_vals_approx[i] = exp(laplace_approx_only_likelihood(θ₀[1], data[1], ϕ₀, M))
    p_vals_approx_all[i] = exp(laplace_approx_full_likelihood(θ₀[1], data[1], ϕ₀, M))
end

size_inches = (7.75, 3.5)
size_pt = size_inches .* 72
fig = Figure(size = size_pt, fontsize = 10, dpi = 300)
ax = Axis(fig[1, 1]; ax_kwargs..., xlabel = L"\rho")
lines!(ax, πv_grid, p_vals_exact, color = colors[1])
# scatter!(ax, πv_grid, p_vals_approx, color = :blue)
scatter!(ax, πv_grid[1:2:end], p_vals_approx_all[1:2:end], color = colors[2], markersize = 6)

##

θ₀, ϕ₀ = intialise_params()

n = 100
p_vals_exact = zeros(n)
p_vals_approx_all = zeros(n)
p_vals_approx = zeros(n)

R0_grid = range(5, 13, length = n)

for (i, R0) in enumerate(R0_grid)
    θ₀[1].R₀ = R0
    p_vals_exact[i] = exp(exact_likelihood(θ₀[1], data[1], ϕ₀, M))
    p_vals_approx[i] = exp(laplace_approx_only_likelihood(θ₀[1], data[1], ϕ₀, M))
    # p_vals_exact[i] = exact_likelihood(θ₀[1], data[1], ϕ₀, M)
    # p_vals_approx[i] = laplace_approx_only_likelihood(θ₀[1], data[1], ϕ₀, M)
    p_vals_approx_all[i] = exp(laplace_approx_full_likelihood(θ₀[1], data[1], ϕ₀, M))
end

ax = Axis(fig[1, 2]; ax_kwargs..., xlabel = L"R_0")
lines!(ax, R0_grid, p_vals_exact, color = colors[1])
# scatter!(ax, R0_grid, p_vals_approx, color = :blue)
scatter!(ax, R0_grid[1:2:end], p_vals_approx_all[1:2:end], color = colors[2], markersize = 6)

##

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, R0_grid, p_vals_exact - p_vals_approx, color = :red)
scatter!(ax, R0_grid, p_vals_exact - p_vals_approx_all, color = :blue)
display(fig)

##

θ₀, ϕ₀ = intialise_params()

p_vals_exact = zeros(100)
p_vals_approx_all = zeros(100)
p_vals_approx = zeros(100)

δ_grid = range(1, 1.8, length = 100)

for (i, δ) in enumerate(δ_grid)
    θ₀[1].δ = δ
    p_vals_exact[i] = exp(exact_likelihood(θ₀[1], data[1], ϕ₀, M))
    p_vals_approx[i] = exp(laplace_approx_only_likelihood(θ₀[1], data[1], ϕ₀, M))
    p_vals_approx_all[i] = exp(laplace_approx_full_likelihood(θ₀[1], data[1], ϕ₀, M))
end

ax = Axis(fig[1, 3]; ax_kwargs..., xlabel = L"\delta")
lines!(ax, δ_grid, p_vals_exact, color = colors[1])
# scatter!(ax, δ_grid, p_vals_approx, color = :blue)
scatter!(ax, δ_grid[1:2:end], p_vals_approx_all[1:2:end], color = colors[2], markersize = 6)

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

display(fig)

save(results_dir("likelihood_comparison.pdf"), fig)

max_error = maximum(abs.(p_vals_exact - p_vals_approx_all))

##

if PILOT_RUN
    N_indiv_pars = sum(1 - v for (k, v) in fixed_individual_params)
    N_shared_pars = sum(1 - v for (k, v) in fixed_shared_params)
    Σ_individuals_diag = 0.1 * ones(N_indiv_pars)
    Σ_individuals_diag = [1.0, 0.1, 0.1, 0.1]
    # Σ_individuals_diag = [1.0, 0.1, 0.1]
    # Σ_shared_diag = [3.0, 1e-4, 0.1, 1e-4, 0.1, 1e-4]
    Σ_shared_diag = 1e-3 * ones(N_shared_pars)
    # Σ_shared_diag[1] = 0.1
    Σs = intialise_Σs(N, Σ_individuals_diag, Σ_shared_diag)
else
    Σs = read_Σs(N; path = "sim_tuned_covariances/")
end

##

# samples_laplace = deepcopy(samples)
# df_samples_laplace = create_sampling_df(N, samples_laplace)

Random.seed!(10)

samples = metropolis_within_gibbs(θ₀, ϕ₀, data, Σs, 50_000; save_every = 1)

df = create_sampling_df(N, samples)

##

scale = (2.38 / sqrt(N_indiv_pars))^2

burnin = 10_000

samples_shared_log = samples[burnin:end, (end - 6):end]

for i in 1:N
    df_tmp = Matrix(df[:, ["z_R₀_$i", "z_δ_$i", "z_πv_$i", "infection_time_$i"]])
    tmp = Matrix(df[:, ["z_R₀_$i", "z_δ_$i", "z_πv_$i", "infection_time_$i"]])
    Σs[i] = 1.2 * cov(tmp)
    θ₀[i].z_R₀ = mean(df_tmp[:, 1])
    θ₀[i].z_δ = mean(df_tmp[:, 2])
    θ₀[i].z_πv = mean(df_tmp[:, 3])
    θ₀[i].infection_time = mean(df_tmp[:, 4])
    θ₀[i].R₀ = θ₀[i].z_R₀ * mean(df.σ_R₀) + mean(df.μ_R₀)
    θ₀[i].δ = θ₀[i].z_δ * mean(df.σ_δ) + mean(df.μ_δ)
    θ₀[i].πv = θ₀[i].z_πv * mean(df.σ_πv) + mean(df.μ_πv)
end

scale = (2.38 / sqrt(N_shared_pars))^2

Σs[end] = 0.1 * cov(Matrix(df[:, ["μ_R₀", "σ_R₀", "μ_δ", "σ_δ", "μ_πv", "σ_πv", "κ"]]))
ϕ₀.μ_R₀ = mean(df[:, "μ_R₀"])
ϕ₀.σ_R₀ = mean(df[:, "σ_R₀"])
ϕ₀.μ_δ = mean(df[:, "μ_δ"])
ϕ₀.σ_δ = mean(df[:, "σ_δ"])
ϕ₀.μ_πv = mean(df[:, "μ_πv"])
ϕ₀.σ_πv = mean(df[:, "σ_πv"])
ϕ₀.κ = mean(df[:, "κ"])

##

function sample_starting_points()
    θ₀, ϕ₀ = intialise_params()

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
        tmp = sample_starting_points()
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
    println("Finished chain $i.")
    df = create_sampling_df(N, samples)
    CSV.write(results_dir("samples_sim_$i.csv"), df)
end

save_Σs(Σs; path = "sim_tuned_covariances/")
