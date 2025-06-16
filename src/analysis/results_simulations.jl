include("../inference/within_host_inference.jl")
include("results.jl")
include("../plotting.jl")

##

Random.seed!(2023)
(data, ids) = get_cleaned_data("data/sims/sim_data_clean.csv")
df_true_pars = CSV.read(data_dir("sims/parameters.csv"), DataFrame)
df_true_hyper_pars = CSV.read(data_dir("sims/hyper_parameters.csv"), DataFrame)

true_infection_times = df_true_pars.infection_time

fig_loc = "figures/simulation/"
if isdir(fig_loc) == false
    mkdir(fig_loc)
end

##

# Now check to see that extinction prob calculation is equal no matter the way we get it
S0 = Int(8e7)
# Individual parameters (means)
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

obs_t = 50
tspan = (0, 50)
ode_pars = [8, μ_k, μ_δ, μ_πv, μ_c]
Z0_static = SA[Z0...]
prob = ODEProblem(tcl_deterministic, Z0_static, tspan, ode_pars)
sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, save_idxs = 4)

plot(sol.t, log10p0.(sol.u))

##

pars0 = deepcopy(mean_pars)
nn = load_nn()

## Testing the likelihood for a single individual
S0 = Z0[1]
Z0_bp = Z0[2:end]
LOD = 2.0

integrator = f -> quadgk(f, -7, 7)[1]

## Testing the likelihood for a single individual

σ_R₀, σ_k, σ_δ, σ_πv, σ_c = df_true_hyper_pars[1, [:σ_R₀, :σ_k, :σ_δ, :σ_πv, :σ_c]]

##

infection_time_ranges = [zeros(2) for _ in eachindex(true_infection_times)]
for (i, dat) in enumerate(data)
    obs_peak_timing = dat.obs_times[findmax(dat.vl)[2]]

    earliest_timing = obs_peak_timing - 20
    latest_timing = obs_peak_timing + 10

    infection_time_ranges[i] = [earliest_timing, latest_timing]
end

##

# df_samples = CSV.read(results_dir("samples_sim.csv"), DataFrame)
df_samples = [CSV.read(results_dir("samples_sim_$i.csv"), DataFrame) for i in 1:4]
burnin = 10000
thin = 10
df_samples = [df_samples[i][burnin:thin:end, :] for i in 1:4]
df_samples = vcat(df_samples...)
samples = Matrix(df_samples)

df = CSV.read(data_dir("sims/parameters.csv"), DataFrame)
select!(df, [:ID, :R₀, :k, :δ, :πv, :c, :infection_time])

##

id = 1

# get params
# df_samples_ind = extract_individual_params(df_samples, id)

function get_df_samples_ind(df_samples, id)
    df_samples_ind = DataFrame()
    df_samples_ind.R₀ = df_samples[:, "z_R₀_$id"] .* df_samples.σ_R₀ .+ df_samples.μ_R₀
    df_samples_ind.k .= μ_k
    df_samples_ind.δ = df_samples[:, "z_δ_$id"] .* df_samples.σ_δ .+ df_samples.μ_δ
    df_samples_ind.πv = df_samples[:, "z_πv_$id"] .* df_samples.σ_πv .+ df_samples.μ_πv
    df_samples_ind.c .= μ_c
    df_samples_ind.infection_time = df_samples[:, "infection_time_$id"]

    return df_samples_ind
end

df_samples_ind = get_df_samples_ind(df_samples, id)

# df_samples_ind.πv .= 3.0
# df_samples_ind.β .= 38.0
κ_post = df_samples[:, :κ]

##

T = 100
n_sims = 1000
post_sims = ppc_simulation(df_samples_ind, Z0_bp, nn, prob, κ_post, n_sims, T; t0 = -20, Δt = 0.1)
post_sims_summ = summarise_ppc_sims(post_sims)

##

param_sigmas = CSV.read(data_dir("param_sigma.csv"), DataFrame)
σ_β, σ_δ, σ_πv = param_sigmas[1, ["β", "δ", "πv"]]

nsamples_prior = 1000
prior_samples = zeros(nsamples_prior, 6)

j = 1

for (i, symbol) in enumerate(fieldnames(Params))
    symbol ∈ [:infection_time_range, :z_β, :z_δ, :z_πv] && continue

    if symbol == :infection_time
        prior_samples[:, j] = rand(Uniform(infection_time_ranges[id]...), nsamples_prior)
    elseif symbol == :k
        prior_samples[:, j] .= μ_k
    elseif symbol == :c
        prior_samples[:, j] .= μ_c
    else
        μ = rand(hyper_priors[string_to_symbol("μ_", symbol)], nsamples_prior)
        σ = rand(hyper_priors[string_to_symbol("σ_", symbol)], nsamples_prior)
        if symbol == :β
            prior_samples[:, j] = rand.(Truncated.(Normal.(μ, σ), 0.1, 100))
        elseif symbol == :πv
            # σ = σ_πv
            prior_samples[:, j] = rand.(Truncated.(Normal.(μ, σ), 0.1, 100))
        else
            # σ = rand(hyper_priors[string_to_symbol("σ_", symbol)], nsamples_prior)
            prior_samples[:, j] = rand.(Truncated.(Normal.(μ, σ), 0.25, 100))
        end
    end
    j += 1
end

κ_prior = rand(hyper_priors[:κ], nsamples_prior)

prior_samples = DataFrame(prior_samples, [:β, :k, :δ, :πv, :c, :infection_time])

prior_sims = ppc_simulation(prior_samples, Z0_bp, nn, prob, κ_post, n_sims, T; t0 = -20, Δt = 0.1)
prior_sims_summ = summarise_ppc_sims(prior_sims)

##

fig = Figure(size = size_pt, fontsize = 12, dpi = 300, colormap = Makie.wong_colors())
ax = Axis(fig[1, 1])
plot!(ax, data[id].obs_times, data[id].vl, color = :black)
lines!(ax, post_sims_summ["t"], post_sims_summ["median"], color = :dodgerblue)
band!(
    ax,
    post_sims_summ["t"],
    post_sims_summ["bottom"],
    post_sims_summ["top"],
    color = (:dodgerblue, 0.2)
)

ind_simmed_pars = df_true_pars[id, [:R₀, :k, :δ, :πv, :c, :infection_time, :τ]]
R₀, k, δ, πv, c, t_inf, τ = ind_simmed_pars

prob = ODEProblem(tcl_deterministic, Z0_static, (t_inf - τ, t_inf - τ + 30), [R₀, k, δ, πv, c])
sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, save_idxs = 4)

path_likelihood(0, 0.5, t_inf, data[id], sol)

lines!(ax, sol.t, log10p0.(sol.u), color = :black, linestyle = :dash)
xlims!(ax, -10, 25)
display(fig)

save(fig_loc * "predictive_plot_individual.pdf", fig, pt_per_unit = 1.0)

##

function get_means(df)
    means = zeros(6)
    for (i, symbol) in enumerate(names(df))
        if symbol == :infection_time
            means[i] = mean(df.infection_time)
        else
            means[i] = mean(df[!, symbol])
        end
    end
    return means
end

get_means(df_samples_ind)

fig = Figure()
axs = [Axis(fig[i, j]) for i in 1:2, j in 1:3]
for (i, s) in enumerate(["R₀", "δ", "πv", "infection_time"])
    hist!(axs[i], df_samples_ind[!, s], bins = 50, color = :blue, normalization = :pdf)
    vlines!(axs[i], df_true_pars[id, s], color = :red)
end
display(fig)

##

quantile(df_samples_ind.R₀ .- df_true_pars[id, :R₀], 0.025)

##

# Spaghetti plot

fig = Figure(size = size_pt, fontsize = 12, dpi = 300)
axs = [
    Axis(fig[1, 1], ylabel = L"\textrm{viral load } (\log_{10})", xlabel = L"\textrm{time (days)}"),
    Axis(fig[1, 2], ylabel = L"\textrm{viral load } (\log_{10})", xlabel = L"\textrm{time (days)}")
]
for i in 1:10
    lines!(
        axs[1],
        prior_sims[:, 1],
        prior_sims[:, rand(1:size(prior_sims, 2))],
        color = :green,
        alpha = 0.4
    )
    ind = rand(2:size(prior_sims, 2))
    println(ind)
    lines!(axs[2], post_sims[:, 1], post_sims[:, ind], color = :red, alpha = 0.4)
end
xlims!(axs[1], data[id].obs_times[1] - 1, data[id].obs_times[end] + 1)
xlims!(axs[2], data[id].obs_times[1] - 1, data[id].obs_times[end] + 1)
display(fig)

save(fig_loc * "predictive_spaghetti.pdf", fig, pt_per_unit = 1.0)

## --- Sample a few people at onece ---

Random.seed!(2024)

# order[21]
# order[24]
# order[31]
# order[35]
# order[42]

# ids_pivotal = [21, 24, 31, 35, 42]
ids_pivotal = sort([
    81, 52, 72, 1, 19, 6, 67, 44, 36, 9, 31, 14, 3, 51, 77, 33, 59, 13, 86, 54, 88, 38
])
# ids_pivotal = [order[x] for x in ids_pivotal]

size_inches = (7.5, 5.5)
size_pt = size_inches .* 72
fig = Figure(size = size_pt, fontsize = 10, dpi = 300)
# fig[1:4, 1:5] = [Axis(fig) for _ in 1:20]
# axs = [
#     Axis(fig[i, j], titlealign = :left, xgridvisible = false, ygridvisible = false) for i in 1:4,
#     j in 1:5
# ]

row = 1
col = 1

@showprogress for id in ids_pivotal
    ax = Axis(fig[row, col]; ax_kwargs...)

    df_samples_ind = get_df_samples_ind(df_samples, id)
    κ_post = df_samples[:, :κ]

    T = 100
    n_sims = 1000
    post_sims = ppc_simulation(
        df_samples_ind, Z0_bp, nn, prob, κ_post, n_sims, T; t0 = -20, Δt = 0.1
    )

    # prior_sims_summ = summarise_ppc_sims(prior_sims)
    post_sims_summ = summarise_ppc_sims(post_sims)

    if id ∈ ids_pivotal
        lines!(ax, post_sims_summ["t"], post_sims_summ["median"], color = colors[1])
        band!(
            ax,
            post_sims_summ["t"],
            post_sims_summ["lower"],
            post_sims_summ["upper"],
            color = (colors[1], 0.2)
        )
        band!(
            ax,
            post_sims_summ["t"],
            post_sims_summ["bottom"],
            post_sims_summ["top"],
            color = (colors[1], 0.2)
        )
    end
    plot!(ax, data[id].obs_times, data[id].vl, color = :black, markersize = 4)
    text!(
        ax,
        15,
        6,
        text = "ID: $id",
        fontsize = 8,
        justification = :center,
        align = (:center, :bottom)
    )
    # ax.title = "ID: $id"
    xlims!(ax, -10, 21)

    ax.yticks = 0:3:9
    ylims!(ax, low = 0)

    row += 1
    if row > 5
        row = 1
        col += 1
    end
    ylims!(ax, low = 2.5)
end

Label(fig[1:5, 0], L"\log_{10}(\textrm{viral load})", rotation = pi / 2)
Label(fig[6, 1:5], L"\textrm{time (days) since peak VL}")

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)
display(fig)

save(fig_loc * "predictive_plot_multi_individuals.pdf", fig, pt_per_unit = 1.0)

## CRPS

## --- Now do some sampling using the hierarchical part of the model ---

# Now sample completely new individuals using the hyper parameters
df_samples_joint = df_samples[
    :, [:μ_R₀, :σ_R₀, :μ_k, :σ_k, :μ_δ, :σ_δ, :μ_πv, :σ_πv, :μ_c, :σ_c, :κ]
]

# now we need to draw new samples
nsamples = 50_000
new_samples = zeros(nsamples, 6)

j = 1
for (i, symbol) in enumerate(fieldnames(Params))
    symbol ∈ [:infection_time_range, :z_R₀, :z_πv, :z_δ] && continue

    if symbol == :infection_time
        new_samples[:, j] .= 0.0
    elseif symbol == :k
        new_samples[:, j] .= μ_k
    elseif symbol == :c
        new_samples[:, j] .= μ_c
    else
        μ = df_samples_joint[1:nsamples, string_to_symbol("μ_", symbol)]
        σ = df_samples_joint[1:nsamples, string_to_symbol("σ_", symbol)]
        new_samples[:, j] = rand.(Truncated.(Normal.(μ, σ), 0.01, 100))
    end
    j += 1
end

new_samples = DataFrame(new_samples, [:R₀, :k, :δ, :πv, :c, :infection_time])

##

post_sims = ppc_simulation(new_samples, Z0_bp, nn, prob, κ_post, n_sims, T; t0 = -20.0, Δt = 0.1)
post_sims_summ = summarise_ppc_sims(post_sims)

nsamples_prior = 20_000
prior_samples = zeros(nsamples_prior, 6)

j = 1
for (i, symbol) in enumerate(fieldnames(Params))
    symbol ∈ [:infection_time_range, :z_R₀, :z_δ, :z_πv] && continue

    if symbol == :infection_time
        prior_samples[:, j] .= 0.0
    elseif symbol == :k
        prior_samples[:, j] .= μ_k
    elseif symbol == :c
        prior_samples[:, j] .= μ_c
    else
        μ = rand(hyper_priors[string_to_symbol("μ_", symbol)], nsamples_prior)
        σ = rand(hyper_priors[string_to_symbol("σ_", symbol)], nsamples_prior)
        prior_samples[:, j] = rand.(Truncated.(Normal.(μ, σ), 0.01, 100))
    end
    j += 1
end

prior_samples = DataFrame(prior_samples, [:R₀, :k, :δ, :πv, :c, :infection_time])
κ_prior = rand(hyper_priors[:κ], nsamples_prior)

prior_sims = ppc_simulation(prior_samples, Z0_bp, nn, prob, κ_prior, n_sims, T; t0 = -20, Δt = 0.1)
prior_sims_summ = summarise_ppc_sims(prior_sims)

##

fig = Figure()
ax = Axis(fig[1, 1])
# hist!(ax, new_samples.β, bins = 50, color = :blue, normalization = :pdf)
# hist!(ax, prior_samples.β, bins = 50, color = :red, normalization = :pdf)
hist!(ax, new_samples.δ, bins = 50, color = :blue, normalization = :pdf)
hist!(ax, prior_samples.δ, bins = 50, color = :red, normalization = :pdf)
display(fig)

##

fig = Figure(size = size_pt, fontsize = 11, dpi = 300)
ax = Axis(fig[1, 1], xlabel = L"\textrm{time (days)}", ylabel = L"\textrm{viral load } (\log_{10})")
for id in eachindex(data)
    plot!(
        ax,
        data[id].obs_times .- true_infection_times[id],
        data[id].vl,
        color = :black,
        markersize = 6
    )
end
lines!(ax, post_sims_summ["t"], post_sims_summ["median"], color = :red)
band!(ax, post_sims_summ["t"], post_sims_summ["bottom"], post_sims_summ["top"], color = (:red, 0.4))
# lines!(ax, prior_sims_summ["t"], prior_sims_summ["median"], color = :green)
# band!(
#     ax,
#     prior_sims_summ["t"],
#     prior_sims_summ["bottom"],
#     prior_sims_summ["top"],
#     color = (:green, 0.4)
# )
xlims!(ax, 0, 21)
display(fig)

save(fig_loc * "predictive_plot_population.pdf", fig, pt_per_unit = 1.0)

##

fig = Figure(size = size_pt, fontsize = 12, dpi = 300)
axs = [
    Axis(fig[1, 1], ylabel = L"\textrm{viral load } (\log_{10})", xlabel = L"\textrm{time (days)}"),
    Axis(fig[1, 2], ylabel = L"\textrm{viral load } (\log_{10})", xlabel = L"\textrm{time (days)}")
]
for i in 1:20
    lines!(
        axs[1],
        prior_sims[:, 1],
        prior_sims[:, rand(2:size(prior_sims, 2))],
        color = :green,
        alpha = 0.4
    )
    lines!(
        axs[2],
        post_sims[:, 1],
        post_sims[:, rand(2:size(prior_sims, 2))],
        color = :red,
        alpha = 0.4
    )
end
xlims!(axs[1], 0, 21)
xlims!(axs[2], 0, 21)
# xlims!(axs[1], data[id].obs_times[1] - 1, data[id].obs_times[end] + 1)
# xlims!(axs[2], data[id].obs_times[1] - 1, data[id].obs_times[end] + 1)
display(fig)

save(fig_loc * "predictive_population_spaghetti.pdf", fig, pt_per_unit = 1.0)
