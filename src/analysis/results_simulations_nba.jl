include("../inference/within_host_inference.jl")
include("results.jl")
include("../plotting.jl")

##

Random.seed!(2023)
(data, id_mapping) = get_cleaned_data("data/nba/nba_data_clean.csv")

# scatter(data[N].obs_times, data[N].vl, color = :black)

N = length(data)

df_true_pars = CSV.read(data_dir("sims/parameters.csv"), DataFrame)
df_true_hyper_pars = CSV.read(data_dir("sims/hyper_parameters.csv"), DataFrame)

fig_loc = "figures/nba/"
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

# test_pars
prob = ODEProblem(tcl_deterministic!, Z0, tspan, [8.0; mean_pars[2:end]])
sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, save_idxs = 4)

plot(sol)

##

pars0 = deepcopy(mean_pars)
nn = load_nn()

## Testing the likelihood for a single individual
S0 = Z0[1]
Z0_bp = Z0[2:end]
LOD = ct_to_vl(40.0)

## Testing the likelihood for a single individual

σ_R₀, σ_k, σ_δ, σ_πv, σ_c = df_true_hyper_pars[1, [:σ_R₀, :σ_k, :σ_δ, :σ_πv, :σ_c]]

##

infection_time_ranges = [zeros(2) for _ in range(1, N)]
for (i, dat) in enumerate(data)
    obs_peak_timing = dat.obs_times[findmax(dat.vl)[2]]

    earliest_timing = obs_peak_timing - 20
    latest_timing = obs_peak_timing + 10

    infection_time_ranges[i] = [earliest_timing, latest_timing]
end

##

# df_samples = CSV.read(results_dir("samples_nba.csv"), DataFrame)
df_samples = [CSV.read(results_dir("samples_nba_$i.csv"), DataFrame) for i in 1:4]
burnin = 10000
thin = 10
df_samples = [df_samples[i][burnin:thin:end, :] for i in 1:4]
df_samples = vcat(df_samples...)
samples = Matrix(df_samples)
# samples = Matrix(df_samples)

## --- Sample new realisations for a known individual ---

# id = 18
id = 159
id = 125
# id = 14

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

# get params
# df_samples_ind = extract_individual_params(df_samples, id)
df_samples_ind = get_df_samples_ind(df_samples, id)
κ_post = df_samples[:, :κ]

T = 100
n_sims = 3000
# df_samples_ind.πv .= 3.0
# df_samples_ind.δ .= 1.5
# df_samples_ind.πv *
post_sims = ppc_simulation(df_samples_ind, Z0_bp, nn, prob, κ_post, n_sims, T; t0 = -20, Δt = 0.1)

post_sims_summ = summarise_ppc_sims(post_sims)

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, data[id].obs_times, data[id].vl, color = :black)
lines!(ax, post_sims_summ["t"], post_sims_summ["median"], color = :red)
band!(
    ax, post_sims_summ["t"], post_sims_summ["lower"], post_sims_summ["upper"], color = (:red, 0.4)
)
band!(ax, post_sims_summ["t"], post_sims_summ["bottom"], post_sims_summ["top"], color = (:red, 0.4))
# lines!(ax, prior_sims_summ["t"], prior_sims_summ["median"], color = :green)
# band!(
#     ax,
#     prior_sims_summ["t"],
#     prior_sims_summ["lower"],
#     prior_sims_summ["upper"],
#     color = (:green, 0.4)
# )
# xlims!(ax, data[id].obs_times[1] - 1, data[id].obs_times[end] + 1)
xlims!(ax, -14, 14)
display(fig)

hist(df_samples_ind.infection_time, bins = 50)

##

nsamples_prior = 10_00
prior_samples = zeros(nsamples_prior, 6)

j = 1

for (i, symbol) in enumerate(fieldnames(Params))
    symbol ∈ [:infection_time_range, :z_R₀, :z_δ, :z_πv] && continue

    if symbol == :infection_time
        prior_samples[:, j] = rand(Uniform(infection_time_ranges[id]...), nsamples_prior)
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

κ_prior = rand(hyper_priors[:κ], nsamples_prior)

prior_samples = DataFrame(prior_samples, [:R₀, :k, :δ, :πv, :c, :infection_time])

prior_sims = ppc_simulation(prior_samples, Z0_bp, nn, prob, κ_prior, n_sims, T; t0 = -20, Δt = 0.1)

##

post_sims_summ = summarise_ppc_sims(post_sims)
prior_sims_summ = summarise_ppc_sims(prior_sims)

##

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, data[id].obs_times, data[id].vl, color = :black)
lines!(ax, post_sims_summ["t"], post_sims_summ["median"], color = :red)
band!(
    ax, post_sims_summ["t"], post_sims_summ["lower"], post_sims_summ["upper"], color = (:red, 0.4)
)
band!(ax, post_sims_summ["t"], post_sims_summ["bottom"], post_sims_summ["top"], color = (:red, 0.4))
# lines!(ax, prior_sims_summ["t"], prior_sims_summ["median"], color = :green)
# band!(
#     ax,
#     prior_sims_summ["t"],
#     prior_sims_summ["lower"],
#     prior_sims_summ["upper"],
#     color = (:green, 0.4)
# )
# xlims!(ax, data[id].obs_times[1] - 1, data[id].obs_times[end] + 1)
display(fig)

##

# Spaghetti plot

# Random.seed!(100)

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

δs_zitz = [
    2.05
    0.99
    1.42
    1.26
    1.3
    1.01
    1.08
    1.33
    1.25
    1.29
    1.89
    0.91
    0.89
    1.03
    1.44
    1.36
    1.78
    1.33
    1.88
    1.58
    1.72
    1.18
    1.28
    0.78
    1
]

πs_zitz = [
    3.09
    3.06
    3.09
    3.06
    3.07
    3.08
    3.06
    3.05
    3.08
    3.08
    3.03
    3.06
    3.05
    3.07
    3.07
    3.05
    3.05
    3.08
    3.08
    3.07
    3.08
    3.11
    3.06
    3.09
    3.06
]

βs_zitz =
    10 .^ [
        -6.36
        -6.37
        -6.36
        -6.37
        -6.37
        -6.37
        -6.37
        -6.38
        -6.36
        -6.36
        -6.38
        -6.37
        -6.37
        -6.37
        -6.37
        -6.38
        -6.37
        -6.37
        -6.36
        -6.37
        -6.36
        -6.36
        -6.37
        -6.36
        -6.37
    ]

infection_times_zitz = [
    -7.3
    -6.7
    -9.1
    -6.9
    -6.5
    -8.5
    -7.4
    -7.4
    -7.7
    -7.1
    -6.9
    -7.7
    -6.6
    -8.4
    -6.9
    -6.7
    -6.7
    -8.2
    -6.9
    -6.9
    -7.7
    -10.8
    -7.3
    -10.1
    -7.7
]

c = 10.0
k = 4.0

sols = Dict()
ids_zitzmann = [
    87,
    219,
    283,
    285,
    293,
    315,
    407,
    439,
    496,
    615,
    657,
    737,
    755,
    777,
    942,
    1273,
    1368,
    1375,
    1628,
    1647,
    1740,
    2349,
    2463,
    3485,
    3491
]
for (i, id) in enumerate(ids_zitzmann)
    Z0 = [S0 - (E0 + I0), E0, I0, V0]
    R₀ = βs_zitz[i] * πs_zitz[i] * S0 / (δs_zitz[i] * c)
    tspan = (infection_times_zitz[i], infection_times_zitz[i] + 30)

    prob = ODEProblem(tcl_deterministic!, Z0, tspan, [R₀, k, δs_zitz[i], πs_zitz[i], c])
    sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, save_idxs = 4, saveat = 1.0)
    sols[id] = sol
end

# plot(sols[1].t, log10p0.(sols[1].u))

##

for (i, d) in enumerate(data)
    if length(d.vl .> LOD) <= 6
        println(i)
    end
end

##

# ids_missing = findall(x -> x ∉ ids_zitzmann)

ids_pivotal = findall(x -> x in ids_zitzmann, id_mapping)

setdiff(ids_zitzmann, id_mapping[ids_pivotal])

ids_zitz_pivotal = deepcopy(ids_pivotal)
# while length(ids) < 20
#     id_test = rand(1:N)
#     if id_test ∉ ids
#         push!(ids, id_test)
#     end
# end

ids = ids_zitz_pivotal[1:4]
push!(ids, 158)
push!(ids, 152)
push!(ids, 150)
push!(ids, 125)
# push!(ids, 122)
# push!(ids, 116)

##

for (i, D) in enumerate(data)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "{i} - $(i)")
    plot!(ax, D.obs_times, D.vl, color = :black)

    display(fig)
end

##

size_inches = (7.5, 4.5)
size_pt = size_inches .* 72
fig = Figure(size = size_pt, fontsize = 10, dpi = 300, linewidth = 1)

(row, col) = (1, 1)

j = 1
@showprogress for id in ids
    ax = Axis(fig[row, col]; ax_kwargs...)

    df_samples_ind = get_df_samples_ind(df_samples, id)
    κ_post = df_samples[:, :κ]

    T = 100
    n_sims = 3000
    post_sims = ppc_simulation(
        df_samples_ind, Z0_bp, nn, prob, κ_post, n_sims, T; t0 = -20, Δt = 0.1
    )

    # prior_sims_summ = summarise_ppc_sims(prior_sims)
    post_sims_summ = summarise_ppc_sims(post_sims)

    lines!(ax, post_sims_summ["t"], post_sims_summ["median"], color = colors[1])
    band!(
        ax,
        post_sims_summ["t"],
        post_sims_summ["lower"],
        post_sims_summ["upper"],
        color = (colors[1], 0.3)
    )
    band!(
        ax,
        post_sims_summ["t"],
        post_sims_summ["bottom"],
        post_sims_summ["top"],
        color = (colors[1], 0.2)
    )

    if id_mapping[id] ∈ keys(sols)
        scatter!(
            ax,
            sols[id_mapping[id]].t,
            log10p0.(sols[id_mapping[id]].u),
            color = colors[2],
            # linestyle = :dash
            marker = :cross,
            markersize = 4
        )
    end
    j += 1

    plot!(ax, data[id].obs_times, data[id].vl, color = :black, markersize = 4)
    text!(
        ax,
        13,
        8,
        text = "ID: $(id_mapping[id])",
        fontsize = 8,
        justification = :center,
        align = (:center, :bottom)
    )

    # ax.title = "ID: $(id_mapping[id])"

    ylims!(ax, low = 0)

    low = min(minimum(data[id].obs_times) - 1, -10)
    xlims!(ax, low = low, high = 21)

    ylims!(ax, low = 0.95 * LOD)

    col += 1
    if col > 4
        row += 1
        col = 1
    end
end

Label(fig[1:2, 0], L"\log_{10}(\textrm{viral load})", rotation = pi / 2)
Label(fig[3, 1:4], L"\textrm{time (days) since peak VL}")

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

display(fig)

save(fig_loc * "predictive_plot_multi_individuals_nba.pdf", fig, pt_per_unit = 1.0)

##

a = 1
n_ax = 1

@showprogress for id in eachindex(data)
    if mod(id, 25) == 0
        save(fig_loc * "predictive_plot_multi_individuals_$a.pdf", fig, pt_per_unit = 1.0)
        fig = Figure(size = (575, 575), fontsize = 12, dpi = 300)
        axs = [Axis(fig[i, j]) for i in 1:5, j in 1:5]
        a += 1
        n_ax = 1
    end

    df_samples_ind = extract_individual_params(df_samples, id)
    κ_post = df_samples[:, :κ]

    T = 100
    n_sims = 500
    post_sims = ppc_simulation(
        df_samples_ind, Z0_bp, nn, prob, κ_post, n_sims, T; t0 = -20, Δt = 0.1
    )

    # nsamples_prior = 1000
    # prior_samples = zeros(nsamples_prior, 6)

    # for (i, symbol) in enumerate(fieldnames(Params))
    #     symbol ∈ [:infection_time_range] && continue

    #     if symbol == :infection_time
    #         prior_samples[:, i] = rand(
    #             Uniform(infection_time_ranges[id]...), nsamples_prior
    #         )
    #     elseif symbol == :k
    #         prior_samples[:, i] .= μ_k
    #     elseif symbol == :c
    #         prior_samples[:, i] .= μ_c
    #     else
    #         μ = rand(hyper_priors[string_to_symbol("μ_", symbol)], nsamples_prior)
    #         if symbol ∈ [:β, :k, :δ, :πv, :c]
    #             σ = df_true_hyper_pars[1, "σ_" * string(symbol)]
    #         else
    #             σ = rand(hyper_priors[string_to_symbol("σ_", symbol)], nsamples_prior)
    #         end
    #         prior_samples[:, i] = rand.(Truncated.(Normal.(μ, σ), 0.01, 100))
    #     end
    # end

    # κ_prior = rand(hyper_priors[:κ], nsamples_prior)

    # prior_sims = ppc_simulation(
    #     prior_samples, Z0_bp, nn_func, prob, κ_prior, n_sims, T; t0 = -20, Δt = 0.1
    # )

    # prior_sims_summ = summarise_ppc_sims(prior_sims)
    post_sims_summ = summarise_ppc_sims(post_sims)

    ax = axs[n_ax]

    plot!(ax, data[id].obs_times, data[id].vl, color = :black)
    lines!(ax, post_sims_summ["t"], post_sims_summ["median"], color = :red)
    band!(
        ax,
        post_sims_summ["t"],
        post_sims_summ["lower"],
        post_sims_summ["upper"],
        color = (:red, 0.4)
    )
    band!(
        ax,
        post_sims_summ["t"],
        post_sims_summ["bottom"],
        post_sims_summ["top"],
        color = (:red, 0.4)
    )
    # lines!(ax, prior_sims_summ["t"], prior_sims_summ["median"], color = :green)
    # band!(
    #     ax,
    #     prior_sims_summ["t"],
    #     prior_sims_summ["lower"],
    #     prior_sims_summ["upper"],
    #     color = (:green, 0.4)
    # )

    min_t = min(-10, floor(data[id].obs_times[1] - 2))
    max_t = max(14, ceil(data[id].obs_times[end] + 2))
    # xlims!(ax, min_t, max_t)
    xlims!(ax, -14, 14)

    n_ax += 1
end

##

@showprogress for (id, ax) in zip(ids, axs)
    df_samples_ind = extract_individual_params(df_samples, id)
    κ_post = df_samples[:, :κ]

    T = 100
    n_sims = 3000
    post_sims = ppc_simulation(
        df_samples_ind, Z0_bp, nn, prob, κ_post, n_sims, T; t0 = -20, Δt = 0.1
    )

    # nsamples_prior = 1000
    # prior_samples = zeros(nsamples_prior, 6)

    # for (i, symbol) in enumerate(fieldnames(Params))
    #     symbol ∈ [:infection_time_range] && continue

    #     if symbol == :infection_time
    #         prior_samples[:, i] = rand(
    #             Uniform(infection_time_ranges[id]...), nsamples_prior
    #         )
    #     elseif symbol == :k
    #         prior_samples[:, i] .= μ_k
    #     elseif symbol == :c
    #         prior_samples[:, i] .= μ_c
    #     else
    #         μ = rand(hyper_priors[string_to_symbol("μ_", symbol)], nsamples_prior)
    #         if symbol ∈ [:β, :k, :δ, :πv, :c]
    #             σ = df_true_hyper_pars[1, "σ_" * string(symbol)]
    #         else
    #             σ = rand(hyper_priors[string_to_symbol("σ_", symbol)], nsamples_prior)
    #         end
    #         prior_samples[:, i] = rand.(Truncated.(Normal.(μ, σ), 0.01, 100))
    #     end
    # end

    # κ_prior = rand(hyper_priors[:κ], nsamples_prior)

    # prior_sims = ppc_simulation(
    #     prior_samples, Z0_bp, nn_func, prob, κ_prior, n_sims, T; t0 = -20, Δt = 0.1
    # )

    # prior_sims_summ = summarise_ppc_sims(prior_sims)
    post_sims_summ = summarise_ppc_sims(post_sims)

    plot!(ax, data[id].obs_times, data[id].vl, color = :black)
    lines!(ax, post_sims_summ["t"], post_sims_summ["median"], color = :red)
    band!(
        ax,
        post_sims_summ["t"],
        post_sims_summ["lower"],
        post_sims_summ["upper"],
        color = (:red, 0.4)
    )
    band!(
        ax,
        post_sims_summ["t"],
        post_sims_summ["bottom"],
        post_sims_summ["top"],
        color = (:red, 0.4)
    )
    # lines!(ax, prior_sims_summ["t"], prior_sims_summ["median"], color = :green)
    # band!(
    #     ax,
    #     prior_sims_summ["t"],
    #     prior_sims_summ["lower"],
    #     prior_sims_summ["upper"],
    #     color = (:green, 0.4)
    # )

    min_t = min(-10, floor(data[id].obs_times[1] - 2))
    max_t = max(14, ceil(data[id].obs_times[end] + 2))
    # xlims!(ax, min_t, max_t)
    xlims!(ax, -14, 14)
end

display(fig)

save(fig_loc * "predictive_plot_multi_individuals.pdf", fig, pt_per_unit = 1.0)

## --- Now do some sampling using the hierarchical part of the model ---

# Now sample completely new individuals using the hyper parameters
df_samples_joint = df_samples[
    1:2:end, [:μ_β, :σ_β, :μ_k, :σ_k, :μ_δ, :σ_δ, :μ_πv, :σ_πv, :μ_c, :σ_c, :κ]
]

# now we need to draw new samples
nsamples = 10_000
new_samples = zeros(nsamples, 6)

j = 1
for (i, symbol) in enumerate(fieldnames(Params))
    symbol ∈ [:infection_time_range, :z_β, :z_δ, :z_πv] && continue

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

new_samples = DataFrame(new_samples, [:β, :k, :δ, :πv, :c, :infection_time])

κ_post = df_samples[:, :κ]

##

post_sims = ppc_simulation(new_samples, Z0_bp, nn, prob, κ_post, 5000, 100; Δt = 0.1)

post_sims_summ = summarise_ppc_sims(post_sims)

prior_samples = zeros(nsamples, 6)

j = 1
for (i, symbol) in enumerate(fieldnames(Params))
    symbol ∈ [:infection_time_range, :z_β, :z_δ, :z_πv] && continue

    if symbol == :infection_time
        prior_samples[:, j] .= 0.0
        # elseif symbol == :β
        #     prior_samples[:, j] .= μ_β
    elseif symbol == :k
        prior_samples[:, j] .= μ_k
    elseif symbol == :c
        prior_samples[:, j] .= μ_c
    else
        μ = rand(hyper_priors[string_to_symbol("μ_", symbol)], nsamples)
        if symbol == :β
            σ = df_true_hyper_pars[1, "σ_β"]

        elseif symbol == :πv
            σ = df_true_hyper_pars[1, "σ_πv"]
        else
            σ = rand(hyper_priors[string_to_symbol("σ_", symbol)], nsamples)
        end
        prior_samples[:, j] = rand.(Truncated.(Normal.(μ, σ), 0.01, 100))
    end
    j += 1
end

prior_samples = DataFrame(prior_samples, [:β, :k, :δ, :πv, :c, :infection_time])
κ_prior = rand(hyper_priors[:κ], nsamples)

prior_sims = ppc_simulation(prior_samples, Z0_bp, nn, prob, κ_prior, 5000, 100; Δt = 0.1)
prior_sims_summ = summarise_ppc_sims(prior_sims)

##

size_inches = (5.5, 3.5)
size_pt = size_inches .* 72
fig = Figure(
    size = size_pt,
    fontsize = 11,
    dpi = 300,
    sharey = true,
    xgridvisible = false,
    ygridvisible = false
)
ax = Axis(fig[1, 1], ylabel = L"\textrm{viral load } (\log_{10})", titlealign = :left)
lines!(ax, prior_sims_summ["t"], prior_sims_summ["median"], color = :dodgerblue)
band!(
    ax,
    prior_sims_summ["t"],
    prior_sims_summ["lower"],
    prior_sims_summ["upper"],
    color = (:dodgerblue, 0.2)
)
band!(
    ax,
    prior_sims_summ["t"],
    prior_sims_summ["bottom"],
    prior_sims_summ["top"],
    color = (:dodgerblue, 0.2)
)
xlims!(ax, 0, 21)
ax.title = "(A)"

ax = Axis(fig[1, 2], titlealign = :left)
lines!(ax, post_sims_summ["t"], post_sims_summ["median"], color = :red)
band!(
    ax, post_sims_summ["t"], post_sims_summ["lower"], post_sims_summ["upper"], color = (:red, 0.3)
)
band!(ax, post_sims_summ["t"], post_sims_summ["bottom"], post_sims_summ["top"], color = (:red, 0.3))
xlims!(ax, 0, 21)
ax.title = "(B)"
Label(fig[2, 1:2], "time (days) post infection")

display(fig)

save(fig_loc * "predictive_plot_population.pdf", fig, pt_per_unit = 1.0)

##

fig = Figure(resolution = size_pt, fontsize = 12, dpi = 300)
axs = [
    Axis(fig[1, 1], ylabel = L"\textrm{viral load } (\log_{10})", xlabel = L"\textrm{time (days)}"),
    Axis(fig[1, 2], ylabel = L"\textrm{viral load } (\log_{10})", xlabel = L"\textrm{time (days)}")
]
for i in 1:10
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

##
