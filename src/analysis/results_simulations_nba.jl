include("../inference/within_host_inference.jl")
include("results.jl")
include("../plotting.jl")

##

Random.seed!(2023)
(data, id_mapping) = get_cleaned_data("data/nba/nba_data_clean.csv")

N = length(data)

df_true_pars = CSV.read(data_dir("sims/parameters.csv"), DataFrame)
df_true_hyper_pars = CSV.read(data_dir("sims/hyper_parameters.csv"), DataFrame)

fig_loc = "figures/"
if isdir(fig_loc) == false
    mkdir(fig_loc)
end

##

# Initial conditions
S0 = Int(8e7)
E0 = 1
I0 = 0
V0 = 0
Z0 = [S0 - (E0 + I0), E0, I0, V0]
Z0_bp = Z0[2:end]

# Individual parameters (means) to set up the ODE problem
μ_R₀ = df_true_hyper_pars[1, "μ_R₀"]
μ_k = df_true_hyper_pars[1, "μ_k"]
μ_δ = df_true_hyper_pars[1, "μ_δ"]
μ_πv = df_true_hyper_pars[1, "μ_πv"]
μ_c = df_true_hyper_pars[1, "μ_c"]
mean_pars = [μ_R₀, μ_k, μ_δ, μ_πv, μ_c]

κ = df_true_hyper_pars[1, "κ"]

##

# Initialise ODE solver to check it works
obs_t = 50
tspan = (0, 50)

# test_pars
prob = ODEProblem(tcl_deterministic!, Z0, tspan, [8.0; mean_pars[2:end]])
sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, save_idxs = 4)

plot(sol)

##

# Other bits for simulator (i.e. neural net)

pars0 = deepcopy(mean_pars)
nn = load_nn()
LOD = ct_to_vl(40.0)
σ_R₀, σ_k, σ_δ, σ_πv, σ_c = df_true_hyper_pars[1, [:σ_R₀, :σ_k, :σ_δ, :σ_πv, :σ_c]]

##

# Read in posterior samples
df_samples = [CSV.read(results_dir("samples_nba_$i.csv"), DataFrame) for i in 1:3]
df_samples = vcat(df_samples...)
samples = Matrix(df_samples)

##

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

# Test param getter and simulator for a single individual
id = 1
df_samples_ind = get_df_samples_ind(df_samples, id)
κ_post = df_samples[:, :κ]

T = 100
n_sims = 3000
post_sims = ppc_simulation(df_samples_ind, Z0_bp, nn, prob, κ_post, n_sims, T; t0 = -20, Δt = 0.1)
post_sims_summ = summarise_ppc_sims(post_sims)

## --- Sample multiple people at once ---

# Transcribed parameters from Zitzmann et al 2024 Supporting Information
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

# Solve the Zitzmann et al models
for (i, id) in enumerate(ids_zitzmann)
    Z0 = [S0 - (E0 + I0), E0, I0, V0]
    R₀ = βs_zitz[i] * πs_zitz[i] * S0 / (δs_zitz[i] * c)
    tspan = (infection_times_zitz[i], infection_times_zitz[i] + 30)

    prob = ODEProblem(tcl_deterministic!, Z0, tspan, [R₀, k, δs_zitz[i], πs_zitz[i], c])
    sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, save_idxs = 4, saveat = 1.0)
    sols[id] = sol
end

##

# Get the zitzmann ids that are also in the nba data and keep the first 4
ids_pivotal = findall(x -> x in ids_zitzmann, id_mapping)
setdiff(ids_zitzmann, id_mapping[ids_pivotal])
ids_zitz_pivotal = deepcopy(ids_pivotal)
ids = ids_zitz_pivotal[1:4]

# Add some individuals we hand picked with less data
push!(ids, 158)
push!(ids, 152)
push!(ids, 150)
push!(ids, 125)

##

size_inches = (7.25, 4)
size_pt = size_inches .* inch
fig = Figure(size = size_pt, fontsize = fontsize, dpi = dpi, linewidth = 1.5)

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
        lines!(
            ax,
            sols[id_mapping[id]].t,
            log10p0.(sols[id_mapping[id]].u),
            color = :red,
            linestyle = :dash
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

save(fig_loc * "predictive_plot_multi_individuals_nba.png", fig, px_per_unit = dpi / inch)
save(fig_loc * "predictive_plot_multi_individuals_nba.pdf", fig)
