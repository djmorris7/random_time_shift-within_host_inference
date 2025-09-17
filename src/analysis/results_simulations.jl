include("../inference/within_host_inference.jl")
include("results.jl")
include("../plotting.jl")

##

Random.seed!(2023)

# Read in an example dataset
i = 1
(data, ids) = get_cleaned_data("data/sims/covid_data_clean_$i.csv")
df_true_pars = CSV.read(data_dir("sims/covid_parameters_$i.csv"), DataFrame)
df_true_hyper_pars = CSV.read(data_dir("sims/covid_hyper_parameters_$i.csv"), DataFrame)
true_infection_times = df_true_pars.infection_time

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

# Individual parameters (means)
μ_R₀ = df_true_hyper_pars[1, "μ_R₀"]
μ_k = df_true_hyper_pars[1, "μ_k"]
μ_δ = df_true_hyper_pars[1, "μ_δ"]
μ_πv = df_true_hyper_pars[1, "μ_πv"]
μ_c = df_true_hyper_pars[1, "μ_c"]
mean_pars = [μ_R₀, μ_k, μ_δ, μ_πv, μ_c]

κ = df_true_hyper_pars[1, "κ"]

##

# Setup and test the ODE solver
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
σ_R₀, σ_k, σ_δ, σ_πv, σ_c = df_true_hyper_pars[1, [:σ_R₀, :σ_k, :σ_δ, :σ_πv, :σ_c]]

##

# Get the posterior samples
df_samples = [
    CSV.read(results_dir("sim_samples/dataset_$i/samples_$j.csv"), DataFrame) for j in 1:3
]
df_samples = vcat(df_samples...)
samples = Matrix(df_samples)

df = CSV.read(data_dir("sims/covid_parameters_$i.csv"), DataFrame)
select!(df, [:ID, :R₀, :k, :δ, :πv, :c, :infection_time])

##

# Test whether we can sample a trajectory for one individual
id = 1
df_samples_ind = get_df_samples_ind(df_samples, id)
κ_post = df_samples[:, :κ]

T = 100
n_sims = 1000
post_sims = ppc_simulation(df_samples_ind, Z0_bp, nn, prob, κ_post, n_sims, T; t0 = -20, Δt = 0.1)
post_sims_summ = summarise_ppc_sims(post_sims)

fig = Figure()
ax = Axis(fig[1, 1])
for i in 1:20
    lines!(ax, post_sims[:, 1], post_sims[:, rand(2:size(post_sims, 2))], color = :red, alpha = 0.4)
end
display(fig)

## --- Sample a few people at onece ---

Random.seed!(2024)

size_inches = (7.25, 4.0)
size_pt = size_inches .* inch
fig = Figure(size = size_pt, fontsize = fontsize, dpi = dpi, linewidth = 1)

row = 1
col = 1

Random.seed!(2025)

ids = sample(1:100, 8, replace = false)

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

    if id ∈ ids
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
    xlims!(ax, -8, 13)

    ax.yticks = 0:3:9
    ylims!(ax, low = 0)

    row += 1
    if row > 2
        row = 1
        col += 1
    end
    ylims!(ax, low = 2.5)
end

Label(fig[1:2, 0], L"\log_{10}(\textrm{viral load})", rotation = pi / 2)
Label(fig[3, 1:4], L"\textrm{time (days) since peak VL}")

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

resize_to_layout!(fig)

display(fig)

save(fig_loc * "predictive_plot_multi_individuals.png", fig, px_per_unit = dpi / inch)
save(fig_loc * "predictive_plot_multi_individuals.pdf", fig)
