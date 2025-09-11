include("results.jl")
include("../plotting.jl")

##

fig_loc = "figures/simulation/"

posterior_summaries = DataFrame(
    "parameter" => [],
    "simulation_no" => Int[],
    "median" => Float64[],
    "mean" => Float64[],
    "0.025" => Float64[],
    "0.975" => Float64[],
    "std" => Float64[]
)

hyper_param_symbols = [:μ_R₀, :σ_R₀, :μ_δ, :σ_δ, :μ_πv, :σ_πv, :κ]

@showprogress for n in 1:50
    loc = results_dir("sim_samples/dataset_$n/")

    df = vcat([CSV.read(loc * "samples_$i.csv", DataFrame) for i in 1:3]...)
    # df = CSV.read(loc * "samples_$n.csv", DataFrame)[5000:end, :]

    # TODO: this needs to exist basically everywhere.
    df.σ_R₀ = exp.(df.σ_R₀)
    df.σ_δ = exp.(df.σ_δ)
    df.σ_πv = exp.(df.σ_πv)

    for sym in hyper_param_symbols
        v = @view df[:, sym]

        tmp_df = DataFrame(
            "parameter" => sym,
            "simulation_no" => n,
            "median" => median(v),
            "mean" => mean(v),
            "0.025" => quantile(v, 0.025),
            "0.975" => quantile(v, 0.975),
            "std" => std(v)
        )
        append!(posterior_summaries, tmp_df)
    end
end

CSV.write(results_dir("sim_samples/all_sample_summaries.csv"), posterior_summaries)

##

posterior_summaries = CSV.read(results_dir("sim_samples/all_sample_summaries.csv"), DataFrame)

true_param_values = CSV.read(data_dir("sims/covid_hyper_parameters_1.csv"), DataFrame)

##

# 1) Build a lookup Dict from the truth row (keyed by the *string* column names)
truth_map = Dict(string(n) => true_param_values[1, n] for n in names(true_param_values))

# 2) Attach the matching true value for each parameter (missing if not present in truth_map)
posterior_summaries[:, :true_value] = [
    get(truth_map, string(p), missing) for p in posterior_summaries.parameter
]

##

using CairoMakie, DataFrames

# Example: posterior_summaries :: DataFrame
# columns: :parameter, :simulation_no, :0.025, :0.975
# truths :: 1-row DataFrame with true values per parameter

n_params = 7
n_rows = 3
n_cols = 3

# Create the figure with enough panels (will leave some empty if needed)
size_inches = (7.5, 5.5)
size_pt = size_inches .* 72
fig = Figure(size = size_pt, fontsize = 10, dpi = 300, sharex = true, sharey = true, linewidth = 1)
# Map true values
truth_map = Dict(string(n) => true_param_values[1, n] for n in names(true_param_values))

param_map = Dict(
    :μ_R₀ => L"$\mu_{R_0}$",
    :σ_R₀ => L"$\sigma_{R_0}$",
    :μ_δ => L"$\mu_{\delta}$",
    :σ_δ => L"$\sigma_{\delta}$",
    :μ_πv => L"$\mu_{\rho}$",
    :σ_πv => L"$\sigma_{\rho}$",
    :κ => L"$\kappa$"
)

for (i, param) in enumerate(hyper_param_symbols)
    row, col = divrem(i - 1, n_cols) .+ 1
    ax = Axis(fig[row, col]; ylabel = param_map[param], ax_kwargs...)

    # Filter simulations for this parameter
    df = filter(r -> r.parameter == string(param), posterior_summaries)

    xs = df.simulation_no
    meds = df[!, "median"]
    ys_low = df[!, "0.025"]
    ys_high = df[!, "0.975"]

    # Plot 95% CI as vertical lines
    for (x, l, h) in zip(xs, ys_low, ys_high)
        lines!(ax, [x, x], [l, h], color = colors[1], linewidth = 1)
    end
    scatter!(ax, xs, meds, color = colors[1], markersize = 4)

    # Plot the true value as horizontal red line
    hlines!(ax, truth_map[String(param)], color = :red, linestyle = :dash)
end

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

resize_to_layout!(fig)

Label(fig[4, 1:3], "Simulation number")

fig

save(fig_loc * "sim_posteriors_coverage.pdf", fig, pt_per_unit = 1.0)

##

posterior_summaries.covers =
    (posterior_summaries.true_value .>= posterior_summaries[:, "0.025"]) .&&
    (posterior_summaries.true_value .<= posterior_summaries[:, "0.975"])

coverage = Dict(s => 0.0 for s in hyper_param_symbols)
interval_widths_avg = Dict(s => 0.0 for s in hyper_param_symbols)
interval_widths_ci = Dict(s => [Inf, Inf] for s in hyper_param_symbols)
biases = Dict(s => 0.0 for s in hyper_param_symbols)
biases_ci = Dict(s => [Inf, Inf] for s in hyper_param_symbols)

for (k, v) in coverage
    df_tmp = filter(r -> r.parameter == string(k), posterior_summaries)
    coverage[k] = sum(df_tmp.covers) / nrow(df_tmp)
    widths = df_tmp[:, "0.975"] .- df_tmp[:, "0.025"]
    interval_widths_avg[k] = mean(widths)
    interval_widths_ci[k] .= round.(quantile(widths, (0.1, 0.9)), digits = 3)

    biases_tmp = (df_tmp.mean - df_tmp.true_value[1]) / df_tmp.true_value[1]
    biases[k] = mean(biases_tmp)
    biases_ci[k] .= round.(quantile(biases_tmp, (0.1, 0.9)), digits = 3)
end

##

coverage
interval_widths_avg
interval_widths_ci
biases
biases_ci
