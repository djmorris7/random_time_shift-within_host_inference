include("../inference/within_host_inference.jl")
include("results.jl")
include("../plotting.jl")

##

Random.seed!(2023)

(data, ids_mapping) = get_cleaned_data("data/nba/nba_data_clean.csv")

N = length(data)

fig_loc = "figures/"
if isdir(fig_loc) == false
    mkdir(fig_loc)
end

##

df_true_hyper_pars = CSV.read(data_dir("sims/covid_hyper_parameters_1.csv"), DataFrame)

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
prob = ODEProblem(tcl_deterministic!, Z0, tspan, [mean_pars[1] / S0; mean_pars[2:end]])
sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, save_idxs = 4)

##

pars0 = deepcopy(mean_pars)

S0 = Z0[1]
Z0_bp = Z0[2:end]

integrator = f -> quadgk(f, -7, 7)[1]

σ_R₀, σ_k, σ_δ, σ_πv, σ_c = df_true_hyper_pars[1, [:σ_R₀, :σ_k, :σ_δ, :σ_πv, :σ_c]]

##

df_samples = [CSV.read(results_dir("samples_nba_$i.csv"), DataFrame) for i in 1:3]
df_samples = vcat(df_samples...)

function summarise_samples(df_samples)
    μ_R₀ = mean(df_samples.μ_R₀)
    σ_R₀ = mean(df_samples.σ_R₀)
    μ_δ = mean(df_samples.μ_δ)
    σ_δ = mean(df_samples.σ_δ)
    μ_πv = mean(df_samples.μ_πv)
    σ_πv = mean(df_samples.σ_πv)
    κ = mean(df_samples.κ)

    μ_R₀_cri = quantile(df_samples.μ_R₀, [0.025, 0.975])
    σ_R₀_cri = quantile(df_samples.σ_R₀, [0.025, 0.975])
    μ_δ_cri = quantile(df_samples.μ_δ, [0.025, 0.975])
    σ_δ_cri = quantile(df_samples.σ_δ, [0.025, 0.975])
    μ_πv_cri = quantile(df_samples.μ_πv, [0.025, 0.975])
    σ_πv_cri = quantile(df_samples.σ_πv, [0.025, 0.975])
    κ_cri = quantile(df_samples.κ, [0.025, 0.975])

    # print them out nicely in a table
    println("Parameter means:")
    println("μ_R₀: $μ_R₀", " (95% CI: ", μ_R₀_cri[1], ", ", μ_R₀_cri[2], ")")
    println("σ_R₀: $σ_R₀", " (95% CI: ", σ_R₀_cri[1], ", ", σ_R₀_cri[2], ")")
    println("μ_δ: $μ_δ", " (95% CI: ", μ_δ_cri[1], ", ", μ_δ_cri[2], ")")
    println("σ_δ: $σ_δ", " (95% CI: ", σ_δ_cri[1], ", ", σ_δ_cri[2], ")")
    println("μ_πv: $μ_πv", " (95% CI: ", μ_πv_cri[1], ", ", μ_πv_cri[2], ")")
    println("σ_πv: $σ_πv", " (95% CI: ", σ_πv_cri[1], ", ", σ_πv_cri[2], ")")
    println("κ: $κ", " (95% CI: ", κ_cri[1], ", ", κ_cri[2], ")")

    return nothing
end

df = CSV.read(data_dir("sims/covid_parameters_1.csv"), DataFrame)
select!(df, [:ID, :R₀, :k, :δ, :πv, :c, :infection_time])

##

size_inches = (7.5, 4.0)
size_pt = size_inches .* inch
fig = Figure(size = size_pt, fontsize = fontsize, dpi = dpi)

ax = Axis(fig[1, 1]; ax_kwargs...)
hist!(
    ax,
    df_samples.μ_R₀,
    bins = 20,
    normalization = :pdf,
    color = (colors[1], 0.3),
    strokewidth = 0.7,
    strokecolor = :black
)
plot!(ax, hyper_priors[:μ_R₀], color = colors[2])
# vlines!(ax, [μ_R₀], color = :black, linestyle = :dash)
# vlines!(ax, [true_pars_sample_vals["μ_R₀"]], color = :black, linestyle = :dash)
xlims!(ax, 0.9 * minimum(df_samples.μ_R₀), maximum(df_samples.μ_R₀) * 1.1)
ylims!(ax, low = 0.0)
ax.xlabel = L"\mu_{R_0}"
ax.xticks = 10:3:25

ax = Axis(fig[2, 1]; ax_kwargs...)
# stephist!(ax, df_samples.μ_πv, bins = 20, normalization = :pdf, color = colors[1])
hist!(
    ax,
    df_samples.μ_πv,
    bins = 20,
    normalization = :pdf,
    color = (colors[1], 0.3),
    strokewidth = 0.7,
    strokecolor = :black
)
plot!(ax, hyper_priors[:μ_πv], color = colors[2])
# vlines!(ax, [μ_πv], color = :black, linestyle = :dash)
# vlines!(ax, [true_pars_sample_vals["μ_πv"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.μ_πv), maximum(df_samples.μ_πv) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"\mu_{\rho}"

ax = Axis(fig[1, 2]; ax_kwargs...)
# stephist!(ax, df_samples.μ_δ, bins = 20, normalization = :pdf, color = colors[1])
hist!(
    ax,
    df_samples.μ_δ,
    bins = 20,
    normalization = :pdf,
    color = (colors[1], 0.3),
    strokewidth = 0.7,
    strokecolor = :black
)
plot!(ax, hyper_priors[:μ_δ], color = colors[2])
# vlines!(ax, [μ_δ], color = :black, linestyle = :dash)
# vlines!(ax, [true_pars_sample_vals["μ_δ"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.μ_δ), maximum(df_samples.μ_δ) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"\mu_{\delta}"

ax = Axis(fig[2, 2]; ax_kwargs...)
# stephist!(ax, df_samples.σ_δ, bins = 20, normalization = :pdf, color = colors[1])
hist!(
    ax,
    df_samples.σ_δ,
    bins = 20,
    normalization = :pdf,
    color = (colors[1], 0.3),
    strokewidth = 0.7,
    strokecolor = :black
)
plot!(ax, hyper_priors[:σ_δ], color = colors[2])
# vlines!(ax, [σ_δ], color = :black, linestyle = :dash)
# vlines!(ax, [true_pars_sample_vals["σ_δ"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.σ_δ), maximum(df_samples.σ_δ) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"\sigma_{\delta}"

ax = Axis(fig[1, 3]; ax_kwargs...)
# stephist!(ax, df_samples.σ_R₀, bins = 20, normalization = :pdf, color = colors[1])
hist!(
    ax,
    df_samples.σ_R₀,
    bins = 20,
    normalization = :pdf,
    color = (colors[1], 0.3),
    strokewidth = 0.7,
    strokecolor = :black
)
plot!(ax, hyper_priors[:σ_R₀], color = colors[2])
# vlines!(ax, [true_pars_sample_vals["σ_R₀"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.σ_R₀), maximum(df_samples.σ_R₀) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"σ_{R_0}"

ax = Axis(fig[2, 3]; ax_kwargs...)
# stephist!(ax, df_samples.σ_πv, bins = 20, normalization = :pdf, color = colors[1])
hist!(
    ax,
    df_samples.σ_πv,
    bins = 20,
    normalization = :pdf,
    color = (colors[1], 0.3),
    strokewidth = 0.7,
    strokecolor = :black
)
plot!(ax, hyper_priors[:σ_πv], color = colors[2])
# vlines!(ax, [true_pars_sample_vals["σ_πv"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.σ_πv), maximum(df_samples.σ_πv) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"σ_{\rho}"

ax = Axis(fig[1, 4]; ax_kwargs...)
# stephist!(ax, df_samples.κ, bins = 20, normalization = :pdf, color = colors[1])
hist!(
    ax,
    df_samples.κ,
    bins = 20,
    normalization = :pdf,
    color = (colors[1], 0.3),
    strokewidth = 0.7,
    strokecolor = :black
)
plot!(ax, hyper_priors[:κ], color = colors[2])
# vlines!(ax, [κ], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.κ), maximum(df_samples.κ) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"\kappa"

resize_to_layout!(fig)

Label(fig[1:2, 0], "density", rotation = pi / 2)

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

resize_to_layout!(fig)

display(fig)

save(fig_loc * "nba_hyper_pars_posteriors.png", fig, px_per_unit = dpi / inch)
save(fig_loc * "nba_hyper_pars_posteriors.pdf", fig)
