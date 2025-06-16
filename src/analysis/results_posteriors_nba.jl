include("../inference/within_host_inference.jl")
include("results.jl")
include("../plotting.jl")

##

Random.seed!(2023)

(data, ids_mapping) = get_cleaned_data("data/nba/nba_data_clean.csv")

N = length(data)

fig_loc = "figures/nba/"
if isdir(fig_loc) == false
    mkdir(fig_loc)
end

##

df_true_hyper_pars = CSV.read(data_dir("sims/hyper_parameters.csv"), DataFrame)

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

df_samples = [CSV.read(results_dir("samples_nba_$i.csv"), DataFrame) for i in 1:4]
burnin = 10000
thin = 10
df_samples = [df_samples[i][burnin:thin:end, :] for i in 1:4]
df_samples = vcat(df_samples...)

df = CSV.read(data_dir("sims/parameters.csv"), DataFrame)
select!(df, [:ID, :R₀, :k, :δ, :πv, :c, :infection_time])

##

# size_inches = (7.5, 5.0)
size_inches = (7.5, 4.5)
size_pt = size_inches .* 72
fig = Figure(size = size_pt, fontsize = 10, dpi = 300)

ax = Axis(fig[1, 1]; ax_kwargs...)
stephist!(ax, df_samples.μ_R₀, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.μ_R₀, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:μ_R₀], color = colors[2])
# vlines!(ax, [μ_R₀], color = :black, linestyle = :dash)
# vlines!(ax, [true_pars_sample_vals["μ_R₀"]], color = :black, linestyle = :dash)
xlims!(ax, 0.9 * minimum(df_samples.μ_R₀), maximum(df_samples.μ_R₀) * 1.1)
ylims!(ax, low = 0.0)
ax.xlabel = L"\mu_{R_0}"
ax.xticks = 10:3:25

ax = Axis(fig[2, 1]; ax_kwargs...)
stephist!(ax, df_samples.μ_πv, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.μ_πv, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:μ_πv], color = colors[2])
# vlines!(ax, [μ_πv], color = :black, linestyle = :dash)
# vlines!(ax, [true_pars_sample_vals["μ_πv"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.μ_πv), maximum(df_samples.μ_πv) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"\mu_{\rho}"

ax = Axis(fig[1, 2]; ax_kwargs...)
stephist!(ax, df_samples.μ_δ, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.μ_δ, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:μ_δ], color = colors[2])
# vlines!(ax, [μ_δ], color = :black, linestyle = :dash)
# vlines!(ax, [true_pars_sample_vals["μ_δ"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.μ_δ), maximum(df_samples.μ_δ) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"\mu_{\delta}"

ax = Axis(fig[2, 2]; ax_kwargs...)
stephist!(ax, df_samples.σ_δ, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.σ_δ, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:σ_δ], color = colors[2])
# vlines!(ax, [σ_δ], color = :black, linestyle = :dash)
# vlines!(ax, [true_pars_sample_vals["σ_δ"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.σ_δ), maximum(df_samples.σ_δ) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"\sigma_{\delta}"

ax = Axis(fig[1, 3]; ax_kwargs...)
stephist!(ax, df_samples.σ_R₀, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.σ_R₀, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:σ_R₀], color = colors[2])
# vlines!(ax, [true_pars_sample_vals["σ_R₀"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.σ_R₀), maximum(df_samples.σ_R₀) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"σ_{R_0}"

ax = Axis(fig[2, 3]; ax_kwargs...)
stephist!(ax, df_samples.σ_πv, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.σ_πv, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:σ_πv], color = colors[2])
# vlines!(ax, [true_pars_sample_vals["σ_πv"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.σ_πv), maximum(df_samples.σ_πv) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"σ_{\rho}"

ax = Axis(fig[1, 4]; ax_kwargs...)
stephist!(ax, df_samples.κ, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.κ, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:κ], color = colors[2])
# vlines!(ax, [κ], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_samples.κ), maximum(df_samples.κ) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"\kappa"

resize_to_layout!(fig)

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

display(fig)

save(fig_loc * "nba_hyper_pars_posteriors.pdf", fig, pt_per_unit = 1.0)

##

# df_samples = CSV.read(results_dir("samples_nba.csv"), DataFrame)
# Thin the number of samples
# df_samples = df_samples[1:5:end, :]

##

function add_transformed_parameters_to_df!(df_samples, N)
    for i in 1:N
        df_samples[:, "R₀_$i"] =
            df_samples[:, "μ_R₀"] .+ df_samples[:, "σ_R₀"] .* df_samples[:, "z_R₀_$i"]
        df_samples[:, "δ_$i"] =
            df_samples[:, "μ_δ"] .+ df_samples[:, "σ_δ"] .* df_samples[:, "z_δ_$i"]
        df_samples[:, "πv_$i"] =
            df_samples[:, "μ_πv"] .+ df_samples[:, "σ_πv"] .* df_samples[:, "z_πv_$i"]
    end
    return nothing
end

add_transformed_parameters_to_df!(df_samples, N)

R0s = Dict()

N = length(data)

# for ind in 1:N
#     target_cols = ["β_$ind", "δ_$ind", "πv_$ind", "c_$ind"]
#     df_samples_ind = df_samples[:, target_cols]
#     R0s[ind] =
#         (df_samples_ind[:, 1] .* df_samples_ind[:, 3]) ./
#         (df_samples_ind[:, 2] .* df_samples_ind[:, 4])
# end

R0s = Dict()
βs = Dict()
πs = Dict()
δs = Dict()

for ind in 1:N
    target_cols = ["R₀_$ind", "δ_$ind", "πv_$ind"]
    df_samples_ind = df_samples[:, target_cols]
    R0s[ind] = df_samples_ind[:, 1]

    # βs[ind] = deepcopy(df_samples_ind[:, 1])
    δs[ind] = deepcopy(df_samples_ind[:, 2])
    πs[ind] = deepcopy(df_samples_ind[:, 3])
end

##

x_gp = Int[]
y = Float64[]
n_samples = size(df_samples, 1)

R0s_var = [var(R0s[i]) for i in 1:N]
order = Dict()

# R0_vars = [var(R0s[i]) for i in 1:N]

# order = sortperm(R0_vars)

# for i in range(1, N)
#     x_gp = vcat(x_gp, i * ones(n_samples))
#     y = vcat(y, R0s[order[i]])
# end

size_inches = (7.5, 9.0)
size_pt = size_inches .* 72
fig = Figure(size = size_pt, fontsize = 11, dpi = 300, sharex = true, linewidth = 1)
ax = Axis(fig[1, 1], xgridvisible = false, ygridvisible = false, titlealign = :left)
# boxplot!(ax, x_gp, y, show_outliers = false, color = (:purple, 0.5))
# violin!(ax, x_gp, y, color = (:purple, 0.5))

lows = [quantile(R0s[i], 0.025) for i in 1:N]
highs = [quantile(R0s[i], 0.975) for i in 1:N]
order = sortperm(highs - lows)
vals = order
meds = [quantile(R0s[i], 0.5) for i in 1:N]

rangebars!(
    1:N,
    lows[order],
    highs[order],
    # color = LinRange(0, 1, length(vals)),
    linewidth = 1,
    color = :grey
    # whiskerwidth = 3
)

ids_pivotal = [130, 129, 113, 1, 83, 68, 170, 173]
for i in 1:N
    if i ∈ ids_pivotal
        rangebars!(
            [i],
            lows[order[i]],
            highs[order[i]],
            # color = LinRange(0, 1, length(vals)),
            linewidth = 1,
            color = :red
            # whiskerwidth = 3
        )
    end
end
scatter!(1:N, meds[order], color = :black, markersize = 3, strokewidth = 1, strokecolor = :black)

# for (i, o) in enumerate(order)
#     if meds[o] > 16.5
#         rangebars!([i], [lows[o]], [highs[o]], color = :red, linewidth = 1)
#     end
# end

# ax.xlabel = "Individual ID"
ax.ylabel = L"R_0"
ax.title = "(A)"
ax.xticks = 0:10:N

display(fig)

ax = Axis(fig[2, 1], xgridvisible = false, ygridvisible = false, titlealign = :left)
# boxplot!(ax, x_gp, y, show_outliers = false, color = (:purple, 0.5))
# violin!(ax, x_gp, y, color = (:purple, 0.5))

lows = [quantile(δs[i], 0.025) for i in 1:N]
highs = [quantile(δs[i], 0.975) for i in 1:N]
order = sortperm(highs - lows)
vals = order
meds = [quantile(δs[i], 0.5) for i in 1:N]

rangebars!(
    1:N,
    lows[order],
    highs[order],
    # color = LinRange(0, 1, length(vals)),
    linewidth = 1,
    color = :grey
    # whiskerwidth = 3
)

ids_pivotal = [130, 129, 113, 1, 83, 68, 170, 173]
for i in 1:N
    if i ∈ ids_pivotal
        rangebars!(
            [i],
            lows[order[i]],
            highs[order[i]],
            # color = LinRange(0, 1, length(vals)),
            linewidth = 1,
            color = :red
            # whiskerwidth = 3
        )
    end
end
scatter!(1:N, meds[order], color = :black, markersize = 3, strokewidth = 1, strokecolor = :black)

# for (i, o) in enumerate(order)
#     if meds[o] > 16.5
#         rangebars!([i], [lows[o]], [highs[o]], color = :red, linewidth = 1)
#     end
# end

# ax.xlabel = "Individual ID"
ax.ylabel = L"\delta"
ax.title = "(B)"
ax.xticks = 0:10:N

ax = Axis(fig[3, 1], xgridvisible = false, ygridvisible = false, titlealign = :left)
# boxplot!(ax, x_gp, y, show_outliers = false, color = (:purple, 0.5))
# violin!(ax, x_gp, y, color = (:purple, 0.5))

lows = [quantile(πs[i], 0.025) for i in 1:N]
highs = [quantile(πs[i], 0.975) for i in 1:N]
order = sortperm(highs - lows)
vals = order
meds = [quantile(πs[i], 0.5) for i in 1:N]

rangebars!(
    1:N,
    lows[order],
    highs[order],
    # color = LinRange(0, 1, length(vals)),
    linewidth = 1,
    color = :grey
    # whiskerwidth = 3
)

ids_pivotal = [130, 129, 113, 1, 83, 68, 170, 173]
for i in 1:N
    if i ∈ ids_pivotal
        rangebars!(
            [i],
            lows[order[i]],
            highs[order[i]],
            # color = LinRange(0, 1, length(vals)),
            linewidth = 1,
            color = :red
            # whiskerwidth = 3
        )
    end
end
scatter!(1:N, meds[order], color = :black, markersize = 3, strokewidth = 1, strokecolor = :black)

# for (i, o) in enumerate(order)
#     if meds[o] > 16.5
#         rangebars!([i], [lows[o]], [highs[o]], color = :red, linewidth = 1)
#     end
# end

# ax.xlabel = "Individual ID"
ax.ylabel = L"\varrho"
ax.title = "(C)"
ax.xticks = 0:10:N

infection_time_true = Dict()

for ind in 1:N
    target_cols = ["infection_time"]
    df_ind = df[df.ID .== ind, target_cols]
    # infection_time_true[ind] = df_ind.infection_time[1]
end

infection_times = Dict()

N = length(data)

for ind in 1:N
    target_cols = ["infection_time_$ind"]
    df_samples_ind = df_samples[:, target_cols]
    infection_times[ind] = df_samples_ind[:, 1]
end

x_gp = Int[]
y = Float64[]
n_samples = nrow(df_samples)

for i in range(1, N)
    x_gp = vcat(x_gp, i * ones(n_samples))
    y = vcat(y, infection_times[order[i]])
end

# infection_time_true_vec = zeros(N)
# for i in range(1, N)
#     infection_time_true_vec[i] = infection_time_true[i]
# end

ax = Axis(fig[4, 1], xgridvisible = false, ygridvisible = false, titlealign = :left)
# boxplot!(ax, x_gp, y, show_outliers = false, color = (:purple, 0.5))

lows = [quantile(infection_times[i], 0.025) for i in 1:N]
highs = [quantile(infection_times[i], 0.975) for i in 1:N]
meds = [quantile(infection_times[i], 0.5) for i in 1:N]

rangebars!(1:N, lows[order], highs[order], color = :grey, linewidth = 1)

for i in 1:N
    if i ∈ ids_pivotal
        rangebars!(
            [i],
            lows[order[i]],
            highs[order[i]],
            # color = LinRange(0, 1, length(vals)),
            linewidth = 1,
            color = :red
            # whiskerwidth = 3
        )
    end
end

scatter!(
    ax, 1:N, meds[order], color = :black, markersize = 3, strokewidth = 1, strokecolor = :black
)

# scatter!(
#     ax,
#     1:N,
#     infection_time_true_vec[order],
#     color = :white,
#     markersize = 6,
#     strokewidth = 1,
#     strokecolor = :black
# )
ax.xlabel = "Individual ID"
ax.ylabel = L"t_0 \textrm{ (days before peak VL)}"
ax.title = "(D)"
ax.xticks = 0:10:N
display(fig)

save(fig_loc * "nba_posteriors.pdf", fig, pt_per_unit = 1.0)
