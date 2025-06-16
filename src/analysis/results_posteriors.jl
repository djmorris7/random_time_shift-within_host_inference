include("../inference/within_host_inference.jl")
include("results.jl")
include("../plotting.jl")

##

Random.seed!(2023)

(data, ids) = get_cleaned_data("data/sims/sim_data_clean.csv")

df_true_pars = CSV.read(data_dir("sims/parameters.csv"), DataFrame)
df_true_hyper_pars = CSV.read(data_dir("sims/hyper_parameters.csv"), DataFrame)

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

pars0 = deepcopy(mean_pars)

S0 = Z0[1]
Z0_bp = Z0[2:end]

integrator = f -> quadgk(f, -7, 7)[1]

σ_R₀, σ_k, σ_δ, σ_πv, σ_c = df_true_hyper_pars[1, [:σ_R₀, :σ_k, :σ_δ, :σ_πv, :σ_c]]

##

infection_time_ranges = [zeros(2) for _ in eachindex(df_true_pars.infection_time)]
for (i, dat) in enumerate(data)
    obs_peak_timing = dat.obs_times[findmax(dat.vl)[2]]

    earliest_timing = obs_peak_timing - 20
    latest_timing = obs_peak_timing + 10

    a = floor(df_true_pars.infection_time[i]) - 1
    b = ceil(df_true_pars.infection_time[i]) + 1

    infection_time_ranges[i] = [a, b]
end

##

df_samples = [CSV.read(results_dir("samples_sim_$i.csv"), DataFrame) for i in 1:4]
burnin = 10000
thin = 10
df_samples = [df_samples[i][burnin:thin:end, :] for i in 1:4]
df_samples = vcat(df_samples...)
samples = Matrix(df_samples)

df = CSV.read(data_dir("sims/parameters.csv"), DataFrame)
select!(df, [:ID, :R₀, :k, :δ, :πv, :c, :infection_time])

param_sigmas = CSV.read(data_dir("param_sigma.csv"), DataFrame)
σ_β, σ_δ, σ_πv = param_sigmas[1, ["β", "δ", "πv"]]

σ_β = 1.0
σ_πv = 1.0
σ_δ = 0.5

##

# induced priors
nsamples_prior = 100_00
prior_samples = zeros(nsamples_prior, 3)

j = 1

for (i, symbol) in enumerate(fieldnames(Params))
    symbol ∈ [:infection_time_range, :infection_time, :z_R₀, :z_δ, :z_πv, :c, :k] && continue

    μ = rand(hyper_priors[string_to_symbol("μ_", symbol)], nsamples_prior)
    σ = rand(hyper_priors[string_to_symbol("σ_", symbol)], nsamples_prior)
    if symbol == :R₀
        prior_samples[:, j] = rand.(Truncated.(Normal.(μ, σ), 0.1, 100))
    elseif symbol == :πv
        prior_samples[:, j] = rand.(Truncated.(Normal.(μ, σ), 0.1, 100))
    elseif symbol == :δ
        # σ = rand(hyper_priors[string_to_symbol("σ_", symbol)], nsamples_prior)
        prior_samples[:, j] = rand.(Truncated.(Normal.(μ, σ), 0.25, 100))
    end
    j += 1
end

##

function get_nice_xlims_for_posterior(x; scaling = 0.03)
    x_min = minimum(x)
    x_max = maximum(x)

    return ((1 - scaling) * x_min, (1 + scaling) * x_max)
end

df_true_pars

true_pars_sample_vals = Dict()
for s in ["R₀", "δ", "πv"]
    true_pars_sample_vals["μ_" * s] = mean(df_true_pars[:, s])
    true_pars_sample_vals["σ_" * s] = std(df_true_pars[:, s])
end

μ_R₀ = 8.0

# size_inches = (7.5, 5.0)
size_inches = (7.75, 4.5)
size_pt = size_inches .* 72
fig = Figure(size = size_pt, fontsize = 10, dpi = 300)

ax = Axis(fig[1, 1]; ax_kwargs...)
stephist!(ax, df_samples.μ_R₀, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.μ_R₀, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
# density!(ax, df_samples.μ_R₀, color = (colors[1], 0.5))
plot!(ax, hyper_priors[:μ_R₀], color = colors[2])
vlines!(ax, [true_pars_sample_vals["μ_R₀"]], color = :black, linestyle = :dash)
xlims!(ax, get_nice_xlims_for_posterior(df_samples.μ_R₀))
ylims!(ax, low = 0.0)
ax.xlabel = L"\mu_{R_0}"

ax = Axis(fig[2, 1]; ax_kwargs...)
stephist!(ax, df_samples.μ_πv, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.μ_πv, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:μ_πv], color = colors[2])
vlines!(ax, [true_pars_sample_vals["μ_πv"]], color = :black, linestyle = :dash)
xlims!(ax, get_nice_xlims_for_posterior(df_samples.μ_πv))
ylims!(ax, low = 0.0)
ax.xlabel = L"\mu_{\rho}"

ax = Axis(fig[1, 2]; ax_kwargs...)
stephist!(ax, df_samples.μ_δ, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.μ_δ, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:μ_δ], color = colors[2])
vlines!(ax, [true_pars_sample_vals["μ_δ"]], color = :black, linestyle = :dash)
xlims!(ax, get_nice_xlims_for_posterior(df_samples.μ_δ))
ylims!(ax, low = 0.0)
ax.xlabel = L"\mu_{\delta}"

ax = Axis(fig[2, 2]; ax_kwargs..., xticks = 0:0.04:2.0)
stephist!(ax, df_samples.σ_δ, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.σ_δ, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:σ_δ], color = colors[2])
vlines!(ax, [true_pars_sample_vals["σ_δ"]], color = :black, linestyle = :dash)
xlims!(ax, get_nice_xlims_for_posterior(df_samples.σ_δ))
ylims!(ax, low = 0.0)
ax.xlabel = L"\sigma_{\delta}"

ax = Axis(fig[1, 3]; ax_kwargs..., xticks = 0:0.5:2.0)
stephist!(ax, df_samples.σ_R₀, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.σ_R₀, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:σ_R₀], color = colors[2])
vlines!(ax, [true_pars_sample_vals["σ_R₀"]], color = :black, linestyle = :dash)
xlims!(ax, get_nice_xlims_for_posterior(df_samples.σ_R₀))
ylims!(ax, low = 0.0)
ax.xlabel = L"\sigma_{R_0}"

ax = Axis(fig[2, 3]; ax_kwargs...)
stephist!(ax, df_samples.σ_πv, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.σ_πv, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:σ_πv], color = colors[2])
vlines!(ax, [true_pars_sample_vals["σ_πv"]], color = :black, linestyle = :dash)
xlims!(ax, get_nice_xlims_for_posterior(df_samples.σ_πv))
ylims!(ax, low = 0.0)
ax.xlabel = L"\sigma_{\rho}"

ax = Axis(fig[1, 4]; ax_kwargs..., xticks = 0:0.05:2)
stephist!(ax, df_samples.κ, bins = 30, normalization = :pdf, color = colors[1])
hist!(ax, df_samples.κ, bins = 30, normalization = :pdf, color = (colors[1], 0.3))
plot!(ax, hyper_priors[:κ], color = colors[2])
vlines!(ax, [κ], color = :black, linestyle = :dash)
xlims!(ax, get_nice_xlims_for_posterior(df_samples.κ))
ylims!(ax, low = 0.0)
ax.xlabel = L"\kappa"

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

# resize_to_layout!(fig)

display(fig)

save(fig_loc * "sim_hyper_pars_posteriors.pdf", fig, pt_per_unit = 1.0)

##

fig = Figure()
ax = Axis(fig[1, 1])
hexbin!(ax, df_samples.μ_R₀, df_samples.μ_δ, bins = 50)
scatter!(ax, df_true_hyper_pars.μ_R₀, df_true_hyper_pars.μ_δ, color = :red)
ax = Axis(fig[1, 2])
hexbin!(ax, df_samples.μ_R₀, df_samples.μ_πv, bins = 50)
scatter!(ax, df_true_hyper_pars.μ_R₀, df_true_hyper_pars.μ_πv, color = :red)
ax = Axis(fig[1, 3])
hexbin!(ax, df_samples.μ_δ, df_samples.μ_πv, bins = 50)
scatter!(ax, df_true_hyper_pars.μ_δ, df_true_hyper_pars.μ_πv, color = :red)
display(fig)

fig = Figure()
ax = Axis(fig[1, 1])
hexbin!(ax, df_samples.μ_δ ./ df_samples.μ_πv, df_samples.μ_δ, bins = 50)
scatter!(
    ax, df_true_hyper_pars.μ_δ ./ df_true_hyper_pars.μ_πv, df_true_hyper_pars.μ_δ, color = :red
)

display(fig)

##

prior_predictive_pars = Dict()

n = 10000

prior_predictive_pars["R₀"] =
    rand.(Truncated.(Normal.(rand(hyper_priors[:μ_R₀], n), rand(hyper_priors[:σ_R₀], n)), 0.1, 100))
prior_predictive_pars["δ"] =
    rand.(Truncated.(Normal.(rand(hyper_priors[:μ_δ], n), rand(hyper_priors[:σ_δ], n)), 0.25, 100))
prior_predictive_pars["πv"] =
    rand.(Truncated.(Normal.(rand(hyper_priors[:μ_πv], n), rand(hyper_priors[:σ_πv], n)), 0.1, 100))

posterior_predictive_pars = Dict()

posterior_predictive_pars["R₀"] = df_samples.μ_R₀ + df_samples.σ_R₀ .* randn(size(df_samples, 1))
posterior_predictive_pars["δ"] = df_samples.μ_δ + df_samples.σ_δ .* randn(size(df_samples, 1))
posterior_predictive_pars["πv"] = df_samples.μ_πv + df_samples.σ_πv .* randn(size(df_samples, 1))

size_inches = (6.5, 3.0)
size_pt = size_inches .* 72
fig = Figure(size = size_pt, fontsize = 10, dpi = 300)

ax = Axis(fig[1, 1]; ax_kwargs...)
stephist!(ax, posterior_predictive_pars["R₀"], bins = 30, normalization = :pdf, color = colors[1])
stephist!(ax, prior_predictive_pars["R₀"], bins = 30, normalization = :pdf, color = colors[2])
hist!(
    ax, posterior_predictive_pars["R₀"], bins = 30, normalization = :pdf, color = (colors[1], 0.3)
)
# vlines!(ax, df_true_pars.R₀, color = (:black, 0.3))
hist!(ax, df_true_pars.R₀, color = (:black, 0.3), normalization = :pdf)
# vlines!(ax, [true_pars_sample_vals["μ_R₀"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_true_pars.R₀), maximum(df_true_pars.R₀) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"R_0"

ax = Axis(fig[1, 2]; ax_kwargs...)
stephist!(ax, posterior_predictive_pars["δ"], bins = 30, normalization = :pdf, color = colors[1])
stephist!(ax, prior_predictive_pars["δ"], bins = 30, normalization = :pdf, color = colors[2])
hist!(ax, posterior_predictive_pars["δ"], bins = 30, normalization = :pdf, color = (colors[1], 0.3))
# vlines!(ax, df_true_pars.δ, color = (:black, 0.3))
hist!(ax, df_true_pars.δ, color = (:black, 0.3), normalization = :pdf)
# vlines!(ax, [true_pars_sample_vals["μ_δ"]], color = :black, linestyle = :dash)
xlims!(ax, 0.95 * minimum(df_true_pars.δ), maximum(df_true_pars.δ) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"δ"

ax = Axis(fig[1, 3]; ax_kwargs...)
stephist!(ax, posterior_predictive_pars["πv"], bins = 30, normalization = :pdf, color = colors[1])
stephist!(ax, prior_predictive_pars["πv"], bins = 30, normalization = :pdf, color = colors[2])
hist!(
    ax, posterior_predictive_pars["πv"], bins = 30, normalization = :pdf, color = (colors[1], 0.3)
)
# vlines!(ax, df_true_pars.πv, color = (:black, 0.3))
hist!(ax, df_true_pars.πv, color = (:black, 0.3), normalization = :pdf)
xlims!(ax, 0.95 * minimum(df_true_pars.πv), maximum(df_true_pars.πv) * 1.05)
ylims!(ax, low = 0.0)
ax.xlabel = L"\rho"

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

# resize_to_layout!(fig)

display(fig)

save(fig_loc * "sim_posterior_predictive_pars.pdf", fig, pt_per_unit = 1.0)

##

μ = df_samples.μ_πv
σ_tmp = df_samples.σ_πv

x = rand.(Normal.(μ, σ_tmp))

fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, x, normalization = :pdf, color = (:blue, 0.5), bins = 50)
hist!(ax, df_true_pars.πv, color = (:grey, 0.5), bins = 50, normalization = :pdf)
display(fig)

##

# function sample_generalized_gamma(pars)
#     a, d, p = pars
#     u = rand()
#     # Use the inverse of the incomplete Gamma function
#     y = gamma_inc_inv(d / p, u, 1 - u)
#     # Apply transformation to obtain Generalized Gamma random variables
#     x = a * y^(1 / p)
#     return x
# end

function get_gg_pars_nn(θ, nn, Z0_bp, S0)
    """
    Uses the neural network to calculate the parameters of the generalised gamma distribution.
    """
    R₀, δ, πv = θ
    k = 4.0
    c = 10.0

    β = get_bp_β(R₀, k, δ, πv, c)

    # Calculate omega matrix and artefacts from that.
    Ω = [
        -k k 0
        0 -δ πv
        β 0 -c
    ]

    λ, u_norm, _ = calculate_BP_contributions(Ω)
    # Calc expected value of W
    μ_w = dot(Z0_bp, u_norm)
    # Use the neural network to calculate the parameters of the generalised gamma distribution
    pars_m3 = nn([R₀, δ, πv])

    return μ_w, λ, pars_m3
end

nn = load_nn()

size_inches = (3, 3)
size_pt = size_inches .* 72
fig = Figure(size = size_pt, fontsize = 10, dpi = 300)
ax = Axis(fig[1, 1]; ax_kwargs...)

prior_time_shift = zeros(nsamples_prior)

τ_range = -3:0.01:3
nn_pars = []
μ_w = 0.0

n_samples = 100

j = 1
for i in 1:n_samples
    λ = -Inf
    τ = 0.0
    nn_pars = []

    while λ <= 0.0 || τ < -7.0 || τ > 7.0
        j = rand(1:nsamples_prior)
        θ = prior_samples[j, :]
        μ_w, λ, nn_pars = get_gg_pars_nn(θ, nn, Z0_bp, S0)
        w = sample_generalized_gamma(nn_pars)
        τ = (log(w) - log(μ_w)) / λ
    end

    lines!(
        ax,
        τ_range,
        [exp(log_τ_prior(τ, nn_pars, μ_w, λ)) for τ in τ_range],
        color = (colors[2], 0.2)
    )

    prior_time_shift[i] = τ
end

# df_samples_ind = extract_individual_params(df_samples, 1)
df_samples_ind = df_samples[:, 1:3]
df_samples_ind.R₀ = df_samples.μ_R₀ + df_samples.σ_R₀ .* df_samples_ind.z_R₀_1
df_samples_ind.δ = df_samples.μ_δ + df_samples.σ_δ .* df_samples_ind.z_δ_1
df_samples_ind.πv = df_samples.μ_πv + df_samples.σ_πv .* df_samples_ind.z_πv_1

posterior_time_shift = zeros(nsamples_prior)

for i in 1:n_samples
    j = rand(1:size(df_samples_ind, 1))
    θ = df_samples_ind[j, [:R₀, :δ, :πv]]
    μ_w, λ, nn_pars = get_gg_pars_nn(θ, nn, Z0_bp, S0)

    w = sample_generalized_gamma(nn_pars)
    τ = (log(w) - log(μ_w)) / λ

    lines!(
        ax,
        τ_range,
        [exp(log_τ_prior(τ, nn_pars, μ_w, λ)) for τ in τ_range],
        color = (colors[1], 0.2)
    )
    # posterior_time_shift[i] = τ
end

# density!(ax, prior_time_shift, color = (:blue, 0.2), label = "Prior")
# density!(ax, posterior_time_shift, color = (:red, 0.2), label = "Posterior")

ax.xlabel = L"\tau"
xlims!(-4, 4)
# xlims!(-4, 4)
display(fig)
save(fig_loc * "time_shift.pdf", fig, pt_per_unit = 1.0)

##

R0s = Dict()

N = length(data)

# for ind in 1:N
#     target_cols = ["β_$ind", "δ_$ind", "πv_$ind", "c_$ind"]
#     df_samples_ind = df_samples[:, target_cols]
#     R0s[ind] =
#         (df_samples_ind[:, 1] .* df_samples_ind[:, 3]) ./
#         (df_samples_ind[:, 2] .* df_samples_ind[:, 4])
# end

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
πs = Dict()
δs = Dict()
infection_times = Dict()

for ind in 1:N
    target_cols = ["R₀_$ind", "δ_$ind", "πv_$ind", "infection_time_$ind"]
    df_samples_ind = df_samples[:, target_cols]
    R0s[ind] = df_samples_ind[:, 1]

    # βs[ind] = deepcopy(df_samples_ind[:, 1])
    δs[ind] = deepcopy(df_samples_ind[:, 2])
    πs[ind] = deepcopy(df_samples_ind[:, 3])
    infection_times[ind] = deepcopy(df_samples_ind[:, 4])
end

##

# Compute the summaries for each posterior sample
df_order = DataFrame(
    ID = 1:N,
    R0_true = df_true_pars.R₀,
    δ_true = df_true_pars.δ,
    πv_true = df_true_pars.πv,
    infection_time_true = df_true_pars.infection_time
)

function compute_summaries!(df_order, R0s, δs, πs, infection_times; difference = false)
    if !difference
        df_order[:, "R0_meds"] = [median(R0s[i]) for i in 1:N]
        df_order[:, "R0_lows"] = [quantile(R0s[i], 0.025) for i in 1:N]
        df_order[:, "R0_highs"] = [quantile(R0s[i], 0.975) for i in 1:N]
        df_order[:, "δ_meds"] = [median(δs[i]) for i in 1:N]
        df_order[:, "δ_lows"] = [quantile(δs[i], 0.025) for i in 1:N]
        df_order[:, "δ_highs"] = [quantile(δs[i], 0.975) for i in 1:N]
        df_order[:, "πv_meds"] = [median(πs[i]) for i in 1:N]
        df_order[:, "πv_lows"] = [quantile(πs[i], 0.025) for i in 1:N]
        df_order[:, "πv_highs"] = [quantile(πs[i], 0.975) for i in 1:N]
        df_order[:, "infection_time_meds"] = [median(infection_times[i]) for i in 1:N]
        df_order[:, "infection_time_lows"] = [quantile(infection_times[i], 0.025) for i in 1:N]
        df_order[:, "infection_time_highs"] = [quantile(infection_times[i], 0.975) for i in 1:N]
    else
        df_order[:, "R0_meds"] = [median(R0s[i] .- df_order[i, :R0_true]) for i in 1:N]
        df_order[:, "R0_lows"] = [quantile(R0s[i] .- df_order[i, :R0_true], 0.025) for i in 1:N]
        df_order[:, "R0_highs"] = [quantile(R0s[i] .- df_order[i, :R0_true], 0.975) for i in 1:N]
        df_order[:, "δ_meds"] = [median(δs[i] .- df_order[i, :δ_true]) for i in 1:N]
        df_order[:, "δ_lows"] = [quantile(δs[i] .- df_order[i, :δ_true], 0.025) for i in 1:N]
        df_order[:, "δ_highs"] = [quantile(δs[i] .- df_order[i, :δ_true], 0.975) for i in 1:N]
        df_order[:, "πv_meds"] = [median(πs[i] .- df_order[i, :πv_true]) for i in 1:N]
        df_order[:, "πv_lows"] = [quantile(πs[i] .- df_order[i, :πv_true], 0.025) for i in 1:N]
        df_order[:, "πv_highs"] = [quantile(πs[i] .- df_order[i, :πv_true], 0.975) for i in 1:N]
        df_order[:, "infection_time_meds"] = [
            median(infection_times[i] .- df_order[i, :infection_time_true]) for i in 1:N
        ]
        df_order[:, "infection_time_lows"] = [
            quantile(infection_times[i] .- df_order[i, :infection_time_true], 0.025) for i in 1:N
        ]
        df_order[:, "infection_time_highs"] = [
            quantile(infection_times[i] .- df_order[i, :infection_time_true], 0.975) for i in 1:N
        ]
    end

    return df_order
end

compute_summaries!(df_order, R0s, δs, πs, infection_times; difference = true)

##

function get_number_pre_peak_observations(data::IndividualData)
    obs_peak_timing = data.obs_times[findmax(data.vl)[2]]
    return length(data.obs_times[data.obs_times .< obs_peak_timing])
end

function get_number_post_peak_observations(data::IndividualData)
    obs_peak_timing = data.obs_times[findmax(data.vl)[2]]
    return length(data.obs_times[data.obs_times .> obs_peak_timing])
end

pre_peak_obs = [get_number_pre_peak_observations(dat) for dat in data]
post_peak_obs = [get_number_post_peak_observations(dat) for dat in data]
combos = zip(pre_peak_obs, post_peak_obs)

uniq_combos_order = unique(combos)
uniq_combos_order = sortperm(unique(combos))

findmin(pre_peak_obs)
findmin(post_peak_obs)

fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, pre_peak_obs, bins = 0:maximum(pre_peak_obs), color = :blue, normalization = :pdf)
vlines!(ax, [median(pre_peak_obs)], color = :red, linestyle = :dash)
ax = Axis(fig[2, 1])
hist!(ax, post_peak_obs, bins = 0:maximum(post_peak_obs), color = :blue, normalization = :pdf)
vlines!(ax, [median(post_peak_obs)], color = :red, linestyle = :dash)
display(fig)

median(pre_peak_obs)
median(post_peak_obs)

using DataStructures

group_type = SortedDict("HH" => [], "HL" => [], "LH" => [], "LL" => [])

pre_peak_median = median(pre_peak_obs)
post_peak_median = median(post_peak_obs)

for i in 1:N
    if pre_peak_obs[i] > pre_peak_median && post_peak_obs[i] > post_peak_median
        push!(group_type["HH"], i)
    elseif pre_peak_obs[i] > pre_peak_median && post_peak_obs[i] <= post_peak_median
        push!(group_type["HL"], i)
    elseif pre_peak_obs[i] <= pre_peak_median && post_peak_obs[i] > post_peak_median
        push!(group_type["LH"], i)
    else
        push!(group_type["LL"], i)
    end
end

group_type["HH"]
group_type["HL"]
group_type["LH"]
group_type["LL"]

plot(data[16].obs_times, data[16].vl)
plot(data[76].obs_times, data[76].vl)

uniq_pre_obs = sort(unique(pre_peak_obs))
uniq_post_obs = sort(unique(post_peak_obs))

##

interesting_ids = Set()

size_inches = (7.5, 5.5)
size_pt = size_inches .* 72
fig = Figure(size = size_pt, fontsize = 10, dpi = 300, sharex = true, sharey = true, linewidth = 1)
axs = [Axis(fig[i, j]; ax_kwargs...) for i in 1:2, j in 1:2]
axs[1].title = L"(A) $R_0$"
axs[2].title = L"(B) $\delta$"
axs[3].title = L"(C) $\rho$"
axs[4].title = L"(D) $t_0$"

offsets = Dict(
    :R0 => 0.08 * (maximum(df_order[:, :R0_highs]) - minimum(df_order[:, :R0_lows])),
    :δ => 0.08 * (maximum(df_order[:, :δ_highs]) - minimum(df_order[:, :δ_lows])),
    :πv => 0.08 * (maximum(df_order[:, :πv_highs]) - minimum(df_order[:, :πv_lows])),
    :infection_time =>
        0.08 *
        (maximum(df_order[:, :infection_time_highs]) - minimum(df_order[:, :infection_time_lows]))
)

for i in 1:N
    if 0 < df_order[i, :R0_lows] || 0 > df_order[i, :R0_highs]
        rangebars!(
            axs[1],
            [i],
            [df_order[i, "R0_lows"]],
            [df_order[i, "R0_highs"]],
            linewidth = 1,
            color = :red
        )

        if 0 < df_order[i, :R0_lows]
            text!(
                axs[1],
                i,
                df_order[i, :R0_highs] + offsets[:R0],
                text = string(i),
                fontsize = 8,
                justification = :center,
                align = (:center, :top)
            )
        else
            text!(
                axs[1],
                i,
                df_order[i, :R0_lows] - offsets[:R0],
                text = string(i),
                fontsize = 8,
                justification = :center,
                align = (:center, :bottom)
            )
        end

        interesting_ids = union(interesting_ids, Set([i]))
    else
        rangebars!(
            axs[1],
            [i],
            [df_order[i, "R0_lows"]],
            [df_order[i, "R0_highs"]],
            linewidth = 1,
            color = :grey
        )
    end
end

hlines!(axs[1], [0.0], color = :black, linestyle = :dash)
scatter!(
    axs[1],
    1:N,
    df_order[:, :R0_meds],
    color = :white,
    markersize = 3,
    strokewidth = 1,
    strokecolor = :black
)
for i in 1:N
    if 0 < df_order[i, :δ_lows] || 0 > df_order[i, :δ_highs]
        rangebars!(
            axs[2],
            [i],
            [df_order[i, "δ_lows"]],
            [df_order[i, "δ_highs"]],
            linewidth = 1,
            color = :red
        )

        if 0 < df_order[i, :δ_lows]
            text!(
                axs[2],
                i,
                df_order[i, :δ_highs] + offsets[:δ],
                text = string(i),
                fontsize = 8,
                justification = :center,
                align = (:center, :top)
            )
        else
            text!(
                axs[2],
                i,
                df_order[i, :δ_lows] - offsets[:δ],
                text = string(i),
                fontsize = 8,
                justification = :center,
                align = (:center, :bottom)
            )
        end

        interesting_ids = union(interesting_ids, Set([i]))
    else
        rangebars!(
            axs[2],
            [i],
            [df_order[i, "δ_lows"]],
            [df_order[i, "δ_highs"]],
            linewidth = 1,
            color = :grey
        )
    end
end
hlines!(axs[2], [0.0], color = :black, linestyle = :dash)
scatter!(
    axs[2],
    1:N,
    df_order[:, :δ_meds],
    color = :white,
    markersize = 3,
    strokewidth = 1,
    strokecolor = :black
)
for i in 1:N
    if 0 < df_order[i, :πv_lows] || 0 > df_order[i, :πv_highs]
        rangebars!(
            axs[3],
            [i],
            [df_order[i, "πv_lows"]],
            [df_order[i, "πv_highs"]],
            linewidth = 1,
            color = :red
        )

        if 0 < df_order[i, :πv_lows]
            text!(
                axs[3],
                i,
                df_order[i, :πv_highs] + offsets[:πv],
                text = string(i),
                fontsize = 8,
                justification = :center,
                align = (:center, :top)
            )
        else
            text!(
                axs[3],
                i,
                df_order[i, :πv_lows] - offsets[:πv],
                text = string(i),
                fontsize = 8,
                justification = :center,
                align = (:center, :bottom)
            )
        end

        interesting_ids = union(interesting_ids, Set([i]))
    else
        rangebars!(
            axs[3],
            [i],
            [df_order[i, "πv_lows"]],
            [df_order[i, "πv_highs"]],
            linewidth = 1,
            color = :grey
        )
    end
end
hlines!(axs[3], [0.0], color = :black, linestyle = :dash)
scatter!(
    axs[3],
    1:N,
    df_order[:, :πv_meds],
    color = :white,
    markersize = 3,
    strokewidth = 1,
    strokecolor = :black
)
for i in 1:N
    if 0 < df_order[i, :infection_time_lows] || 0 > df_order[i, :infection_time_highs]
        rangebars!(
            axs[4],
            [i],
            [df_order[i, "infection_time_lows"]],
            [df_order[i, "infection_time_highs"]],
            linewidth = 1,
            color = :red
        )

        if 0 < df_order[i, :infection_time_lows]
            text!(
                axs[4],
                i,
                df_order[i, :infection_time_highs] + offsets[:infection_time],
                text = string(i),
                fontsize = 8,
                justification = :center,
                align = (:center, :top)
            )
        else
            text!(
                axs[4],
                i,
                df_order[i, :infection_time_lows] - offsets[:infection_time],
                text = string(i),
                fontsize = 8,
                justification = :center,
                align = (:center, :bottom)
            )
        end

        interesting_ids = union(interesting_ids, Set([i]))
    else
        rangebars!(
            axs[4],
            [i],
            [df_order[i, "infection_time_lows"]],
            [df_order[i, "infection_time_highs"]],
            linewidth = 1,
            color = :grey
        )
    end
end
hlines!(axs[4], [0.0], color = :black, linestyle = :dash)
scatter!(
    axs[4],
    1:N,
    df_order[:, :infection_time_meds],
    color = :white,
    markersize = 3,
    strokewidth = 1,
    strokecolor = :black
)

Label(fig[1:2, 0], "Difference in parameter value (inferred - true)", rotation = pi / 2)
Label(fig[3, 1:2], "Individual ID")

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

display(fig)

resize_to_layout!(fig)

save(fig_loc * "sim_posteriors_all_with_ests.pdf", fig, pt_per_unit = 1.0)

##

interesting_ids
