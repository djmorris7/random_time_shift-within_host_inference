include("../inference/within_host_inference.jl")
include("results.jl")
include("../plotting.jl")

##

Random.seed!(2023)

dataset_id = 1

(data, ids) = get_cleaned_data("data/sims/covid_data_clean_$dataset_id.csv")

df_true_pars = CSV.read(data_dir("sims/covid_parameters_$dataset_id.csv"), DataFrame)
df_true_hyper_pars = CSV.read(data_dir("sims/covid_hyper_parameters_$dataset_id.csv"), DataFrame)

fig_loc = "figures/"
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

df_samples = [
    CSV.read(results_dir("sim_samples/dataset_$dataset_id/samples_$i.csv"), DataFrame) for i in 1:3
]
burnin = 10000
thin = 10
df_samples = vcat(df_samples...)
df_samples.σ_R₀ = exp.(df_samples.σ_R₀)
df_samples.σ_δ = exp.(df_samples.σ_δ)
df_samples.σ_πv = exp.(df_samples.σ_πv)

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

##

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

size_inches = (3.5, 3.0)
size_pt = size_inches .* inch
fig = Figure(size = size_pt, fontsize = fontsize, dpi = dpi)
ax = Axis(fig[1, 1]; ax_kwargs...)

prior_time_shift = zeros(nsamples_prior)

τ_range = -5:0.01:5
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

ax.ylabel = L"\textrm{density}"
ax.xlabel = L"\tau"
xlims!(-3, 2)
# xlims!(-4, 4)
display(fig)
save(fig_loc * "time_shift.png", fig, px_per_unit = dpi / inch)
save(fig_loc * "time_shift.pdf", fig)
