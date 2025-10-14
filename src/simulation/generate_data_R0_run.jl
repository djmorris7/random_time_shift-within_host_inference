"""
This script generates some data from the stochastic TCL model and adds some observation noise in.
We generate the VL trajectories for all individuals assuming the same initial conditions.

Outputs are saved in the data/sims directory. The following files are generated:
    - data.csv: The noisy and true VL trajectories for each individual.
    - parameters.csv: The parameters for each individual.
    - hyper_parameters.csv: The hyperparameters/shared parameters for the model.
"""

include("generate_data_R0.jl")

##

# Set seed for reproducibility.
Random.seed!(2023)

# Testing the gillespie simulator
S0 = Int(8e7)

K = S0

# Individual parameters (means)
μ_R₀ = 8.0
μ_k = 4.0
μ_δ = 1.3
δ_range_ke = [1.2, 1.35]
μ_πv = 3.0
πv_range_ke = [2.76, 3.41]
μ_c = 10.0

μ = [μ_R₀, μ_k, μ_δ, μ_πv, μ_c]

fixed = [false, true, true, false, false]
σs = [0.5, 0, 0.15, 0.25, 0]

##

# X(t) = (U, E, I, V)
Z0 = [S0 - 1, 1, 0, 0]

LOD = 2.6576090679593496
# LOD = 2.6576090679593496
ct_LOD = 40.0

u0 = Z0
obs_t = 50
tspan = (0, 50)
t_inf = 15.2
model_pars = deepcopy(μ)
prob = ODEProblem(tcl_deterministic!, u0, (0, 20), model_pars)
sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, save_idxs = 4)

V_det = log10p0.(sol.u)

model_pars_stoch = deepcopy(model_pars)
# model_pars_stoch[1] = model_pars_stoch[1] / S0

V_stoch = []
for i in 1:100
    t, Z, _ = tcl_gillespie(model_pars_stoch, Z0)
    V = log10p0.(stack(Z, dims = 1)[:, 4])
    push!(V_stoch, (t, V))
end

##

fig = Figure()
ax = Axis(fig[1, 1])
for V_i in V_stoch
    lines!(ax, V_i[1], V_i[2], color = ("black", 0.2))
end
lines!(ax, sol.t, V_det, color = :red)
display(fig)

##

save_at = 1.0
t0_span = (0.0, 30.0)
obs_t = 22.0

##

nn = load_nn()

pars = [8.0, μ_k, μ_δ, μ_πv, μ_c]

T_obs = 70
Z0_bp = Z0[2:end]

o, y, τ = approx_sample_tcl([8.0, μ_k, μ_δ, μ_πv, μ_c], 0, Z0_bp, nn, prob, T_obs)

κ = 0.5
o_noisy, y_noisy = add_noise_vls(o, y, κ)

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, o, y, color = :black)
plot!(ax, o_noisy, y_noisy, color = :red)
display(fig)

##

fig = Figure()
ax = Axis(fig[1, 1])
for V_i in V_stoch
    lines!(ax, V_i[1], V_i[2], color = ("black", 0.2))
end
lines!(ax, sol.t, V_det, color = :red)

for n in 1:100
    o, y, τ = approx_sample_tcl([8.0, μ_k, μ_δ, μ_πv, μ_c], 0, Z0_bp, nn, prob, T_obs)
    if is_sim_valid(o, y)
        scatter!(ax, o, y, color = :blue, alpha = 0.5)
    end
end
xlims!(0, 25)
display(fig)

##

hyper_params = Dict(
    :μ_R₀ => μ_R₀,
    :σ_R₀ => σs[1],
    :μ_k => μ_k,
    :σ_k => σs[2],
    :μ_δ => μ_δ,
    :σ_δ => σs[3],
    :μ_πv => μ_πv,
    :σ_πv => σs[4],
    :μ_c => μ_c,
    :σ_c => σs[5]
)

##

# n_samples = 1000

# fig = Figure()
# ax = Axis(fig[1, 1])
# for n in 1:n_samples
#     o, y = approx_sample_tcl(θ, t_inf, Z0_bp, nn, prob, T_obs)
#     if is_sim_valid(o, y)
#         lines!(ax, o, y, color = (:black, 0.3))
#     end
# end
# xlims!(floor(t_inf) - 2, 0)
# vlines!(ax, [t_inf], color = :red)
# display(fig)

##

o, y, θ, t_inf, τ = sim_till_valid(priors, hyper_params, Z0_bp, nn, prob, T_obs)
o2, y2, θ2, t_inf, τ = sim_till_valid(priors, hyper_params, Z0_bp, nn, prob, T_obs)

o_noisy = deepcopy(o)
o_noisy, y_noisy = add_noise_vls(o, y, κ)

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, o, y, color = :black)
plot!(ax, o2, y2, color = :black)
# plot!(ax, o_noisy, y_noisy, color = :red)
display(fig)

##

Random.seed!(2025)

N_datasets = 200
N = 100

if !isdir(data_dir("sims"))
    mkpath(data_dir("sims"))
end

@showprogress for i in 1:N_datasets
    θ = deepcopy(μ)

    IDs = Vector{Int}()
    ind_pars = Vector{Vector{Float64}}()
    obs_times = Vector{Float64}()
    vls = Vector{Float64}()
    obs_vls = Vector{Float64}()
    obs_vls_no_lod = Vector{Float64}()

    # Now generate the data for the N individuals
    for i in 1:N
        # Draw parameters
        # Fix latent period and infectious period of infected cells
        (o, y, θ, t_inf, τ) = sim_till_valid(priors, hyper_params, Z0_bp, nn, prob, T_obs)

        # y_noisy = y
        o_noisy, y_noisy, y_noisy_no_lod = add_noise_vls(o, y, κ)
        t_till_peak = argmax(y_noisy)
        # Shift the observation times back by the time taken to peak. This forces
        # the peak time to be 0 for all individuals.
        peak_time = o_noisy[t_till_peak]
        o_noisy = o_noisy .- peak_time
        # Then need to shift the infection time relative to the peak time
        t_inf = t_inf - peak_time

        id = fill(i, length(o_noisy))
        append!(IDs, id)
        push!(ind_pars, [i; t_inf; deepcopy(θ); τ])
        append!(obs_times, o_noisy)
        append!(vls, y)
        append!(obs_vls, y_noisy)
        append!(obs_vls_no_lod, y_noisy_no_lod)
    end

    ind_pars = stack(ind_pars, dims = 1)

    df_data = DataFrame(
        ID = IDs,
        t = obs_times,
        log_vl = vls,
        noisy_log_vl = obs_vls,
        noisy_log_vl_no_lod = obs_vls_no_lod
    )

    df_params = DataFrame(ind_pars, ["ID", "infection_time", "R₀", "k", "δ", "πv", "c", "τ"])

    CSV.write(data_dir("sims/covid_data_$i.csv"), df_data)
    CSV.write(data_dir("sims/covid_parameters_$i.csv"), df_params)

    all_hyper_params = [μ[1], σs[1], μ[2], 0, μ[3], σs[3], μ[4], σs[4], μ[5], 0, κ]'

    param_labels = ["μ_R₀", "σ_R₀", "μ_k", "σ_k", "μ_δ", "σ_δ", "μ_πv", "σ_πv", "μ_c", "σ_c", "κ"]

    df_hyper_params = DataFrame(all_hyper_params, param_labels)

    CSV.write(data_dir("sims/covid_hyper_parameters_$i.csv"), df_hyper_params)
end
