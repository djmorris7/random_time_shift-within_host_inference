"""
This script generates some data from the stochastic TCL model and adds some observation noise in.
We generate the VL trajectories for all individuals assuming the same initial conditions.

Outputs are saved in the data/sims directory. The following files are generated:
    - data.csv: The noisy and true VL trajectories for each individual.
    - parameters.csv: The parameters for each individual.
    - hyper_parameters.csv: The hyperparameters/shared parameters for the model.
"""

# Include these once or things break
include("../pkgs.jl")
include("io.jl")
# Includes relative to this files location.
include("tcl/tcl_simulation.jl")
include("inference/priors.jl")
include("inference/within_host_inference.jl")

function log10p0(x)
    """
    Compute the log10 of x but clip to positive values to deal with instabilities
    in the ODE solutions (i.e. for small compartment counts near 0).
    """
    # Clip to positive values
    z = max(x, 0)
    if z > 1
        return log10(z)
    else
        return 0.0
    end
end

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

function tcl_deterministic!(dx, x, pars, t; S0 = S0)
    """
    Deterministic version of the TCL model.
    """
    R₀, k, δ, πv, c = pars
    s, e, i, v = x

    β = R₀ * δ * c / πv
    β′ = β / S0

    dx[1] = -β′ * v * s
    dx[2] = β′ * v * s - k * e
    dx[3] = k * e - δ * i
    dx[4] = πv * i - c * v

    return nothing
end

# X(t) = (U, E, I, V)
Z0 = [S0 - 1, 1, 0, 0]

LOD = 2.6576090679593496
ct_LOD = 40.0

u0 = Z0
obs_t = 50
tspan = (0, 50)
t_inf = 15.2
model_pars = deepcopy(μ)
prob = ODEProblem(tcl_deterministic!, u0, (0, 20), model_pars)
sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, save_idxs = 4)
sol2 = solve(prob, Tsit5(); save_idxs = 4)
sol3 = solve(prob, Tsit5(); save_idxs = 4, saveat = 0:1:20)

t_save = 0:1:20
V1 = log10p0.(sol.(t_save))
V2 = log10p0.(sol2.(t_save))
V3 = log10p0.(sol3.u)

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, t_save, V1 - V2, color = :red)
scatter!(ax, t_save, V2 - V3, color = :blue)
# scatter!(ax, t_save, V2, color = :red)
display(fig)

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

function get_gg_pars_nn(pars, Z0_bp, nn; S0 = S0)
    """
    Get the parameters of the generalised gamma distribution for the within-host model
    using the neural network.
    """
    R₀, k, δ, πv, c = pars

    β_bp = get_bp_β(R₀, k, δ, πv, c)

    # Calculate omega matrix and artefacts from that.
    Ω = [
        -k k 0
        0 -δ πv
        β_bp 0 -c
    ]

    λ, u_norm, _ = calculate_BP_contributions(Ω)
    # Calc expected value of W
    μ_w = dot(Z0_bp, u_norm)

    # Use the neural network to calculate the parameters of the generalised gamma distribution
    pars_m3 = nn([R₀, δ, πv])

    return μ_w, λ, pars_m3
end

function tcl_extinct_ode(q, pars, t; S0 = 8e7)
    """
    The ODEs for the extinction probability of the within-host model. This is derived
    from the backwards equations
    ∂F_i(s, t)/∂t = λ * [f_i(F(s, t)) - F_i(s, t)], for i = 1,..., n,
    where F(s, t) is the probability
    generating function of the number of infected cells at time t. f(s) is the offspring
    distribution.
    """
    R₀, k, δ, πv, c = pars

    β = get_ode_β(R₀, k, δ, πv, c)

    β = β * S0

    lifetimes = (k, δ + πv, c + β)

    d1 = lifetimes[1] * ((k * q[2]) / lifetimes[1] - q[1])
    d2 = lifetimes[2] * ((δ + πv * q[2] * q[3]) / lifetimes[2] - q[2])
    d3 = lifetimes[3] * ((c + β * q[1] * q[3]) / lifetimes[3] - q[3])

    return SA[d1, d2, d3]
end

function calculate_extinction_prob(pars)
    """
    Calculate the extinction probability of the within-host model.
    """
    q0 = SA[0.0, 0.0, 0.0]
    tspan = (0, 20)

    prob = ODEProblem(tcl_extinct_ode, q0, tspan, pars)
    sol = solve(prob, Tsit5(); save_start = false, save_everystep = false, save_end = true)

    return min(sol.u[1][1], 1.0)
end

function solve_quadratic(a, b, c)
    """
    Solve the quadratic equation ax^2 + bx + c = 0.
    """
    Δ = b^2 - 4 * a * c
    x1 = (-b + sqrt(Δ)) / (2 * a)
    x2 = (-b - sqrt(Δ)) / (2 * a)

    return x1, x2
end

function solve_exact_extinction_probs(pars)
    R₀, k, δ, πv, c = pars
    β = get_bp_β(R₀, k, δ, πv, c)

    a = [k, δ + πv, c + β]
    A = πv * a[3]
    B = β * δ - c * πv - a[2] * a[3]
    C = c * a[2]

    # x3 = solve_quadratic(A, B, C)

    x3 = solve_quadratic(A, B, C)
    x_out = ones(3)
    # Return the minimal non-negative solution
    for x in x3
        if x >= 0 && x < x_out[3]
            x_out[3] = x
        end
    end

    x_out[1] = δ / (a[2] - πv * x_out[3])
    x_out[2] = x_out[1]

    return x_out[1]
end

pars = [8.0, μ_k, μ_δ, μ_πv, μ_c]

##

extincts = zeros(1000)

for i in eachindex(extincts)
    _, _, extinct = tcl_gillespie(pars, Z0)
    extincts[i] = extinct
end

mean(extincts)

##

function get_μ(t, τ, sol)
    t_inf = sol.t[1]

    t_eval = t + τ

    # We first need to handle whether the individual is actual infected
    # or not. If the time is before the infection time, set the viral load to 0
    # We also set the viral load to 0 if the actual evaluation time is before the
    # infection time too since sol(t + τ) for t + τ < t_inf is not valid.
    if t < t_inf || t_eval < t_inf
        return log10p0(sol.u[1])
    elseif t_eval > sol.t[end]
        return log10p0(sol.u[end])
    else
        return log10p0(sol(t_eval))
    end
end

function approx_sample_tcl(pars, t_inf, Z0_bp, nn, prob, T_obs)
    """
    Approximately sample the viral load trajectory for a given set of parameters using the time-shift
    methodology. Any observations before t_inf should be set to 0.
    """
    μ_w, λ, w_pars = get_gg_pars_nn(pars, Z0_bp, nn)
    q_star = calculate_extinction_prob(pars)

    # Generate the observation times which are based on when the individual is
    # actually infected. This could be mapped to just be the end of the current
    # days but for now we will just sample the whole trajectory.
    t_save = collect(t_inf:1:(t_inf + T_obs))

    # If the infection is extinct, return empty data
    goes_extinct = rand(Bernoulli(q_star))
    if goes_extinct
        return (t_save, zeros(Float64, length(t_save)), -Inf)
    end

    # Generate time shift and adjust simulation times
    w = sample_generalized_gamma(w_pars)
    τ = log(w / μ_w) / λ

    # Shifted simulation span which we solve the ode over but note that the
    # individual is not infected until t_inf i.e. sol.t[1]
    t_span = (t_inf, t_inf + T_obs)
    prob = remake(prob, p = pars, tspan = t_span)
    sol = solve(prob, Tsit5(); save_idxs = 4, abstol = 1e-8, reltol = 1e-8)

    y = zeros(Float64, length(t_save))

    for (i, t_obs) in enumerate(t_save)
        y[i] = get_μ(t_obs, τ, sol)
    end

    y[y .<= 0.0] .= 0.0  # Ensure no negative log values

    return (t_save, y, τ)
end

"""
Noisy up that data.
"""
function add_noise_vls(vls, κ; LOD = 2.6576090679593496)
    vls_noisy = deepcopy(vls)
    vls_noisy = rand.(Normal.(vls, κ))
    vls_noisy[vls_noisy .<= LOD] .= LOD
    return vls_noisy
end

##

T_obs = 70
Z0_bp = Z0[2:end]

o, y, τ = approx_sample_tcl([8.0, μ_k, μ_δ, μ_πv, μ_c], 0, Z0_bp, nn, prob, T_obs)

κ = 0.5
y_noisy = add_noise_vls(y, κ)

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, o, y, color = :black)
plot!(ax, o, y_noisy, color = :red)
display(fig)

##

fig = Figure()
ax = Axis(fig[1, 1])
for V_i in V_stoch
    lines!(ax, V_i[1], V_i[2], color = ("black", 0.2))
end
lines!(ax, sol.t, V_det, color = :red)

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

function sample_prior_params(priors, hyper_params)
    feasible_pars = false

    # Initialise parameters to "bad" values to start the loop
    R₀ = -1.0
    δ = -1.0
    πv = -1.0

    while !feasible_pars
        R₀ = hyper_params[:μ_R₀] + hyper_params[:σ_R₀] * randn()
        δ = hyper_params[:μ_δ] + hyper_params[:σ_δ] * randn()
        πv = hyper_params[:μ_πv] + hyper_params[:σ_πv] * randn()
        feasible_pars = R₀ > 0 && δ > 0 && πv > 0
    end

    # TODO: fix the need to pass parameters to this function...
    # infection_time = rand(priors[:infection_time](-10, 10))
    infection_time = rand(Normal(-5, 2.0))

    θ = [R₀, hyper_params[:μ_k], δ, πv, hyper_params[:μ_c]]

    return (θ, infection_time)
end

function is_sim_valid(o, y)
    if length(o) == 0
        return false
    end

    t_till_peak = argmax(y)
    # Check number of non-lod observations as a surrogate for duration of infection
    duration = length(y[y .> 0])
    return any(y .> 4) && (2 <= t_till_peak <= 10) && (duration < 30)
end

function sim_till_valid(priors, hyper_params, Z0_bp, nn, prob, T_obs)
    # Run initial sim prior to loop
    θ, t_inf = sample_prior_params(priors, hyper_params)
    # θ = [7.821054874524198, 4.0, 1.2463613215551794, 2.906378494546395, 10.0]
    # t_inf = -9.121253065475306

    o, y, τ = approx_sample_tcl(θ, t_inf, Z0_bp, nn, prob, T_obs)

    valid_sim = is_sim_valid(o, y)

    while !valid_sim
        θ, t_inf = sample_prior_params(priors, hyper_params)
        o, y, τ = approx_sample_tcl(θ, t_inf, Z0_bp, nn, prob, T_obs)

        # Generate data that looks reasonable. This means the peak time is not too late from the early observations
        # and the decline is not super slow (i.e. inconsistent with clearing of virus)
        valid_sim = is_sim_valid(o, y)
    end

    # pad 5 days of zeros at the start
    for _ in 1:5
        pushfirst!(o, o[1] - 1)
        pushfirst!(y, 0.0)
    end

    return (o, y, θ, t_inf, τ)
end

##

θ, t_inf = sample_prior_params(priors, hyper_params)
o, y = approx_sample_tcl(θ, t_inf, Z0_bp, nn, prob, T_obs)
# θ, t_inf = sample_prior_params(priors, hyper_params)
o1, y1 = approx_sample_tcl(θ, t_inf, Z0_bp, nn, prob, T_obs)

println(is_sim_valid(o, y))
println(is_sim_valid(o1, y1))

θs = stack([sample_prior_params(priors, hyper_params)[1] for _ in 1:10000])

hist(θs[3, :])

##

n_samples = 1000

fig = Figure()
ax = Axis(fig[1, 1])
for n in 1:n_samples
    o, y = approx_sample_tcl(θ, t_inf, Z0_bp, nn, prob, T_obs)
    if is_sim_valid(o, y)
        lines!(ax, o, y, color = (:black, 0.3))
    end
end
xlims!(floor(t_inf) - 2, 0)
vlines!(ax, [t_inf], color = :red)
display(fig)

##

o, y, θ, t_inf, τ = sim_till_valid(priors, hyper_params, Z0_bp, nn, prob, T_obs)
o2, y2, θ2, t_inf, τ = sim_till_valid(priors, hyper_params, Z0_bp, nn, prob, T_obs)

o_noisy = deepcopy(o)
y_noisy = add_noise_vls(y, κ)

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, o, y, color = :black)
plot!(ax, o2, y2, color = :black)
# plot!(ax, o_noisy, y_noisy, color = :red)
display(fig)

##

Random.seed!(2028)

N = 100
# N = 100

θ = deepcopy(μ)

const VVF = Vector{Vector{Float64}}
const VVI = Vector{Vector{Int}}

IDs = Vector{Int}()
ind_pars = Vector{Vector{Float64}}()
obs_times = Vector{Float64}()
vls = Vector{Float64}()
obs_vls = Vector{Float64}()

# Now generate the data for the N individuals
for i in 1:N
    # Draw parameters
    # θ = rand.(Truncated.(Normal.(μ, σs), 0.1, 100))
    # Fix latent period and infectious period of infected cells
    # θ[[2, 5]] .= μ[[2, 5]]
    (o, y, θ, t_inf, τ) = sim_till_valid(priors, hyper_params, Z0_bp, nn, prob, T_obs)

    # y_noisy = y
    y_noisy = add_noise_vls(y, κ)
    t_till_peak = argmax(y_noisy)
    # Shift the observation times back by the time taken to peak. This forces
    # the peak time to be 0 for all individuals.
    peak_time = o[t_till_peak]
    o = o .- peak_time
    # Then need to shift the infection time relative to the peak time
    t_inf = t_inf - peak_time

    id = fill(i, length(o))
    append!(IDs, id)
    push!(ind_pars, [i; t_inf; deepcopy(θ); τ])
    append!(obs_times, o)
    append!(vls, y)
    append!(obs_vls, y_noisy)
end

ind_pars = stack(ind_pars, dims = 1)

df_data = DataFrame(ID = IDs, t = obs_times, log_vl = vls, noisy_log_vl = obs_vls)

df_params = DataFrame(ind_pars, ["ID", "infection_time", "R₀", "k", "δ", "πv", "c", "τ"])

##

λs = zeros(N)

for i in 1:N
    (R₀, k, δ, πv, c) = ind_pars[i, 3:7]
    β = R₀ * δ * c / πv
    # Calculate omega matrix and artefacts from that.
    Ω = [
        -k k 0
        0 -δ πv
        β 0 -c
    ]

    λ, u_norm, _ = calculate_BP_contributions(Ω)
    λs[i] = λ * u_norm[3]
end

hist(λs)

x = 0:6
y = [λ * x for λ in λs]

fig = Figure()
ax = Axis(fig[1, 1])
for y_i in y
    lines!(ax, x, y_i, color = :black, alpha = 0.1)
end
display(fig)

##

check_exist_mkdir(data_dir("sims"))
CSV.write(data_dir("sims/data.csv"), df_data)
CSV.write(data_dir("sims/parameters.csv"), df_params)

all_hyper_params = [μ[1], σs[1], μ[2], 0, μ[3], σs[3], μ[4], σs[4], μ[5], 0, κ]'

param_labels = ["μ_R₀", "σ_R₀", "μ_k", "σ_k", "μ_δ", "σ_δ", "μ_πv", "σ_πv", "μ_c", "σ_c", "κ"]

df_hyper_params = DataFrame(all_hyper_params, param_labels)

CSV.write(data_dir("sims/hyper_parameters.csv"), df_hyper_params)

##

fig = Figure()
axs = [Axis(fig[i, 1]) for i in 1:3]
hist!(axs[1], df_params.R₀)
hist!(axs[2], df_params.δ)
hist!(axs[3], df_params.πv)

display(fig)

##

# Overlay the trajectories to see differeences
fig = Figure()
ax = Axis(fig[1, 1])
for i in 1:100
    df_view = df_data[df_data.ID .== i, :]
    # scatter!(ax, df_view.t, df_view.noisy_log_vl, color = "black", alpha = 0.5)
    lines!(ax, df_view.t, df_view.log_vl, color = "red")
end
display(fig)

##

# Plot some of the trajectories
fig = Figure(size = (1000, 1000))
axs = [Axis(fig[i, j]) for i in 1:7, j in 1:8]
for (i, ax) in enumerate(axs)
    if i > 50
        break
    end
    df_view = df_data[df_data.ID .== i, :]
    scatter!(ax, df_view.t, df_view.noisy_log_vl, color = "black", alpha = 0.5)
    scatter!(ax, df_view.t, df_view.log_vl, color = "red", alpha = 0.5)
    xlims!(ax, -10, 10)
end
display(fig)
