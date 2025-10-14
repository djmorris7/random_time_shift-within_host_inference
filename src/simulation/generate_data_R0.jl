"""
This script generates some data from the stochastic TCL model and adds some observation noise in.
We generate the VL trajectories for all individuals assuming the same initial conditions.

Outputs are saved in the data/sims directory. The following files are generated:
    - data.csv: The noisy and true VL trajectories for each individual.
    - parameters.csv: The parameters for each individual.
    - hyper_parameters.csv: The hyperparameters/shared parameters for the model.
"""

# Include these once or things break
include("../../pkgs.jl")
include("../io.jl")
# Includes relative to this files location.
include("../tcl/tcl_simulation.jl")
include("../inference/priors.jl")
include("../inference/within_host_inference.jl")

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

    # y[y .<= 0.0] .= 0.0  # Ensure no negative log values

    return (t_save, y, τ)
end

"""
Noisy up that data.
"""
function add_noise_vls(o, vls, κ; LOD = 2.6576090679593496, sparse_sampling = false)
    o_noisy = deepcopy(o)
    vls_noisy = deepcopy(vls)
    vls_noisy = rand.(Normal.(vls, κ))
    vls_noisy_no_LOD = deepcopy(vls_noisy)
    vls_noisy[vls_noisy .<= LOD] .= LOD

    keep_mask = trues(length(vls_noisy))

    if sparse_sampling
        above_LOD_idxs = findall(vls_noisy .> LOD)
        # Remove all above LOD values and add in later
        keep_mask[above_LOD_idxs] .= false

        n_keep = rand(3:length(above_LOD_idxs))
        keep_idxs = sort(sample(above_LOD_idxs, n_keep, replace = false))
        keep_mask[keep_idxs] .= true

        # Keep only the relevant observations
        o_noisy = o_noisy[keep_mask]
        vls_noisy = vls_noisy[keep_mask]
    end

    return o_noisy, vls_noisy, vls_noisy_no_LOD
end

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
        pushfirst!(y, log10p0(0.0))
    end

    return (o, y, θ, t_inf, τ)
end
