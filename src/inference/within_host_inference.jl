"""
This is the main file that includes all the machinery for inference of the hierarchical model.
All imports for the actual inference come through here and are subsequently included into the
`within_host_inference_run.jl` and `within_host_inference_nba_run.jl` which is the entry point for the code.
"""

include("../../pkgs.jl")
include("data_structures.jl")
include("priors.jl")
include("../tcl/tcl_simulation.jl")
include("../io.jl")
include("data_processing.jl")

# --- ODE model ---

function get_bp_β(R₀, k, δ, πv, c; S0 = 8e7)
    """
    Calculate the infection rate β from the other parameters.
    """
    return δ * c * R₀ / πv
end

function get_ode_β(R₀, k, δ, πv, c; S0 = 8e7)
    """
    Calculate the infection rate β from the other parameters.
    """
    return δ * c * R₀ / (πv * S0)
end

function tcl_deterministic!(dx, x, pars, t; S0 = 8e7)
    """
    The ODE system for the within-host model.
    """
    R₀, k, δ, πv, c = pars
    s, e, i, v = x

    β = get_ode_β(R₀, k, δ, πv, c)

    dx[1] = -β * v * s
    dx[2] = β * v * s - k * e
    dx[3] = k * e - δ * i
    dx[4] = πv * i - c * v

    return nothing
end

function tcl_deterministic(x, pars, t; S0 = 8e7)
    """
    The ODE system for the within-host model.
    """
    R₀, k, δ, πv, c = pars

    β = get_ode_β(R₀, k, δ, πv, c)

    s, e, i, v = x

    ds = -β * v * s
    de = β * v * s - k * e
    di = k * e - δ * i
    dv = πv * i - c * v

    return SA[ds, de, di, dv]
end

# --- Neural network ---

function load_nn()
    """
    Loads the pre-trained nn model and the scalings for the data. Scalings are used to
    rescale the data before passing it to the neural network.
    """
    # Load the neural network
    model_state = JLD2.load("data/tcl_timeshift_nn.jld2", "model_state")
    # nn = Flux.Chain(Dense(3 => 64, relu), Dense(64 => 64, relu), Dense(64 => 3))
    nn = Flux.Chain(Dense(3 => 64, relu), Dense(64 => 64, relu), Dense(64 => 3), softplus)
    Flux.loadmodel!(nn, model_state)

    # Convert the model to use eltype Float64
    nn = Flux.f64(nn)

    return nn
end

function get_gg_pars_nn(θ::Params, M::ModelInternals)
    """
    Uses the neural network to calculate the parameters of the generalised gamma distribution.
    """
    @unpack_Params θ
    @unpack Z0_bp, nn, S0 = M

    β_bp = get_bp_β(R₀, k, δ, πv, c)

    # Calculate omega matrix and artefacts from that.
    Ω = SA[
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

function get_gg_pars(θ::Params, M::ModelInternals)
    """
    Uses the neural network to calculate the parameters of the generalised gamma distribution.
    """
    @unpack_Params θ
    @unpack Z0_bp, nn, S0 = M

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
    # pars_m3 = nn(pars)
    pars = [R₀, k, δ, πv, c]
    q = calculate_extinction_prob(pars)

    lifetimes = (k, δ + πv, c + β)

    αs = Dict(1 => Dict([1, 2] => k))
    βs = Dict(2 => Dict([2, 2, 3] => πv), 3 => Dict([3, 1, 3] => β))

    moments = calculate_moments_generic(Ω, αs, βs, lifetimes)

    # return moments, q
    # pars_m3 = minimise_loss_single(moments, q)
    # pars_m3 = minimise_log_loss(moments, q)
    pars_m3 = minimise_loss(moments, q)
    # pars_m3 = minimise_log_loss(moments, [q, q, q])

    return μ_w, λ, pars_m3
end

# --- Likelihood and priors ---
function measurement_model(y, μ, κ; lod = 2.65761, epsilon = 1e-3)
    if y <= lod
        return logcdf(Normal(μ, κ), lod)
    else
        return logpdf(Normal(μ, κ), y)
    end
end

function log10p0(x)
    """
    Compute the log10 of x but clip to positive values to deal with instabilities
    in the ODE solutions (i.e. for small compartment counts near 0).
    """
    # Clip to positive values
    z = max(x, 0)
    if z < 1
        return 0.0
    else
        return log10(z)
    end
end

"""
Solve ode over tspan to get f(t)
Shift the solution by τ get m(t) = f(t + τ) where t is the time si

m(0) = f(τ) which is not defined for τ < 0
"""

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

# t_inf = 3.0
# observation_times = 0:1:30
# τ = -3.0
# tspan = (t_inf, 40.0)
# prob = ODEProblem(tcl_deterministic, Z0_static, tspan, vl_model_pars)
# sol = solve(prob, Tsit5(); save_idxs = 4)

# fig = Figure()
# ax = Axis(fig[1, 1])
# plot!(ax, sol.t .- τ, log10p0.(sol.u))
# plot!(ax, observation_times, m)
# xlims!(ax, 0, 20)
# vlines!(ax, t_inf)
# display(fig)

# m = [get_μ(t, τ, sol) for t in observation_times]

function path_likelihood(τ, κ, infection_time, individual_data::IndividualData, sol)
    """
    The log likelihood for a vl trajectory given a time shift τ.
    """
    @unpack obs_times, vl = individual_data

    pdf_val = 0.0

    for (t, y) in zip(obs_times, vl)
        μ = get_μ(t, τ, sol)
        pdf_val += measurement_model(y, μ, κ)
    end

    return pdf_val
end

function ddx(f, x; h = 1e-5)
    return (f(x + h) - f(x - h)) / (2 * h)
end

function d2dx2(f, x; h = 1e-5)
    return (f(x + h) - 2 * f(x) + f(x - h)) / h^2
end

function create_laplace_approx_func(
    θ::Params, individual_data::IndividualData, M::ModelInternals, ϕ::SharedParams
)
    """
    Creates the Laplace approximation function for the likelihood.
    """
    @unpack κ = ϕ
    @unpack infection_time = θ
    @unpack sol, LOD = M

    # Closures for computing the negative log likelihood and its derivatives
    neg_ℓ = τ -> -path_likelihood(τ, κ, infection_time, individual_data, sol)
    neg_ℓ′ = τ -> ForwardDiff.derivative(neg_ℓ, τ)
    neg_ℓ′′ = τ -> ForwardDiff.derivative(neg_ℓ′, τ)
    # Maximum of the timeshift dist should be around 0, so optimize in the vincinity of 0
    opt = Optim.optimize(neg_ℓ, -5, 5)
    τ_0 = opt.minimizer
    ℓ_0 = -opt.minimum
    v2 = max(1e-8, 1.0 / neg_ℓ′′(τ_0))

    return ℓ_0, τ_0, v2
end

function log_integrate_likelihood(f)
    """
    Integrate the log likelihood function.
    """
    # Sometimes instabilities and unlikely par combos get through to
    # the ODE solver and cause the integration to fail. Such a situation
    # is handled by setting the log-likelihood to -Inf.
    try
        return log(quadgk(f, -7, 7)[1])
    catch
        return -Inf
    end
end

function laplace_approx_likelihood(
    θ::Params, individual_data::IndividualData, ϕ::SharedParams, M::ModelInternals
)
    """
    Computes the Laplace approximation for the log-likelihood and returns it.
    """
    ℓ_0, τ_0, v2 = create_laplace_approx_func(θ, individual_data, M, ϕ)
    p2_τ = x -> exp(-0.5 * (x - τ_0)^2 / v2)

    return (ℓ_0, p2_τ)
end

function log_gen_gamma_pdf(w, gg_pars)
    """
    The log probability density function for the generalised gamma distribution.
    """
    a, d, p = gg_pars
    log_pdf_val = log(p) - d * log(a) + (d - 1) * log(w) - (w / a)^p - loggamma(d / p)

    return log_pdf_val
end

function log_τ_prior(τ, gg_pars, μ_w, λ)
    """
    The log prior for the time shift parameter τ.
    """
    w = μ_w * exp(λ * τ)
    dτ = log(μ_w) + log(λ) + λ * τ
    log_pdf_val = dτ + log_gen_gamma_pdf(w, gg_pars)

    return log_pdf_val
end

function τ_prior(τ, gg_pars, μ_w, λ)
    """
    The log prior for the time shift parameter τ.
    """
    a, d, p = gg_pars
    pdf = p * λ * μ_w^d / (a^d * gamma(d / p)) * exp(-(μ_w * exp(λ * τ) / a)^p + λ * τ * d)
    return pdf
end

function get_τ_prior_nn(θ::Params, M::ModelInternals)
    """
    Computes the prior for the time shift parameter τ using the neural network.
    """
    μ_w, λ, pars_m3 = get_gg_pars_nn(θ, M)
    # μ_w, λ, pars_m3 = get_gg_pars(θ, M)
    # We are only interested in cases where λ > 0 since this is means that the
    # infection grows
    early_return = λ <= 0

    if early_return
        return nothing, early_return
    end

    p1_τ = x -> log_τ_prior(x, pars_m3, μ_w, λ)

    return p1_τ, early_return
end

function laplace_approx_only_likelihood(
    θ::Params, individual_data::IndividualData, ϕ::SharedParams, M::ModelInternals
)
    """
    Computes the likelihood for a single individual.
    """
    @unpack_Params θ
    @unpack_SharedParams ϕ

    ode_model_pars = (R₀, k, δ, πv, c)
    M.prob = remake(M.prob; p = ode_model_pars, tspan = (infection_time, infection_time + 40.0))
    M.sol = solve(M.prob, Tsit5(); save_idxs = 4, reltol = 1e-4)
    p1_τ, early_return = get_τ_prior_nn(θ, M)
    if early_return
        return -Inf
    end

    q = solve_exact_extinction_probs(ode_model_pars)
    ℓ_0, p2_τ = laplace_approx_likelihood(θ, individual_data, ϕ, M)
    p_τ = x -> exp(p1_τ(x)) * p2_τ(x)

    # log_like += log(1 - q)

    I = log_integrate_likelihood(p_τ)

    loglike = log(1 - q) + ℓ_0 + I

    return loglike
end

function exact_likelihood(
    θ::Params, individual_data::IndividualData, ϕ::SharedParams, M::ModelInternals
)
    """
    Computes the likelihood for a single individual.
    """
    @unpack_Params θ
    @unpack_SharedParams ϕ

    ode_model_pars = (R₀, k, δ, πv, c)
    M.prob = remake(M.prob; p = ode_model_pars, tspan = (infection_time, infection_time + 40.0))
    M.sol = solve(M.prob, Tsit5(); save_idxs = 4, reltol = 1e-4)
    p1_τ, early_return = get_τ_prior_nn(θ, M)

    if early_return
        return -Inf
    end
    # p2_τ = laplace_approx_likelihood(θ, individual_data, ϕ, M)
    q = solve_exact_extinction_probs(ode_model_pars)
    p2_τ = τ -> path_likelihood(τ, κ, infection_time, individual_data, M.sol)

    p_τ = x -> exp(p1_τ(x) + p2_τ(x))

    I = log_integrate_likelihood(p_τ)

    loglike = log(1 - q) + I

    return loglike
end

function create_laplace_approx_full_func(
    θ::Params, individual_data::IndividualData, M::ModelInternals, ϕ::SharedParams, p1_τ
)
    """
    Creates the Laplace approximation function for the likelihood.
    """
    @unpack κ = ϕ
    @unpack infection_time = θ
    @unpack sol, LOD = M

    # Closures for computing the negative log likelihood and its derivatives
    neg_ℓ = τ -> -path_likelihood(τ, κ, infection_time, individual_data, sol) - p1_τ(τ)
    neg_ℓ′ = τ -> ForwardDiff.derivative(neg_ℓ, τ)
    neg_ℓ′′ = τ -> ForwardDiff.derivative(neg_ℓ′, τ)
    # Maximum of the timeshift dist should be around 0, so optimize in the vincinity of 0
    opt = Optim.optimize(neg_ℓ, -5, 5)
    τ_0 = opt.minimizer
    ℓ_0 = -opt.minimum
    H = sqrt(max(1e-8, 1.0 / neg_ℓ′′(τ_0)))

    return (ℓ_0, τ_0, H)
end

function laplace_approx(
    θ::Params, individual_data::IndividualData, ϕ::SharedParams, M::ModelInternals, p1_τ
)
    """
    Computes the Laplace approximation for the log-likelihood and returns it.
    """
    ℓ_0, τ_0, H = create_laplace_approx_full_func(θ, individual_data, M, ϕ, p1_τ)
    p2_τ = ℓ_0 + 1 / 2 * log(2 * π) + log(H)
    return p2_τ
end

function tcl_extinct_ode(q, pars, t; S0 = 8e7)
    """
    The ODEs for the extinction probability of the within-host model.
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
    The ODEs for the extinction probability of the within-host model. This is derived
    from the backwards equations
    ∂F_i(s, t)/∂t = λ * [f_i(F(s, t)) - F_i(s, t)], for i = 1,..., n,
    where F(s, t) is the probability
    generating function of the number of infected cells at time t. f(s) is the offspring
    distribution.
    """
    q0 = SA[0.0, 0.0, 0.0]
    tspan = (0, 10)

    prob = ODEProblem(tcl_extinct_ode, q0, tspan, pars)
    sol = solve(prob, Tsit5(); save_start = false, save_everystep = false, save_end = true)

    return sol.u[1][1]
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
    """
    These equations come from solving f_i(q) = q_i for i = 1, 2, 3.
    """

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

##

function laplace_approx_full_likelihood(
    θ::Params, individual_data::IndividualData, ϕ::SharedParams, M::ModelInternals
)
    """
    Computes the likelihood for a single individual.
    """
    @unpack_Params θ
    @unpack_SharedParams ϕ

    ode_model_pars = (R₀, k, δ, πv, c)
    M.prob = remake(M.prob; p = ode_model_pars, tspan = (infection_time, infection_time + 40.0))
    # Using a smaller relative tolerance to ensure that the approximation does not become
    # discontinuous through numerical instabilities.
    M.sol = solve(M.prob, Tsit5(); save_idxs = 4, reltol = 1e-4)
    # M.sol = solve(M.prob, Tsit5(); save_idxs = 4)
    p1_τ, early_return = get_τ_prior_nn(θ, M)
    if early_return
        return -Inf
    end

    p_τ = laplace_approx(θ, individual_data, ϕ, M, p1_τ)
    # q = calculate_extinction_prob(ode_model_pars)
    q = solve_exact_extinction_probs(ode_model_pars)

    # println(q1, " ", q)

    # return p_τ

    return log(1 - q) + p_τ
end

function likelihood(θ::Params, individual_data::IndividualData, ϕ::SharedParams, M::ModelInternals)
    """
    Computes the likelihood for a single individual.
    """
    loglike = laplace_approx_full_likelihood(θ, individual_data, ϕ, M)
    # loglike = exact_likelihood(θ, individual_data, ϕ, M)
    # loglike = laplace_approx_only_likelihood(θ, individual_data, ϕ, M)
    return loglike
end

function joint_likelihood(
    all_params::Vector{Params}, data::Vector{IndividualData}, ϕ::SharedParams, M::ModelInternals
)
    """
    Computes the likelihood for all individuals.
    """
    like = 0.0

    for (θ, individual_data) in zip(all_params, data)
        like += likelihood(θ, individual_data, ϕ, M)
    end

    return like
end

# --- Parameter priors ---

function individual_prior(
    θ::Params, ϕ::SharedParams; priors::Dict = priors, fixed_params::Dict = fixed_params
)
    """
    Computes the log prior for the individual parameters.
    This requires the Params objects and uses symbols to easily index into the
    priors dictionary.
    """

    log_p = 0.0

    for x in fieldnames(Params)
        if !(x in keys(priors)) || fixed_params[x]
            continue
        end
        log_p_tmp = 0.0

        x_val = getfield(θ, x)
        prior = priors[x]

        if x == :infection_time
            a, b = θ.infection_time_range
            log_p_tmp += logpdf(prior, x_val)
            # log_p_tmp += logpdf(prior(a, b), x_val)
        elseif x ∈ [:z_R₀, :z_δ, :z_πv]
            log_p_tmp += logpdf(prior, x_val)
        end

        log_p += log_p_tmp
    end

    return log_p
end

function shared_prior(
    ϕ::SharedParams; hyper_priors::Dict = hyper_priors, fixed_params::Dict = fixed_params
)
    """
    Computes the log prior for the shared parameters. This requires the SharedParams
    object and uses symbols to easily index into the hyper_priors dictionary.
    """

    log_p = 0.0

    for x in fieldnames(SharedParams)
        if !(x in keys(hyper_priors)) || fixed_params[x]
            continue
        end
        log_p_tmp = 0.0

        # Get appropriate value and prior
        x_val = getfield(ϕ, x)
        prior = hyper_priors[x]

        log_p_tmp = logpdf(prior, x_val)

        log_p += log_p_tmp
    end

    return log_p
end
