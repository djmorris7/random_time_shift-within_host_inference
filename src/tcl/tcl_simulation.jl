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
    The deterministic version of the within-host model.
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

function tcl_extinct_ode!(dq, q, pars, t; S0 = 8e7)
    """
    The ODEs for the extinction probability of the within-host model.
    """
    R₀, k, δ, πv, c = pars

    β_bp = get_bp_β(R₀, k, δ, πv, c)

    lifetimes = (k, δ + πv, c + β_bp)

    dq[1] = lifetimes[1] * ((k * q[2]) / lifetimes[1] - q[1])
    dq[2] = lifetimes[2] * ((δ + πv * q[2] * q[3]) / lifetimes[2] - q[2])
    dq[3] = lifetimes[3] * ((c + β_bp * q[1] * q[3]) / lifetimes[3] - q[3])

    return nothing
end

function calculate_extinction_prob(pars)
    """
    Calculate the extinction probability of the within-host model.
    """
    q0 = [0.0, 0.0, 0.0]
    tspan = (0, 50)

    prob = ODEProblem(tcl_extinct_ode!, q0, tspan, pars)
    sol = solve(
        prob,
        Tsit5();
        save_start = false,
        save_everystep = false,
        save_end = true,
        abstol = 1e-9,
        reltol = 1e-9
    )

    q0 .= sol.u[1]

    return q0
end

function check_if_tau_leap(Z; Ω = 100)
    """
    Checks whether E, I, or V exceed a threshold Ω.
    """

    _, E, I, V = Z
    return (E > Ω) || (I > Ω) || (V > Ω)
end

function get_rates!(a, pars, Z; S0 = 8e7)
    """
    Sets the rates for the TCL model.
    """

    R₀, σ, γ, p_v, c_v = pars
    β = get_ode_β(R₀, σ, γ, p_v, c_v)

    S, E, I, V = Z
    a[1] = β * V * S
    a[2] = σ * E
    a[3] = γ * I
    a[4] = p_v * I
    a[5] = c_v * V

    return nothing
end

function tau_leap_step!(Z, Δt, Q, a)
    """
    Perform a tau leap step for the TCL model.
    """

    for i in eachindex(Q)
        ne = rand(Poisson(Δt * a[i]))
        Z .+= ne * Q[i]
    end

    return nothing
end

function ssa_step!(Z, a, Q)
    """
    Perform a single step of the Gillespie algorithm for the TCL model.
    """

    cumsum!(a, a)
    a0 = a[end]
    ru = rand() * a0
    event_idx = findfirst(x -> x > ru, a)
    Z .+= Q[event_idx]

    return nothing
end

function tcl_gillespie(pars, Z0; tf = 100.0, Δt = 0.01, t0 = 0.0, V_min = 1e5, save_at = 1.0)
    """
    Simulate a competing birth-death model in a continous time framework. This will save the
    state of the system at every save_at time steps up to the total time tf.
    """

    Z = deepcopy(Z0)
    tf_adj = t0 + tf

    # Calculate the actual number of steps
    n = ceil(Int, (t0 + tf) / save_at)
    Z_mat = Vector{Vector{Float64}}()
    t_vec = Vector{Float64}()
    # Propensity vector
    a = zeros(Float64, 5)
    # Representation of the stoichiometry matrix
    # X(t) = (U, E, I, V)
    Q = [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1]]

    # Store initial state
    # in the general case, the time will be t0 ∈ [day, day + 1) so we cannot just use t0
    # but we are still interested in t0 so we'll output it as element 1 but note that t_vec[2] - t_vec[1] ≠ 1
    push!(t_vec, t0)
    push!(Z_mat, deepcopy(Z))

    t = t0
    if mod(save_at, 1) == 0
        curr_t = ceil(t0 + eps())
    else
        curr_t = t0 + save_at
    end
    curr_ind = 2

    extinct = true
    a0 = 0.0

    while (t < tf_adj) && !iszero(sum(Z[2:end]))
        get_rates!(a, pars, Z)
        tau_leap = check_if_tau_leap(Z)

        # If we're using the tau leap method, we know the time-step otherwise we need to calculate it
        if tau_leap
            dt = Δt
        else
            a0 = sum(a)
            dt = -log(rand()) / a0
        end

        t += dt

        while t > curr_t && curr_ind <= n
            push!(t_vec, curr_t)
            push!(Z_mat, deepcopy(Z))
            curr_ind += 1
            curr_t += save_at
        end

        if tau_leap
            tau_leap_step!(Z, Δt, Q, a)
        else
            ssa_step!(Z, a, Q)
        end

        # Adjust compartment counts to reasonable values
        Z = max.(Z, 0)

        # Check that V = Z[4] is reasonable (whatever that means) for the simulation to be considered not extinct
        if Z[4] > V_min && extinct
            extinct = false
        end
    end

    return t_vec, Z_mat, extinct
end

function tcl_gillespie_end_state(
    pars, Z0; tf = 100.0, Δt = 0.01, t0 = 0.0, V_min = 1e5, save_at = 1.0
)
    """
    Simulate a competing birth-death model in a continous time framework. This will save the
    state of the system at every save_at time steps up to the total time tf.
    """

    Z = deepcopy(Z0)
    tf_adj = t0 + tf

    # Calculate the actual number of steps
    n = ceil(Int, (t0 + tf) / save_at)
    Z_mat = Vector{Vector{Float64}}()
    t_vec = Vector{Float64}()
    # Propensity vector
    a = zeros(Float64, 5)
    # Representation of the stoichiometry matrix
    # X(t) = (U, E, I, V)
    Q = [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1]]

    # Store initial state
    # in the general case, the time will be t0 ∈ [day, day + 1) so we cannot just use t0
    # but we are still interested in t0 so we'll output it as element 1 but note that t_vec[2] - t_vec[1] ≠ 1
    push!(t_vec, t0)
    push!(Z_mat, deepcopy(Z))

    t = t0
    if mod(save_at, 1) == 0
        curr_t = ceil(t0 + eps())
    else
        curr_t = t0 + save_at
    end
    curr_ind = 2

    extinct = true
    a0 = 0.0

    while (t < tf_adj) && !iszero(sum(Z[2:end]))
        get_rates!(a, pars, Z)
        tau_leap = check_if_tau_leap(Z)

        # If we're using the tau leap method, we know the time-step otherwise we need to calculate it
        if tau_leap
            dt = Δt
        else
            a0 = sum(a)
            dt = -log(rand()) / a0
        end

        t += dt

        while t > curr_t && curr_ind <= n
            push!(t_vec, curr_t)
            push!(Z_mat, deepcopy(Z))
            curr_ind += 1
            curr_t += save_at
        end

        if tau_leap
            tau_leap_step!(Z, Δt, Q, a)
        else
            ssa_step!(Z, a, Q)
        end

        # Adjust compartment counts to reasonable values
        Z = max.(Z, 0)

        # Check that V = Z[4] is reasonable (whatever that means) for the simulation to be considered not extinct
        if Z[4] > V_min && extinct
            extinct = false
        end
    end

    return t_vec, Z_mat, extinct
end
