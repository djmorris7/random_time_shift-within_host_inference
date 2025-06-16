include("../tcl/timeshift.jl")

function metrics_τ(pars_m3, Z0_bp, q, μ_w, λ)
    # samples = τ_dist_from_w(pars_m3, Z0_bp, q, μ_w, λ)
    # Timeshift pdf
    f = τ -> τ_prior(τ, pars_m3, μ_w, λ)
    μ = -Inf
    σ2 = -Inf
    quants = [-Inf, -Inf]

    try
        μ = quadgk(τ -> τ * f(τ), -7, 7)[1]
        σ2 = quadgk(τ -> (τ - μ)^2 * f(τ), -7, 7)[1]
        # Get a noisy sample of the distribution
        samples = [sample_generalized_gamma(pars_m3) for _ in 1:1000]
        quants = quantile(samples, [0.025, 0.975])
    catch
        # raise warning
        @warn "Error in calculating moments of the timeshift distribution."

        μ = -Inf
        σ2 = -Inf
        quants = [-Inf, -Inf]
    end

    return μ, σ2, quants
end

function moments_gengamma(n, a, d, p)
    # if (gamma((d + n) / p) == 0) || !isfinite(gamma(d / p)) || (gamma(d / p) == 0)
    # if isfinite(gamma(d / p))
    if a > 0 && d > 0 && p > 0
        return exp(n * log(a) + (loggamma((d + n) / p) - loggamma(d / p)))
    else
        return Inf
    end
end

function loss_func(pars, moments, q; num_moments_loss = 5)
    log_a, log_d, log_p = pars

    loss = 0.0

    for i in 1:num_moments_loss
        μ_n = moments[i] / (1 - q)
        pred = moments_gengamma(i, exp.(log_a), exp.(log_d), exp.(log_p))
        loss += ((pred - μ_n) / μ_n)^2
    end

    return loss
end

function minimise_loss(moments, q)
    """
    Custom implementation of the loss function minimisation for returning the optimum
    as well as the loss value. This is done to ensure that the loss function is minimised
    correctly.
    """
    # Initial guess reflects an exponential distribution p = d = 1 with rate 1
    x0 = log.(ones(3))
    l = x -> loss_func(x, moments[:, 1], q[1])
    sol = Optim.optimize(l, x0, Newton(); autodiff = :forward)

    return exp.(sol.minimizer), sol.minimum
end

function get_ode_β(R₀, k, δ, πv, c; S0 = 8e7)
    """
    Calculate the infection rate β from the other parameters.
    """
    return δ * c * R₀ / (πv * S0)
end

function get_bp_β(R₀, k, δ, πv, c; S0 = 8e7)
    """
    Calculate the infection rate β from the other parameters.
    """
    return δ * c * R₀ / πv
end

function get_gg_pars(pars, Z0_bp; S0 = 8e7)
    """
    Calculate the parameters of the GG distribution for the within-host model.
    """
    R₀, k, δ, πv, c = pars
    # β, k, δ, πv, c = pars

    β_bp = get_bp_β(R₀, k, δ, πv, c)

    # Calculate omega matrix and artefacts from that.
    Ω = [
        -k k 0
        0 -δ πv
        β_bp 0 -c
    ]

    λ, u_norm, v_norm = calculate_BP_contributions(Ω)

    # Calc expected value of W
    μ_w = dot(Z0_bp, u_norm)

    # Get the first 5 moments
    num_moments = 5
    lifetimes = (k, δ + πv, c + β_bp)

    αs = Dict(1 => Dict([1, 2] => k))
    βs = Dict(2 => Dict([2, 2, 3] => πv), 3 => Dict([3, 1, 3] => β_bp))

    moments = calculate_moments_generic(Ω, αs, βs, lifetimes)

    q = calculate_extinction_prob(pars)

    # Using the moments, calculate the parameters of the GG distribution and sample
    pars_m3, loss = minimise_loss(moments, q)

    fitted_pars = Dict("pars_m3" => pars_m3, "loss" => loss, "q" => q, "λ" => λ, "μ_w" => μ_w)

    return fitted_pars
end

function sample_params(Z0_bp = Z0_bp)
    """
    Samples parameter values in terms of the BP scale, i.e. β = β' * S0 where β' is the
    actual parameter value. This is done to ensure that the parameters are on a similar scale
    """
    R₀ = rand(Uniform(1, 35))
    δ = rand(Uniform(0.01, 10))
    πv = rand(Uniform(0.01, 10))
    # k = rand(Uniform(0.01, 10))
    # k and c fixed in the inference model
    k = 4.0
    c = 10.0
    # c = rand(Uniform(0.1, 20))
    pars = (R₀, k, δ, πv, c)

    are_pars_valid = true

    fitted_pars = try
        get_gg_pars(pars, Z0_bp)
    catch
        are_pars_valid = false
    end

    return pars, fitted_pars, are_pars_valid
end

function solve_for_peak_time(pars; Z0 = [8e7 - 1, 1, 0, 0])
    tspan = (0, 30)
    prob = ODEProblem(tcl_deterministic!, Z0, tspan, pars)
    sol = solve(prob, Tsit5(); save_idxs = 4)
    peak_time = sol.t[findmax(sol.u)[2]]
    return peak_time
end

function check_peak_time_valid(peak_time)
    return peak_time < 14
end

function check_data_valid(pars, fitted_pars, Z0_bp, q_star)
    """
    This function checks to see whether the data generated is valid. This is done by checking
    the following conditions:
    - λ > 0: i.e. viral count growing
    - q_star < 0.99: i.e. the infection doesn't have too high a probability of going extinct
    - The mean of the time-shift is too late or early relative to a typical mean curve starting
      at t = 0.
    - The peak time is not too late. This can often be asserted from data or previous studies
      and so we basically want parameter values that produce this within a reasonable range.
    """
    λ = fitted_pars["λ"]
    q = fitted_pars["q"]
    μ_w = fitted_pars["μ_w"]
    pars_m3 = fitted_pars["pars_m3"]

    λ < 0 && return false
    q_star > 0.99 && return false
    peak_time = solve_for_peak_time(pars)
    peak_time >= 14 && return false
    # For now there's a strange issue with p -> Inf which leads to numerical issues
    pars_m3[3] > 100 && return false
    # Check to see whether the mean of the time-shift is too late or early relative
    # to a typical mean curve starting at t = 0.
    μ_τ, var_τ, quants_τ = metrics_τ(pars_m3, Z0_bp, q, μ_w, λ)
    τ_dist_valid = (-7 < μ_τ < 7) && (0.01 <= var_τ <= 7) && (quants_τ[1] > -7 && quants_τ[2] < 7)
    !τ_dist_valid && return false

    return true
end

function generate_data(n_samples, Z0_bp = [1, 0, 0])
    """
    Generates the training and validation data for the neural network. This is done by
    sampling the parameter values and then calculating the moments of the distribution
    and the extinction probability. We then calculate the parameters of the GG distribution
    and sample from it. We then check to see whether the mean of the time-shift is too late
    or early relative to a typical mean curve starting at t = 0.
    """
    check_n_its = ceil(Int, n_samples / 100)

    # input = zeros(Float64, n_samples, 5)
    input = zeros(Float64, n_samples, 3)
    output = zeros(Float64, n_samples, 3)
    data_measures = zeros(Float64, n_samples, 4)

    i = 1
    while i <= n_samples
        pars, fitted_pars, are_pars_valid = sample_params(Z0_bp)
        !are_pars_valid && continue
        # If the value is -Inf, then the parameters are invalid and we shouldn't hit this step
        q_star = sum(fitted_pars["q"] .* Z0_bp)

        # Print out the parameters to see what's going on
        # println(pars)

        if check_data_valid(pars, fitted_pars, Z0_bp, q_star)
            data_measures[i, :] .= [
                fitted_pars["μ_w"], fitted_pars["λ"], q_star, fitted_pars["loss"]
            ]
            # data_measures[i, :] .= [μ_w, λ, q_star]
            input[i, :] .= pars[[1, 3, 4]] # Only use β, δ, πv
            output[i, :] .= fitted_pars["pars_m3"]
            i += 1

            if mod(i, check_n_its) == 0
                curr_percent = round(i / n_samples * 100)
                @info "Percent complete: $curr_percent%"
            end
        end
    end

    df_input = DataFrame(input, :auto)
    df_output = DataFrame(output, :auto)
    df_measures = DataFrame(data_measures, :auto)

    check_exist_mkdir(data_dir("tcl_nn"))

    CSV.write(data_dir("tcl_nn/gg_fitting_data.csv"), df_input)
    CSV.write(data_dir("tcl_nn/gg_fitting_data_output.csv"), df_output)
    CSV.write(data_dir("tcl_nn/gg_fitting_data_measures.csv"), df_measures)

    return nothing
end

function read_data()
    """
    Reads in the data generated by the `generate_data` function and returns it as a dictionary.
    """
    df_input = CSV.read(data_dir("tcl_nn/gg_fitting_data.csv"), DataFrame)
    df_output = CSV.read(data_dir("tcl_nn/gg_fitting_data_output.csv"), DataFrame)
    df_measures = CSV.read(data_dir("tcl_nn/gg_fitting_data_measures.csv"), DataFrame)
    input = Matrix(df_input)
    output = Matrix(df_output)
    data_measures = Matrix(df_measures)

    return Dict("input" => input, "output" => output, "data_measures" => data_measures)
end
