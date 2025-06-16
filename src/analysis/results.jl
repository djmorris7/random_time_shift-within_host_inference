# include("../inference/within_host_inference.jl")
include("../inference/within_host_inference.jl")

function get_gg_pars_nn(pars, Z0_bp, nn_func; S0 = S0)
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
    pars_m3 = nn_func([R₀, δ, πv])

    return μ_w, λ, pars_m3
end

# function tcl_deterministic!(dx, x, pars, t; S0 = S0)
#     """
#     Deterministic version of the TCL model.
#     """
#     β, σ, γ, p_v, c_v = pars
#     s, e, i, v = x

#     # β′ = β / S0

#     # dx[1] = -β′ * v * s
#     # dx[2] = β′ * v * s - σ * e
#     dx[1] = -β * v * s
#     dx[2] = β * v * s - σ * e
#     dx[3] = σ * e - γ * i
#     dx[4] = p_v * i - c_v * v

#     return nothing
# end

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

function approx_sample_tcl(pars, t_inf, Z0_bp, nn, prob, T_obs; Δt = 1.0)
    """
    Approximately sample the viral load trajectory for a given set of parameters using the time-shift
    methodolgy.
    """
    μ_w, λ, w_pars = get_gg_pars_nn(pars, Z0_bp, nn)
    w = sample_generalized_gamma(w_pars)
    τ = log(w / μ_w) / λ

    t_span = (t_inf, t_inf + T_obs)

    R₀, k, δ, πv, c = pars

    prob = remake(prob; p = pars, tspan = t_span)
    sol = solve(prob, Tsit5(); save_idxs = 4)

    t_save = ceil(t_inf):Δt:ceil(t_inf + T_obs)

    o = zeros(Float64, length(t_save))
    y = zeros(Float64, length(t_save))

    for (i, t) in enumerate(t_save)
        o[i] = t

        y[i] = get_μ(t, τ, sol)
    end

    return o, y
end

function add_noise_vls(vls, κ; lod = 2.6576090679593496)
    """
    Noisy up that data.
    """
    vls_noisy = rand.(Normal.(vls, κ))
    vls_noisy[vls_noisy .<= lod] .= lod
    # vls_noisy[vls_noisy .<= LOD] .= 0.0
    return vls_noisy
end

function extract_individual_params(df_samples, id)
    """
    Pulls out the parameters from the samples DataFrame for a given individual.
    """
    selected_cols = [
        col for col in names(df_samples) if
        occursin("_$id", col) && (col[(end - length("_$id") + 1):end] == "_$id") && (col[1] != 'z')
    ]
    df_samples_ind = df_samples[:, selected_cols]

    # Rename the columns
    old_names = names(df_samples_ind)
    new_names = replace.(old_names, "_$id" => "")
    rename!(df_samples_ind, new_names)

    return df_samples_ind
end

function ppc_simulation(
    df_samples_ind,
    Z0_bp,
    nn_func,
    prob,
    κ,
    n_ppc_sims,
    T;
    t0 = 0.0,
    Δt = 1.0,
    lod = 2.6576090679593496
)
    """
    Perform a posterior predictive check simulation.
    """

    t_range = t0:Δt:(t0 + T - Δt)

    ppc_sims = lod * ones(Float64, length(t_range), n_ppc_sims + 1)

    # First column is the times
    ppc_sims[:, 1] .= t_range
    sim_inds = sample(1:size(df_samples_ind, 1), n_ppc_sims, replace = false)

    for sim in 1:n_ppc_sims
        pars_df = df_samples_ind[sim, :]
        # β = pars_df[1] * (pars_df[3] * pars_df[5]) / pars_df[4]

        θ = pars_df[1:(end - 1)]
        t_inf = pars_df[end]

        o, y = approx_sample_tcl(θ, t_inf, Z0_bp, nn_func, prob, 35; Δt = Δt)
        y_noisy = add_noise_vls(y, κ[sim])

        for (o_i, y_i) in zip(o, y_noisy)
            ind = findfirst(ppc_sims[:, 1] .== o_i)
            # Skip if we go off the end of the observation array
            isnothing(ind) && continue
            ppc_sims[ind, sim + 1] = y_i
        end
    end

    return ppc_sims
end

function summarise_ppc_sims(ppc_sims)
    """
    Produces a summary of the posterior predictive check simulations.
    """

    ppc_sims_summ = Dict()
    ppc_sims_summ["t"] = ppc_sims[:, 1]
    ppc_sims_summ["median"] = median(ppc_sims[:, 2:end], dims = 2)[:]
    ppc_sims_summ["bottom"] = [quantile(v, 0.025) for v in eachrow(ppc_sims[:, 2:end])]
    ppc_sims_summ["lower"] = [quantile(v, 0.1) for v in eachrow(ppc_sims[:, 2:end])]
    ppc_sims_summ["upper"] = [quantile(v, 0.9) for v in eachrow(ppc_sims[:, 2:end])]
    ppc_sims_summ["top"] = [quantile(v, 0.975) for v in eachrow(ppc_sims[:, 2:end])]

    return ppc_sims_summ
end

function string_to_symbol(s...)
    return Symbol(join(s))
end

function plot_individual_param_posteriors_comp(ind, samples, prior_samples, df)
    """
    Plots traces and kdes of the individual parameters for a given individual.
    """
    fig = Figure(size = (600, 700))
    axs = [Axis(fig[i, j]) for i in 1:4, j in 1:2]
    row = 1

    for i in 1:6
        i ∈ (2, 5) && continue
        stairs!(axs[row, 1], samples[:, 6 * (ind - 1) + i])
        density!(
            axs[row, 2],
            samples[:, 6 * (ind - 1) + i],
            color = (:blue, 0.3),
            strokecolor = :blue,
            strokewidth = 1,
            strokearound = true
        )
        if i < 6
            density!(
                axs[row, 2],
                prior_samples[:, i],
                color = (:red, 0.3),
                strokecolor = :red,
                strokewidth = 1,
                strokearound = true
            )
        end
        # stairs!(axs[row, 1], samples_at_0[:, 7 * (ind - 1) + i])
        # density!(axs[row, 2], samples_at_0[:, 7 * (ind - 1) + i])
        vlines!(axs[row, 2], df[ind, i + 1], color = "red")
        axs[row, 1].ylabel = names(df)[i + 1]
        axs[row, 2].xlabel = names(df)[i + 1]
        row += 1
    end

    return fig
end

function plot_shared_param_posteriors(samples, true_params; hyper_priors = hyper_priors)
    """
    Plots traces and kdes of the hyper-parameters and the noise parameter.
    """
    fig = Figure(size = (600, 900))
    axs = [Axis(fig[i, j]) for i in 1:6, j in 1:2]
    row = 1
    for i in 1:11
        i ∈ (2, 3, 4, 6, 8, 9, 10, 11) && continue
        param_symbol = fieldnames(SharedParams)[i]
        s = samples[:, end - 11 + i]
        # Traceplot
        stairs!(axs[row, 1], s)
        axs[row, 1].ylabel = String(param_symbol)
        # Marginal posteriors
        density!(
            axs[row, 2],
            s,
            color = (:blue, 0.3),
            strokecolor = :blue,
            strokewidth = 1,
            strokearound = true
        )
        vlines!(axs[row, 2], true_params[i], color = "red")
        lines!(axs[row, 2], hyper_priors[param_symbol], color = "red")

        min_val, max_val = extrema(s)
        xlims!(axs[row, 2], 0.995 * min_val, 1.015 * max_val)
        axs[row, 2].xlabel = string(fieldnames(SharedParams)[i])

        row += 1
    end

    return fig
end

function autocor(x, lags)
    n = length(x)
    μ = mean(x)
    ρ = zeros(length(lags))
    for (i, lag) in enumerate(lags)
        ρ[i] = sum((x[1:(n - lag)] .- μ) .* (x[(lag + 1):n] .- μ)) / sum((x .- μ) .^ 2)
    end
    return ρ
end

function acf_plot(θ; lags = 1:1000)
    ϱ = autocor(θ, lags)
    fig = Figure()
    ax = Axis(fig[1, 1])
    barplot!(ax, 1:length(ϱ), ϱ)
    return fig
end
