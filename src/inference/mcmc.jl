"""
This file contains all the functions for running the Metropolis-Hastings within Gibbs sampler.
The evaluation of the various parts of the model are handled here but a lot of information is taken
from `within_host_inference.jl`.
"""

include("../../pkgs.jl")
include("data_structures.jl")
include("within_host_inference.jl")

"""
Transform the lower-bounded parameters to the log-space.
"""
function lower_bound_transform(x, a)
    return log(x - a)
end

"""
Inverse transform the lower-bounded parameters back.
"""
function inverse_lower_bound_transform(y, a)
    return exp(y) + a
end

"""
The derivative of the lower bound transformation which yields the prior density of the
logged parameters.
"""
function abs_derivative_lower_bound_transform(p_x, y, a)
    x = inverse_lower_bound_transform(y, a)
    p_y = f(x) * exp(y)
    return p_y
end

function extract_sampled_params(
    θ::Params; fixed_params = fixed_params, ignore_params = :infection_time_range
)
    """
    Extracts the sampled parameters from the Params struct.
    """
    sampled_params = Float64[]
    sampled_params_symbols = Symbol[]

    for x in fieldnames(Params)
        if (x == ignore_params) || fixed_params[x]
            continue
        end
        push!(sampled_params, getfield(θ, x))
        push!(sampled_params_symbols, x)
    end

    return sampled_params, sampled_params_symbols
end

function extract_sampled_shared_params(ϕ::SharedParams; fixed_params = fixed_params)
    """
    Extracts the sampled parameters from the Params struct.
    """
    sampled_params = Float64[]
    sampled_params_symbols = Symbol[]

    for x in fieldnames(SharedParams)
        if fixed_params[x]
            continue
        end
        push!(sampled_params, getfield(ϕ, x))
        push!(sampled_params_symbols, x)
    end

    return sampled_params, sampled_params_symbols
end

function is_in_θ_support!(θ; fixed_params = fixed_params)
    """
    Check that the parameters are in the support of the model. Also updates the
    non-centered parameters.
    """
    # Do a simple check to ensure the parameters are positive
    check = (θ.R₀ > 0.0) && (θ.δ > 0.25) && (θ.πv > 0.1)
    return check
end

function is_in_ϕ_support(ϕ; fixed_params = fixed_params)
    """
    Check that the shared parameters are in the support of the model.
    """
    for x in fieldnames(SharedParams)
        fixed_params[x] && continue
        if getfield(ϕ, x) <= 0
            return false
        end
    end
    return true
end

function update_ncps!(θ, ϕ)
    θ.R₀ = θ.z_R₀ * ϕ.σ_R₀ + ϕ.μ_R₀
    θ.δ = θ.z_δ * ϕ.σ_δ + ϕ.μ_δ
    θ.πv = θ.z_πv * ϕ.σ_πv + ϕ.μ_πv

    return nothing
end

function θ_proposal!(θ_new, θ_old, ϕ_old, Σ; fixed_params = fixed_params)
    """
    Samples the individual parameters given the previous values.
    """
    θ_new_v, θ_symbols = extract_sampled_params(θ_old; fixed_params = fixed_params)
    # Proposed parameters are sampled from a multivariate normal distribution
    θ_new_v .+= rand(MvNormal(Σ))

    # Update the Params struct
    for (θ_i, s) in zip(θ_new_v, θ_symbols)
        setfield!(θ_new, s, θ_i)
    end

    # Now update the non-centered parameters
    update_ncps!(θ_new, ϕ_old)

    is_in_support = is_in_θ_support!(θ_new)

    return is_in_support
end

function ϕ_proposal!(ϕ_new, ϕ_old, θ_old, Σ; fixed_params = fixed_params)
    """
    Samples the shared parameters given the previous values.
    """

    ϕ_new_v, ϕ_symbols = extract_sampled_shared_params(ϕ_old; fixed_params = fixed_params)
    ϕ_new_v .+= rand(MvNormal(Σ))

    adj_dens = 0.0

    # Update the SharedParams struct
    for (ϕ_i, s) in zip(ϕ_new_v, ϕ_symbols)
        setfield!(ϕ_new, s, ϕ_i)
    end

    # Now update the non-centered parameters
    for i in eachindex(θ_old)
        update_ncps!(θ_old[i], ϕ_new)
    end

    is_in_support_ϕ = is_in_ϕ_support(ϕ_new)
    is_in_support_θ = all(is_in_θ_support!(θ_i) for θ_i in θ_old)
    is_in_support = is_in_support_ϕ && is_in_support_θ

    return is_in_support, adj_dens
end

function accept_reject(π_new, π_old; adj_dens = 0.0)
    """
    Acceptance-rejection step for the Metropolis-Hastings algorithm. This function
    also includes the adjustment for sampling non-symmetric distributions or when a
    Jacobian is needed. The adjustment defaults to 0.0.
    """
    if isinf(π_new)
        return false
    elseif isinf(π_old) && !isinf(π_new)
        return true
    else
        α = min(0.0, π_new - π_old + adj_dens)
        return α > log(rand())
    end
end

function create_sampling_df(N, samples)
    """
    Creates a DataFrame for storing the samples from the MCMC algorithm.
    """
    df_names = Symbol[]

    for i in 1:N, name in ["z_R₀", "z_δ", "z_πv", "infection_time"]
        push!(df_names, Symbol("$(name)_$i"))
    end

    for name in ["μ_R₀", "σ_R₀", "μ_δ", "σ_δ", "μ_πv", "σ_πv", "κ"]
        push!(df_names, Symbol("$(name)"))
    end

    df = DataFrame(samples, df_names)

    return df
end

function fill_ϕ!(ϕ_to, ϕ_from)
    """
    Fills the ϕ_prop struct with the values from ϕ_old.
    """
    for x in fieldnames(SharedParams)
        setfield!(ϕ_to, x, getfield(ϕ_from, x))
    end

    return nothing
end

function fill_θ!(θ_to, θ_from)
    """
    Fills the θ_prop struct with the values from θ_old.
    """
    for x in fieldnames(Params)
        setfield!(θ_to, x, getfield(θ_from, x))
    end

    return nothing
end

function params_to_vec(θ::Params)
    params_vec = Float64[]
    for name in fieldnames(Params)
        if name == :infection_time_range
            continue
        end
        push!(params_vec, getfield(θ, name))
    end

    return params_vec
end

function vec_to_params(vec::Vector{Float64}, infection_time_range)
    return Params(vec..., infection_time_range)
end

function shared_params_to_vec(θ::SharedParams)
    params_vec = Float64[]
    for name in fieldnames(SharedParams)
        push!(params_vec, getfield(θ, name))
    end

    return params_vec
end

function vec_to_shared_params(vec::Vector{Float64})
    return SharedParams(vec...)
end

#! DEPRECATED
# function convert_samples_to_row(θ_curr, ϕ_curr)
#     return vcat([params_to_vec(θ_i) for θ_i in θ_curr]..., shared_params_to_vec(ϕ_curr))
# end

function convert_samples_to_row(θ_curr, ϕ_curr)
    row = []
    for i in eachindex(θ_curr)
        for s in ["z_R₀", "z_δ", "z_πv", "infection_time"]
            push!(row, getfield(θ_curr[i], Symbol(s)))
        end
    end
    for s in ["μ_R₀", "σ_R₀", "μ_δ", "σ_δ", "μ_πv", "σ_πv", "κ"]
        push!(row, getfield(ϕ_curr, Symbol(s)))
    end

    return row
end

function intialise_Σs(N, Σ_individuals_diag, Σ_shared_diag)
    Σ_individuals = diagm(Σ_individuals_diag)
    Σs = [deepcopy(Σ_individuals) for _ in 1:N]
    Σ_shared = diagm(Σ_shared_diag)

    return push!(Σs, Σ_shared)
end

function compute_acceptance_rates(mcmc_stats, n_current, n_total)
    """
    Computes the acceptance rates for the Metropolis-Hastings within Gibbs sampler.
    """
    accepted_individual_avg = 0.0
    for i in 1:length(mcmc_stats[:n_total_proposals_individual])
        accepted_individual_avg +=
            mcmc_stats[:n_accepted_proposals_individual][i] /
            mcmc_stats[:n_total_proposals_individual][i]
    end
    accepted_individual_avg /= length(mcmc_stats[:n_total_proposals_individual])
    accepted_percent_individual = round(accepted_individual_avg * 100, digits = 2)

    accepted_percent_shared = round(
        100 * mcmc_stats[:n_accepted_proposals_shared] / mcmc_stats[:n_total_proposals_shared];
        digits = 2
    )

    return (accepted_percent_individual, accepted_percent_shared)
end

function step_one!(mcmc_params::MCMC_Params, mcmc_probs::MCMC_Probs, data, M_threads, mcmc_stats)
    """
    The first step of the Metropolis-Hastings within Gibbs sampler. This step loops over the
    individuals and samples parameters per person given the shared parameters.
    """
    @unpack_MCMC_Params mcmc_params
    @unpack_MCMC_Probs mcmc_probs

    N = length(data)

    # 1. Sample individual parameters given current shared parameters
    Threads.@threads for i in 1:N
        # Propose parameter and check it's in support before doing more work
        is_in_support = θ_proposal!(
            θ_prop[i], θ_curr[i], ϕ_curr, Σs[i]; fixed_params = fixed_params
        )
        mcmc_stats[:n_total_proposals_individual][i] += 1

        # Compute current target density
        # TODO: Check that this is actually correctly calculated at this point...
        # current_individual_likelihoods[i] = likelihood(
        #     θ_curr[i], data[i], ϕ_curr, M_threads[Threads.threadid()]
        # )
        current_individual_posteriors[i] =
            current_individual_likelihoods[i] + individual_prior(θ_curr[i], ϕ_curr)

        if is_in_support
            # Compute density of proposed point
            proposed_individual_posteriors[i] = individual_prior(θ_prop[i], ϕ_curr)
            proposed_individual_likelihoods[i] = likelihood(
                θ_prop[i], data[i], ϕ_curr, M_threads[Threads.threadid()]
            )
            proposed_individual_posteriors[i] += proposed_individual_likelihoods[i]

            if accept_reject(proposed_individual_posteriors[i], current_individual_posteriors[i])
                fill_θ!(θ_curr[i], θ_prop[i])
                current_individual_posteriors[i] = proposed_individual_posteriors[i]
                current_individual_likelihoods[i] = proposed_individual_likelihoods[i]
                mcmc_stats[:n_accepted_proposals_individual][i] += 1
            end
        end
    end

    return nothing
end

function step_two!(mcmc_params::MCMC_Params, mcmc_probs::MCMC_Probs, data, M_threads, mcmc_stats)
    """
    The second step of the Metropolis-Hastings within Gibbs sampler. This step samples the shared
    parameters given the current individual parameters.
    """
    @unpack_MCMC_Params mcmc_params
    @unpack_MCMC_Probs mcmc_probs

    # Compute current target density for the shared parameters
    current_shared_posterior =
        shared_prior(ϕ_curr) +
        sum(individual_prior(θ_curr[i], ϕ_curr) for i in eachindex(θ_curr)) +
        sum(current_individual_likelihoods)

    # println(shared_prior(ϕ_curr))
    # println(sum(individual_prior(θ_curr[i], ϕ_curr) for i in eachindex(θ_curr)))
    # println("Sum of likelihoods: ", sum(current_individual_likelihoods))

    N = length(data)

    # Make sure we have the correct parameters at the time of sampling
    for i in 1:N
        fill_θ!(θ_prop[i], θ_curr[i])
    end

    # 2. Sample shared parameters given current individual parameters
    is_in_support, adj_dens = ϕ_proposal!(
        ϕ_prop, ϕ_curr, θ_prop, Σs[end]; fixed_params = fixed_params
    )

    mcmc_stats[:n_total_proposals_shared] += 1

    if is_in_support
        # Initialise the proposed target density to the prior of the shared parameters
        proposed_shared_posterior = shared_prior(ϕ_prop)
        # Compute the conditional posterior given current individual parameters and
        # proposed shared parameters. Don't add individual prior contributions here.
        Threads.@threads for i in 1:N
            # for i in 1:N
            proposed_individual_likelihoods[i] = likelihood(
                θ_prop[i], data[i], ϕ_prop, M_threads[Threads.threadid()]
            )
        end

        proposed_shared_posterior += sum(proposed_individual_likelihoods)
        proposed_shared_posterior += sum(
            individual_prior(θ_prop[i], ϕ_prop) for i in eachindex(θ_prop)
        )

        if accept_reject(proposed_shared_posterior, current_shared_posterior; adj_dens = adj_dens)
            # Store the updated parameters
            fill_ϕ!(ϕ_curr, ϕ_prop)
            for i in 1:N
                fill_θ!(θ_curr[i], θ_prop[i])
            end
            current_individual_likelihoods .= proposed_individual_likelihoods
            mcmc_stats[:n_accepted_proposals_shared] += 1
        end
    end

    return nothing
end

function reset_info(mcmc_stats)
    """
    Resets the information for the Metropolis-Hastings within Gibbs sampler.
    """
    mcmc_stats[:n_accepted_proposals_shared] = 0
    mcmc_stats[:n_total_proposals_shared] = 0
    mcmc_stats[:n_accepted_proposals_individual] .= 0
    mcmc_stats[:n_total_proposals_individual] .= 0
    return nothing
end

function metropolis_within_gibbs(
    θ_all,
    ϕ,
    data,
    Σs,
    n_samples;
    fixed_params = fixed_params,
    sample_hierarchical = true,
    update_interval = 10_000,
    burnin = 30_000,
    save_every = 10,
    fixed_shared_params = fixed_shared_params,
    fixed_individual_params = fixed_individual_params
)
    """
    The Metropolis-Hastings within Gibbs sampler for our hierarchical model. This function
    samples the individual parameters and the shared parameters.
    """
    #! Hard coded for now but this should be passed in or grabbed from somewhere
    n_pars_per_person = sum(1 - v for v in values(fixed_individual_params))
    n_shared_pars = sum(1 - v for v in values(fixed_shared_params))
    N = length(θ_all)

    # Setup data structures for the sampling components
    # The current and proposed values of the parameters
    θ_curr = deepcopy(θ_all)
    θ_prop = deepcopy(θ_all)
    # The current and proposed values of the shared parameters
    ϕ_curr = deepcopy(ϕ)
    ϕ_prop = deepcopy(ϕ)

    # The current and proposed values of the log likelihoods and posteriros
    current_individual_likelihoods = [
        likelihood(θ_curr[i], data[i], ϕ_curr, M_threads[1]) for i in 1:N
    ]
    proposed_individual_likelihoods = [-Inf for i in 1:N]
    current_individual_posteriors = [
        current_individual_likelihoods[i] + individual_prior(θ_curr[i], ϕ_curr) for i in 1:N
    ]
    proposed_individual_posteriors = [-Inf for i in 1:N]
    current_shared_posterior = sum(current_individual_likelihoods) + shared_prior(ϕ_curr)
    proposed_shared_posterior = -Inf

    # Create the data structures for the MCMC algorithm
    mcmc_probs = MCMC_Probs(
        current_individual_likelihoods,
        proposed_individual_likelihoods,
        current_individual_posteriors,
        proposed_individual_posteriors,
        current_shared_posterior,
        proposed_shared_posterior
    )
    mcmc_params = MCMC_Params(θ_curr, θ_prop, ϕ_curr, ϕ_prop)

    scaling_factors = ones(N + 1)
    Σs_unscaled = deepcopy(Σs)

    # The samples
    n_samples_save = n_samples ÷ save_every

    n_cols = N * n_pars_per_person + n_shared_pars
    samples = zeros(Float64, n_samples_save, n_cols)

    mcmc_stats = Dict(
        :n_accepted_proposals_shared => 0,
        :n_total_proposals_shared => 0,
        :n_accepted_proposals_individual => zeros(N),
        :n_total_proposals_individual => zeros(N),
        :start_time => now(),
        :curr_time => now()
    )

    n_curr = 1

    # Instantiate a progress bar that updates every 10 seconds
    p = Progress(n_samples; showspeed = true, dt = 10.0)

    for n in 1:n_samples
        # Sample the individual parameters
        step_one!(mcmc_params, mcmc_probs, data, M_threads, mcmc_stats)
        # Sample the shared parameters
        step_two!(mcmc_params, mcmc_probs, data, M_threads, mcmc_stats)

        if save_every == 1
            samples[n, :] = convert_samples_to_row(θ_curr, ϕ_curr)
        elseif n % save_every == 1
            samples[n_curr, :] = convert_samples_to_row(θ_curr, ϕ_curr)
            n_curr += 1
        end

        (accepted_percent_individual, accepted_percent_shared) = compute_acceptance_rates(
            mcmc_stats, n, n_samples
        )
        # Update prog bar status showing some key metrics like the acceptance rate for
        # the individual and shared parameters
        next!(
            p;
            showvalues = [
                ("iteration count", n),
                ("accepted percent (individual)", accepted_percent_individual),
                ("accepted percent (shared)", accepted_percent_shared)
            ]
        )
    end

    return samples
end

# --- Process MCMC ---

function extract_individual_params(df_samples, id)
    """
    Pulls out the parameters from the samples DataFrame for a given individual.
    """
    selected_cols = [
        col for col in names(df_samples) if
        occursin("_$id", col) && (col[(end - length("_$id") + 1):end] == "_$id")
    ]
    df_samples_ind = df_samples[:, selected_cols]
    return df_samples_ind
end

function tune_Σs!(Σs, df_samples, N, burnin; scalings = 0.3 * ones(N + 1))
    df_samples_burned = df_samples[burnin:end, :]
    for i in 1:N
        df_samples_ind = extract_individual_params(df_samples_burned, i)
        M = Matrix(df_samples_ind[:, ["z_β_$i", "z_δ_$i", "z_πv_$i", "infection_time_$i"]])
        Σs[i] .= scalings[i] * cov(M)
    end

    df_samples_shared = df_samples_burned[:, [:μ_β, :σ_β, :μ_δ, :σ_δ, :μ_πv, :σ_πv, :κ]]
    M = Matrix(df_samples_shared)
    Σs[end] .= scalings[end] * cov(M)

    return nothing
end

function update_start!(samples, infection_time_ranges, N, burnin)
    """
    Update the starting values for the MCMC based off the pilot run using the mean of
    the samples as the initial values.
    """

    samples_burned = samples[burnin:end, :]
    # Compute means
    mean_samples = mean(samples_burned, dims = 1)
    mean_samples_params = mean_samples[:, 1:(end - 11)]
    mean_samples_shared = mean_samples[:, (end - 10):end]

    # Reshape the means for the individual level parameters so each row is a parameter set
    mean_samples_params = reshape(mean_samples_params, (6, N))'
    θ₀ = [Params(m..., y) for (m, y) in zip(eachrow(mean_samples_params), infection_time_ranges)]
    ϕ₀ = SharedParams(mean_samples_shared...)

    return θ₀, ϕ₀
end
