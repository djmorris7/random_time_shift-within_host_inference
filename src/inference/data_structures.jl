using Parameters, OrdinaryDiffEq, Flux

@with_kw struct IndividualData
    """
    Struct for storing the data for individuals. This can then
    """
    obs_times::Vector{Float64}
    vl::Vector{Float64}
end

@with_kw mutable struct Params
    """
    Struct for storing the parameters for the within-host model for a given individual.
    """
    z_R₀::Float64
    R₀::Float64
    k::Float64
    z_δ::Float64
    δ::Float64
    z_πv::Float64
    πv::Float64
    c::Float64
    infection_time::Float64
    # Currently store the support of the infection time here but this is a little hacky
    infection_time_range::Vector{Float64}
end

@with_kw mutable struct SharedParams
    """Struct for storing the shared parameters for the within-host model."""
    μ_R₀::Float64
    σ_R₀::Float64
    μ_k::Float64
    σ_k::Float64
    μ_δ::Float64
    σ_δ::Float64
    μ_πv::Float64
    σ_πv::Float64
    μ_c::Float64
    σ_c::Float64
    κ::Float64
end

@with_kw mutable struct ModelInternals
    """
    Struct for storing the model internals for the likelihood and other calculations.
    These pieces largely remain as instantiated but may change (for example, sol, which is
    the solution to the ODEs, will change as the parameters change).
    """
    Z0::Vector{Int}
    prob::ODEProblem
    nn::Chain{
        Tuple{
            Dense{typeof(relu),Matrix{Float64},Vector{Float64}},
            Dense{typeof(relu),Matrix{Float64},Vector{Float64}},
            Dense{typeof(identity),Matrix{Float64},Vector{Float64}},
            typeof(softplus)
        }
    }
    sol::ODESolution = solve(prob, Tsit5(); save_idxs = 4)
    S0::Int = Z0[1]
    Z0_bp::Vector{Int} = Z0[2:end]
    LOD::Float64 = 2.0
    fixed_params::Dict = fixed_params
end

@with_kw mutable struct MCMC_Probs
    """
    The struct for storing the probabilities for the MCMC algorithm.
    """
    current_individual_likelihoods::Vector{Float64}
    proposed_individual_likelihoods::Vector{Float64}
    current_individual_posteriors::Vector{Float64}
    proposed_individual_posteriors::Vector{Float64}
    current_shared_posterior::Float64
    proposed_shared_posterior::Float64
end

@with_kw mutable struct MCMC_Params
    """
    The struct for storing the parameters for the MCMC algorithm.
    """
    θ_curr::Vector{Params}
    θ_prop::Vector{Params}
    ϕ_curr::SharedParams
    ϕ_prop::SharedParams
end
