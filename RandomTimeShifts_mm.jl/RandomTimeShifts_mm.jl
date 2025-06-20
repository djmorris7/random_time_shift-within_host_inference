module RandomTimeShifts_mm

using Optim,
    Distributions,
    Random,
    Statistics,
    Combinatorics,
    SpecialFunctions,
    LinearAlgebra,
    StaticArrays,
    OrdinaryDiffEq

include("./diff_phi_functional_equations.jl")
export diff_linear,
    diff_quadratic_1D,
    diff_quadratic,
    lhs_coeffs,
    calculate_moments_ND,
    calculate_moments_1D,
    calculate_moments_generic

include("./LST_approximation.jl")
export moment_coeffs, construct_lst, calculate_BP_contributions, F_offspring_ode

include("./moment_matching.jl")
export sample_generalized_gamma, sample_W, minimise_loss, minimise_log_loss, sample_τ

end
