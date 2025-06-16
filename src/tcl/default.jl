function load_default_pars()
    S0 = Int(8e7)
    E0 = 1
    I0 = 0
    V0 = 0

    Z0 = [S0 - (E0 + I0), E0, I0, V0]
    Z0_bp = Z0[2:end]
    S0 = Z0[1]

    # This is a defacto R₀ given some area we don't know
    R₀ = 8.0
    k = 4.0
    δ = 1.28
    πv = 3.07
    c = 10.0
    pars = [R₀, k, δ, πv, c]

    sim_settings = Dict("S0" => S0, "Z0" => Z0, "Z0_bp" => Z0_bp, "pars" => pars)

    return sim_settings
end
