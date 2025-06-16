function gen_gamma_pdf(w, gg_pars)
    a, d, p = gg_pars
    pdf_val = p / a^d * w^(d - 1) * exp(-(w / a)^p) / gamma(d / p)
    return pdf_val
end

function τ_prior(τ, gg_pars, μ_w, λ)
    w = μ_w * exp(λ * τ)
    dτ = μ_w * λ * exp(λ * τ)
    pdf_val = dτ * gen_gamma_pdf(w, gg_pars)
    return pdf_val
end

function τ_dist_from_w(pars_m3, Z0_bp, q, μ_w, λ)
    w = sample_W(10000, pars_m3, q, Z0_bp)
    τ = λ^-1 * log.(w ./ μ_w)
    return τ
end

function sample_generalized_gamma(pars)
    a, d, p = pars
    u = rand()
    # Use the inverse of the incomplete Gamma function
    y = gamma_inc_inv(d / p, u, 1 - u)
    # Apply transformation to obtain Generalized Gamma random variables
    x = a * y^(1 / p)
    return x
end
