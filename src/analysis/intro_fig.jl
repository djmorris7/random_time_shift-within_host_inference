"""
Author: Dylan Morris
Date: 12 Jul 2023 at 11:08:23 am
Description:
This script generates some data from the stochastic TCL model and adds some observation noise in.
We generate the VL trajectories for all individuals assuming the same initial conditions.
"""

# Include these once or things break
include("../../pkgs.jl")
include("../io.jl")
include("../plotting.jl")

##

# Includes relative to this files location.
include("../tcl/tcl_simulation.jl")
include("../tcl/default.jl")
include("../inference/within_host_inference.jl")

##

# Set seed for reproducibility.
Random.seed!(2021)

# Testing the gillespie simulator
S0 = Int(8e7)

sim_settings = load_default_pars()
pars = sim_settings["pars"]
Z0 = sim_settings["Z0"]

R₀, k, δ, πv, c = pars

Z0 = [S0 - 1, 1, 0, 0]
LOD = 2.6576090679593496
ct_LOD = 40.0

sims_dict = Dict()

n_samples = 15
i = 1

while i <= n_samples
    t, Z, extinct = tcl_gillespie(pars, Z0; tf = 14.0, Δt = 0.001, save_at = 0.05)
    if !extinct
        sims_dict[i] = (t, Z)
        i += 1
    end
end

u0 = Z0
tspan = (0, 14)
prob = ODEProblem(tcl_deterministic!, u0, tspan, pars)
sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, save_idxs = 4)

det_traj = [sol.t max.(0.0, log10.(sol.u))]

##

# Now determine peak times and keep 2 before and 2 after the deterministic peak time.
peak_times = zeros(n_samples)
for (k, v) in sims_dict
    t, Z = v
    peak_times[k] = t[argmax([Z[n][4] for n in eachindex(Z)])]
end

det_peak_time = det_traj[argmax(det_traj[:, 2]), 1]
before_det_peak_idx = findall(peak_times .< det_peak_time)[1:3]
after_det_peak_idx = findall(peak_times .> det_peak_time)[1:3]

##

bp_pars = [R₀, k, δ, πv, c]
Z0_bp = [1, 0, 0]

nn = load_nn()
θ = Params(0.0, R₀, k, 0.0, δ, 0.0, πv, c, 0.0, [0.0, 14.0])
m = ModelInternals(Z0 = Z0, prob = prob, nn = nn)
μ_w, λ, pars_m3 = get_gg_pars_nn(θ, m)

x = -5:0.025:1
y = [τ_prior(xi, pars_m3, μ_w, λ) for xi in x]

##

size_inches = (6.5, 3)
size_pt = size_inches .* 72
fig = Figure(size = size_pt, fontsize = 10, dpi = 300)
# ax = Axis(fig[1, 1], xlabel = "time (days)", ylabel = L"\log_{10}(V)")
axs = [
    Axis(
        fig[1, 1];
        ax_kwargs...,
        yaxisposition = :left,
        xlabel = "time (days)",
        ylabel = L"\log_{10}(V)"
    ),
    Axis(
        fig[1, 2];
        ax_kwargs...,
        yaxisposition = :right,
        # xlabel = "time (days)",
        ylabel = L"\log_{10}(V)"
    ),
    Axis(
        fig[1, 2];
        yticklabelcolor = colors[2],
        ylabelcolor = colors[2],
        ax_kwargs...,
        xlabel = "adjusted time (days)",
        ylabel = "density"
    )
]

t_det_above = det_traj[findfirst(det_traj[:, 2] .>= LOD), 1]
for (k, v) in sims_dict
    # if k in before_det_peak_idx || k in after_det_peak_idx
    t, Z = v

    V = [Z[n][4] for n in eachindex(Z)]
    lines!(axs[1], t, log10.(V), color = (:grey, 0.3))

    idxs = findfirst(log10.(V) .>= LOD)
    t = t[(idxs - 1):end] .- t_det_above
    V = V[(idxs - 1):end]
    lines!(axs[2], t, log10.(V), color = (:grey, 0.3))
    # lines!(axs[1], t, [max(0.0, log10(Z[n][4])) for n in eachindex(Z)], color = (:grey, 0.3))
end
lines!(axs[1], det_traj[:, 1], det_traj[:, 2], color = :black, linestyle = :dash)
hlines!(axs[1], LOD, linestyle = :dash, color = :red)

linkxaxes!(axs[2], axs[3])
lines!(axs[2], det_traj[:, 1] .- t_det_above, det_traj[:, 2], color = :black, linestyle = :dash)
# Need to reverse the time-shift x-values since this is now the shift to the initial conditions.
lines!(axs[3], -x, y, color = colors[2])
ylims!(axs[2], LOD, 7.5)
ylims!(axs[3], 0, 1.1)
xlims!(axs[3], -1.5, 7)

axs[1].title = L"\text{(A)}"
axs[3].title = L"\text{(B)}"

fig

save("figures/time_shift_behaviour.pdf", fig, pt_per_unit = 1.0)
