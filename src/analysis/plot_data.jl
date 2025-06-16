include("../inference/within_host_inference.jl")
include("../plotting.jl")
include("results.jl")

##

(nba_data, nba_ids) = get_cleaned_data("data/nba/nba_data_clean.csv")
(sim_data, sim_ids) = get_cleaned_data("data/sims/sim_data_clean.csv")

df_true_pars = CSV.read(data_dir("sims/parameters.csv"), DataFrame)
df_true_hyper_pars = CSV.read(data_dir("sims/hyper_parameters.csv"), DataFrame)

true_infection_times = df_true_pars[!, :infection_time]

fig_loc = "figures/simulation/"
if isdir(fig_loc) == false
    mkdir(fig_loc)
end

##

lod = 2.6576090679593496

size_inches = (6.5, 3.5)
size_pt = (size_inches[1] * 72, size_inches[2] * 72)

fig = Figure(size = size_pt, fontsize = 11, dpi = 300)
axs = [
    Axis(fig[1, 1], title = L"\text{(A) Simulated data}"; ax_kwargs...),
    Axis(fig[1, 2], title = L"\text{(B) NBA data}"; ax_kwargs...)
]
for dat in nba_data
    scatter!(axs[2], dat.obs_times, dat.vl, color = (:black, 0.25), markersize = 5)
end
for (t_inf, dat) in zip(true_infection_times, sim_data)
    scatter!(axs[1], dat.obs_times, dat.vl, color = (:black, 0.25), markersize = 5)
end
xlims!(axs[1], (-14, 14))
xlims!(axs[2], (-14, 14))
ylims!(axs[1], (0.95 * lod, 10.5))
ylims!(axs[2], (0.95 * lod, 10.5))
axs[1].ylabel = L"\log_{10}(\text{VL})"
axs[2].ylabel = L"\log_{10}(\text{VL})"

hlines!(axs[1], [lod], color = :red, linestyle = :dash, label = "LOD")
hlines!(axs[2], [lod], color = :red, linestyle = :dash, label = "LOD")

Label(fig[2, 1:2], "Time (days) post peak VL", valign = :bottom)

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

display(fig)

save(joinpath(fig_loc, "data.pdf"), fig, pt_per_unit = 1)
