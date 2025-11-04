include("../inference/within_host_inference.jl")
include("../plotting.jl")
include("results.jl")

##

(nba_data, nba_ids) = get_cleaned_data("data/nba/nba_data_clean.csv")
(sim_data, sim_ids) = get_cleaned_data("data/sims/covid_data_clean_1.csv")

df_true_pars = CSV.read(data_dir("sims/covid_parameters_1.csv"), DataFrame)
df_true_hyper_pars = CSV.read(data_dir("sims/covid_hyper_parameters_1.csv"), DataFrame)

true_infection_times = df_true_pars[!, :infection_time]

fig_loc = "figures/"
if isdir(fig_loc) == false
    mkdir(fig_loc)
end

##

lod = 2.6576090679593496

size_inches = (6.5, 3.0)
size_pt = (size_inches[1] * inch, size_inches[2] * inch)

fig = Figure(size = size_pt, fontsize = fontsize, dpi = dpi)
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
axs[1].ylabel = L"\log_{10}(\text{viral load})"
axs[2].ylabel = L"\log_{10}(\text{viral load})"

hlines!(axs[1], [lod], color = :red, linestyle = :dash, label = "LOD")
hlines!(axs[2], [lod], color = :red, linestyle = :dash, label = "LOD")

Label(fig[2, 1:2], "Time (days) post peak VL", valign = :bottom)

rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

resize_to_layout!(fig)

display(fig)

save(joinpath(fig_loc, "data.png"), fig, px_per_unit = dpi / inch)
save(joinpath(fig_loc, "data.pdf"), fig)
