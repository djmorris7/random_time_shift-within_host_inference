"""
This script reads in the simulated data and processes it to a sensible
format for our work. The data is then saved to a CSV file for later use.
"""

include("../inference/within_host_inference.jl")
include("../inference/mcmc.jl")
include("../plotting.jl")
include("../io.jl")
include("data_processing.jl")

##

function remove_missing_data(data::IndividualData)
    times_tmp = data.obs_times
    vl_tmp = data.vl

    idxs = findall(x -> !iszero(x), vl_tmp)
    times = times_tmp[idxs]
    vl = vl_tmp[idxs]

    return IndividualData(times, vl)
end

function cutoff_pre_and_post_data(dat, cutoff = 14)
    times = dat.obs_times
    vl = dat.vl

    keep_idxs = (times .>= -cutoff) .& (times .<= cutoff)
    times = times[keep_idxs]
    vl = vl[keep_idxs]

    new_dat = IndividualData(times, vl)

    return new_dat
end

function truncate_to_first_and_last_lod(data::IndividualData, lod = 2.6576090679593496; buffer = 1)
    v = data.vl
    M = length(v) - 1
    first_idx = -Inf
    last_idx = Inf

    # Find first above-LOD after a below-LOD point
    for i in 1:M
        if v[i] <= lod && v[i + 1] > lod
            if isinf(first_idx)
                first_idx = i
            end
        end
    end

    # Find last above-LOD before a below-LOD point
    for i in 1:M
        if v[i] > lod && v[i + 1] <= lod
            last_idx = i + 1
        end
    end

    times_tmp = data.obs_times
    vl_tmp = data.vl

    if isinf(first_idx)
        first_idx = 1
    end

    if isinf(last_idx)
        last_idx = M + 1
    end

    # Expand window slightly to avoid clipping near-LOD peak points
    first_idx = max(1, first_idx - buffer)
    last_idx = min(M + 1, last_idx + buffer)

    # Ensure we keep all above-LOD points in this window
    above_lod_idxs = findall(>(lod), vl_tmp)
    min_peak_idx = minimum(above_lod_idxs)
    max_peak_idx = maximum(above_lod_idxs)
    first_idx = min(first_idx, min_peak_idx)
    last_idx = max(last_idx, max_peak_idx)

    return IndividualData(times_tmp[first_idx:last_idx], vl_tmp[first_idx:last_idx])
end

i = 1

params_loc = data_dir("sims/covid_parameters_$i.csv")
data_loc = data_dir("sims/covid_data_$i.csv")
# params_loc = data_dir("sims/parameters.csv")
# data_loc = data_dir("sims/data.csv")

(data, true_infection_times, obs_t) = load_sim_data(params_loc, data_loc)

function apply_lod(data, lod)
    data_new = deepcopy(data)

    for dat in data_new
        lod_check = dat.vl .<= lod
        dat.vl[lod_check] .= lod
    end

    return data_new
end

lod = 2.6576090679593496  # LOD for the simulated data
# lod = 0.0

data = apply_lod(data, lod)

data = [remove_missing_data(dat) for dat in data]

data = [cutoff_pre_and_post_data(dat) for dat in data]

data = [truncate_to_first_and_last_lod(dat, lod) for dat in data]
# data = [truncate_preserve_peak(dat) for dat in data]

df_data = DataFrame()

for (id, dat) in enumerate(data)
    df_tmp = DataFrame(id = id, time = dat.obs_times, vl = dat.vl)
    append!(df_data, df_tmp)
end

df_data

CSV.write("data/sims/covid_data_clean_$i.csv", df_data)

##

N_datasets = length(filter(f -> occursin(r"data_\d+\.csv$", f), readdir(data_dir("sims"))))

for i in 1:N_datasets
    @info "Processing dataset $i / $N_datasets"
    params_loc = data_dir("sims/covid_parameters_$i.csv")
    data_loc = data_dir("sims/covid_data_$i.csv")

    (data, true_infection_times, obs_t) = load_sim_data(params_loc, data_loc)

    data = apply_lod(data, lod)

    data = [remove_missing_data(dat) for dat in data]

    data = [cutoff_pre_and_post_data(dat) for dat in data]

    data = [truncate_to_first_and_last_lod(dat) for dat in data]

    df_data = DataFrame()

    for (id, dat) in enumerate(data)
        df_tmp = DataFrame(id = id, time = dat.obs_times, vl = dat.vl)
        append!(df_data, df_tmp)
    end

    df_data

    CSV.write("data/sims/covid_data_clean_$i.csv", df_data)
end

##

covid_data_clean = CSV.read(data_dir("sims/covid_data_sparse_clean_1.csv"), DataFrame)
# covid_data_clean = CSV.read(data_dir("sims/covid_data_clean_2.csv"), DataFrame)

N = unique(covid_data_clean.id)

fig = Figure(size = (800, 1600))
axs = [Axis(fig[i, j]) for i in 1:10, j in 1:5]
for i in eachindex(axs)
    ax = axs[i]
    t = covid_data_clean.time[covid_data_clean.id .== i]
    y = covid_data_clean.vl[covid_data_clean.id .== i]
    plot!(ax, t, y, label = "Individual $i")
end
display(fig)
