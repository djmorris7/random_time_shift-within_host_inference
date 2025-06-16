"""
This script reads in the simulated data and processes it to a sensible
format for our work. The data is then saved to a CSV file for later use.
"""

include("within_host_inference.jl")
include("mcmc.jl")
include("../plotting.jl")
include("../io.jl")
include("data_processing.jl")

##

(data, true_infection_times, vl, obs_t) = load_sim_data()

function remove_missing_data(data::IndividualData)
    times_tmp = data.obs_times
    vl_tmp = data.vl

    idxs = findall(x -> !iszero(x), vl_tmp)
    times = times_tmp[idxs]
    vl = vl_tmp[idxs]

    return IndividualData(times, vl)
end

data = [remove_missing_data(dat) for dat in data]

function cutoff_pre_and_post_data(dat, cutoff = 14)
    times = dat.obs_times
    vl = dat.vl

    keep_idxs = (times .>= -cutoff) .& (times .<= cutoff)
    times = times[keep_idxs]
    vl = vl[keep_idxs]

    new_dat = IndividualData(times, vl)

    return new_dat
end

data = [cutoff_pre_and_post_data(dat) for dat in data]

function truncate_to_first_and_last_lod(data::IndividualData, lod = 2.6576090679593496)
    first_idx = -Inf
    last_idx = Inf

    v = data.vl
    M = length(v) - 1

    for i in 1:M
        if v[i] <= lod && v[i + 1] > lod
            if isinf(first_idx)
                first_idx = i
            end
        end

        if v[i] > lod && v[i + 1] <= lod
            if isinf(last_idx)
                last_idx = i + 1
            end
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

    return IndividualData(times_tmp[first_idx:last_idx], vl_tmp[first_idx:last_idx])
end

data = [truncate_to_first_and_last_lod(dat) for dat in data]

df_data = DataFrame()

for (id, dat) in enumerate(data)
    df_tmp = DataFrame(id = id, time = dat.obs_times, vl = dat.vl)
    append!(df_data, df_tmp)
end

df_data

CSV.write("data/sims/sim_data_clean.csv", df_data)

##

fig = Figure(size = (800, 800))
ax = Axis(fig[1, 1])
for (i, dat) in enumerate(data)
    plot!(ax, dat.obs_times, dat.vl, label = "Individual $i")
end
display(fig)

fig = Figure(size = (800, 800))
axs = [Axis(fig[i, j]) for i in 1:10, j in 1:10]
for (i, ax) in enumerate(axs)
    plot!(ax, data[i].obs_times, data[i].vl, label = "Individual $i")
end
display(fig)

plot(data[8].obs_times, data[8].vl, label = "Individual 8")
