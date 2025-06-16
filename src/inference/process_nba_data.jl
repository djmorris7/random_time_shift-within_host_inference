"""
This script reads in the raw NBA data as provided in Zitzmann et al. (2024) and processes it to a sensible
format for our work. The data is then saved to a CSV file for later use.
"""

include("within_host_inference.jl")
include("mcmc.jl")
include("../plotting.jl")
include("../io.jl")
include("../../pkgs.jl")

##

df = CSV.read("data/zitzmann/ct_dat_refined.csv", DataFrame)
df[!, :log_vl] .= ct_to_vl.(df.CtT1)

fig = Figure()
ax = Axis(fig[1, 1])
for id in unique(df.PersonID)
    tmp = df[df.PersonID .== id, :]
    scatter!(ax, tmp.TestDateIndex, tmp.log_vl)
end
xlims!(ax, (-20, 40))
display(fig)

is_vaccinated = .!ismissing.(df.VaccineManufacturer)

df_data = DataFrame(
    "id" => df.PersonID,
    "obs_times" => df.TestDateIndex,
    "vl_noisy" => df.log_vl,
    "vax_status" => is_vaccinated
)

CSV.write("data/zitzmann/full_processed_data.csv", df_data)

##

df = CSV.read("data/nba/ct_dat_refined.csv", DataFrame)
df2 = CSV.read("data/nba/ct_dat_clean.csv", DataFrame)

rename!(df, Dict(:PersonID => :id, :TestDateIndex => :time, :CtT1 => :Ct))
rename!(df2, Dict("Person.ID" => "id", "Date.Index" => "time", "CT.Mean" => "Ct"))

df = df[:, [:id, :time, :Ct]]
df2 = df2[:, [:id, :time, :Ct]]

df.vl = ct_to_vl.(df.Ct)
df2.vl = ct_to_vl.(df2.Ct)

plot(df2[df2.id .== 439, :time], df2[df2.id .== 439, :vl])

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, df[df.id .== 1368, :time], df[df.id .== 1368, :vl])
xlims!(ax, -18, 18)
display(fig)

df_new = DataFrame()
append!(df_new, df)
append!(df_new, df2)

unique(df.id)
unique(df2.id)
unique(df_new.id)

CSV.write("data/nba/nba_data.csv", df_new)

##

(data, id_mapping) = load_all_nba_data(keep_vaccinated = true, keep_lod = true)

function remove_lod(dat::IndividualData, lod = 2.6576090679593496)
    times = dat.obs_times
    vl = dat.vl

    keep_idxs = findall(x -> x > lod, vl)
    times = times[keep_idxs]
    vl = vl[keep_idxs]

    return IndividualData(times, vl)
end

function data_within_cutoff(dat, cutoff = 14)
    times = dat.obs_times
    vl = dat.vl

    # First check that the non-lod data is within the cutoff
    keep = true
    for i in 1:length(vl)
        if vl[i] > 2.6576090679593496
            if times[i] < -cutoff || times[i] > cutoff
                keep = false
                break
            end
        end
    end

    return keep
end

keep_id_mapping = [data_within_cutoff(dat) for dat in data]

data = data[keep_id_mapping]
id_mapping = id_mapping[keep_id_mapping]

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

##

plot(data[7].obs_times, data[7].vl)
plot(data[7].obs_times, data[7].vl)

fig = Figure(size = (1000, 1000))
axs = [Axis(fig[i, j]) for i in 1:10, j in 1:10]

for (i, ax) in enumerate(axs)
    dat = data[i]
    plot!(ax, dat.obs_times, dat.vl)
end
display(fig)

function truncate_to_first_and_last_lod(data::IndividualData, lod = 2.6576090679593496)
    v = data.vl
    M = length(v)

    first_idx = nothing
    last_idx = nothing

    if v[1] > lod
        first_idx = 1
    end

    for i in 1:(M - 1)
        if v[i] ≤ lod && v[i + 1] > lod && isnothing(first_idx)
            first_idx = i  # Keep the last LOD value before first increase
        end

        if v[i] > lod && v[i + 1] ≤ lod
            last_idx = i + 1  # Update last LOD value before dropping below
        end

        if v[i] ≤ lod && v[i + 1] > lod
            last_idx = nothing  # Reset last_idx since VL went back above LOD
        end
    end

    # If all values are below LOD, return empty data
    if isnothing(first_idx) && isnothing(last_idx)
        return IndividualData([], [])
    end

    # If no transition to above LOD, keep the whole dataset
    if isnothing(first_idx)
        first_idx = 1
    end

    if isnothing(last_idx)
        last_idx = M
    end

    return IndividualData(data.obs_times[first_idx:last_idx], v[first_idx:last_idx])
end

function more_than_two_observations_above_lod(dat::IndividualData, lod = 2.6576090679593496)
    v = dat.vl
    M = length(v)

    num_above_lod = 0
    for i in 1:M
        if v[i] > lod
            num_above_lod += 1
        end
    end

    return num_above_lod > 2
end

keep_id_mapping = [more_than_two_observations_above_lod(dat) for dat in data]

data = data[keep_id_mapping]
id_mapping = id_mapping[keep_id_mapping]

##

data = [truncate_to_first_and_last_lod(dat) for dat in data]

##

length(data)

df_data = DataFrame()

findmin([length(d.obs_times) for d in data])

tmp = truncate_to_first_and_last_lod(data[24])
tmp = truncate_to_first_and_last_lod(data[24])
# plot(data[24].obs_times, data[24].vl)
fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, data[24].obs_times, data[24].vl)
plot!(ax, tmp.obs_times, tmp.vl)
display(fig)

for (id, dat) in zip(id_mapping, data)
    df_tmp = DataFrame(id = id, time = dat.obs_times, vl = dat.vl)
    append!(df_data, df_tmp)
end

df_data

CSV.write("data/nba/nba_data_clean.csv", df_data)

##

prop_missing = Dict()
durations = Dict()

LOD = 2.6576090679593496

total = 0
x = 0
# tmp_dat = nothing
for id in unique(df_data.id)
    dat = df_data[df_data.id .== id, :]

    t = floor.(dat.time[dat.vl .> LOD])
    expected_times = t[1]:t[end]

    diff_t = diff(t)
    if length(diff_t) > 0
        durations[id] = maximum(diff(t))
    end

    num_expected = length(expected_times)

    num_observed = length(unique(t))

    x += num_observed
    total += num_expected

    prop_missing[id] = 1 - num_observed / num_expected
end

fig = Figure()
ax = Axis(fig[1, 1])
for (id, prop) in prop_missing
    scatter!(ax, id, prop)
end
display(fig)

x / total

hist(collect(values(prop_missing)))

d = fit(Beta, Float64.(collect(values(prop_missing))))

fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, collect(values(prop_missing)), normalization = :pdf, bins = 50)
plot!(ax, d)
display(fig)

##

id = 1953

plot(df_data[df_data.id .== id, :time], df_data[df_data.id .== id, :vl])

maximum([v for (k, v) in durations])
