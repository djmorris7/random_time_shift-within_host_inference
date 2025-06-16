"""
This file is used to explore the NBA data and process it into a format that can be used for inference.
We save a partially cleaned version of the file.
"""

include("../../pkgs.jl")
include("../plotting.jl")
include("../io.jl")

##

df = CSV.read("data/nba/indiv_data.csv", DataFrame)

unique_ids = unique(df.id)

id = 3
lines(df[df.id .== unique_ids[id], "t"], df[df.id .== unique_ids[id], "y"])

# Create shorthand types
const VVF64 = Vector{Vector{Float64}}
const VVInt = Vector{Vector{Int}}

ids_to_keep = [737, 755, 942, 1273, 1740, 2349, 2463, 3485, 3491]
ids_to_keep = unique_ids
# N = length(unique(df.id))
N = length(ids_to_keep)

# Now collect the data into a simple format
ct_noisy = VVF64(undef, N)
vl_noisy = VVF64(undef, N)
obs_times = VVInt(undef, N)
true_infection_times = zeros(Float64, N)

function ct_to_vl(c; intercept = 40.93733, slope = -3.60971)
    """
    Applies the mapping CT -> VL which is ripped straight from Kissler paper.
    """
    return (c - intercept) / slope * log10(10) + log10(250)
end

# for (i, id) in enumerate(unique(df.id))
for (i, id) in enumerate(ids_to_keep)
    tmp = df[df.id .== id, :]
    true_infection_times[i] = -5
    obs_times[i] = tmp.t
    ct_noisy[i] = tmp.y
    vl_noisy[i] = ct_to_vl.(tmp.y)
end

ids = unique(df.id)

fig = Figure()
ax = Axis(fig[1, 1])
for (o, y) in zip(obs_times, vl_noisy)
    plot!(ax, o, y, alpha = 0.5)
    ax.xlabel = "time since peak CT/VL"
    ax.ylabel = L"VL ($\log_{10}$)"
end
display(fig)

##

id_arr = deepcopy(obs_times)
for (v, id) in zip(id_arr, unique_ids)
    v .= id
end

id_arr = vcat(id_arr...)
obs_times = vcat(obs_times...)
vl_noisy = vcat(vl_noisy...)
ct_noisy = vcat(ct_noisy...)

##

df_data = DataFrame()

##

df_data = DataFrame(
    "id" => id_arr,
    "obs_times" => obs_times,
    "vl_noisy" => vl_noisy,
    "ct_noisy" => ct_noisy
    # "true_infection_times" => true_infection_times
)

CSV.write("data/nba/processed_data.csv", df_data)

##

df = CSV.read("data/zitzmann/zitzmann.csv", DataFrame)

fig = Figure()
ax = [Axis(fig[i, j]) for i in 1:5, j in 1:5]
for (i, id) in enumerate(unique(df.ID))
    tmp = df[df.ID .== id, :]
    scatter!(ax[i], tmp[:, 2], tmp[:, 3])
end
display(fig)

df_data = DataFrame("id" => df.ID, "obs_times" => df[:, 2], "vl_noisy" => df[:, 3])

CSV.write("data/zitzmann/processed_data.csv", df_data)

##

df2 = CSV.read("data/nba/ct_dat_clean.csv", DataFrame)

rename!(df2, Dict("Person.ID" => "id"))

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, df2[df2[!, "Person.ID"] .== 27, :time], df2[df2.id .== 27, :vl])
plot!(
    ax,
    df_novax[df_novax.PersonID .== 27, :TestDateIndex],
    df_novax[df_novax.PersonID .== 27, :log_vl],
    marker = :cross
)
xlims!(-14, 14)
display(fig)

df = CSV.read("data/nba/ct_dat_refined.csv", DataFrame)
df_novax = df[:, [:PersonID, :TestDateIndex, :CtT1, :RowID]]
df_novax[!, :log_vl] .= ct_to_vl.(df_novax.CtT1)

df_combined = join(df_novax, df2, on = :PersonID, kind = :inner)

fig = Figure()
ax = Axis(fig[1, 1])
for id in unique(df_novax.PersonID)
    tmp = df_novax[df_novax.PersonID .== id, :]
    scatter!(ax, tmp.TestDateIndex, tmp.log_vl)
end
xlims!(ax, (-20, 40))
display(fig)

is_vaccinated = .!ismissing.(df.VaccineManufacturer)

df_data = DataFrame(
    "id" => df_novax.PersonID,
    "obs_times" => df_novax.TestDateIndex,
    "vl_noisy" => df_novax.log_vl,
    "vax_status" => is_vaccinated
)

CSV.write("data/zitzmann/full_processed_data.csv", df_data)

##

df = CSV.read("data/nba/ct_dat_clean.csv", DataFrame)

df = df[:, ["Person.ID", "Date.Index", "CT.Mean"]]
df[!, :log_vl] .= ct_to_vl.(df[:, "CT.Mean"])

fig = Figure()
ax = Axis(fig[1, 1])
for id in unique(df[:, "Person.ID"])
    tmp = df[df[:, "Person.ID"] .== id, :]
    scatter!(ax, tmp[:, "Date.Index"], tmp.log_vl)
end
xlims!(ax, (-20, 40))
display(fig)

is_vaccinated = .!ismissing.(df.VaccineManufacturer)

df_data = DataFrame(
    "id" => df_novax.PersonID,
    "obs_times" => df_novax.TestDateIndex,
    "vl_noisy" => df_novax.log_vl,
    "vax_status" => is_vaccinated
)

# CSV.write("data/zitzmann/full_processed_data.csv", df_data)

##

df_tmp = df_data[df_data.id .== 87, :]
plot(df_tmp.obs_times, df_tmp.vl_noisy)

##

ids_unique = unique(df_novax.PersonID)
is_vaccinated_unique = zeros(Int, length(ids_unique))
for (i, id) in enumerate(ids_unique)
    is_vaccinated_unique[i] = !ismissing(df[df.PersonID .== id, :VaccineManufacturer][1])
end

vaccinated_ids = ids_unique[is_vaccinated_unique .== 1]
unvaccinated_ids = ids_unique[is_vaccinated_unique .== 0]

fig = Figure()
axs = [Axis(fig[i, 1]) for i in 1:2]
for (i, id) in enumerate(vaccinated_ids)
    tmp = df_novax[df_novax.PersonID .== id, :]
    scatter!(axs[1], tmp.TestDateIndex, tmp.log_vl)
end
for (i, id) in enumerate(unvaccinated_ids)
    tmp = df_novax[df_novax.PersonID .== id, :]
    scatter!(axs[2], tmp.TestDateIndex, tmp.log_vl)
end

display(fig)

##

max_i = 0
curr_max = 0

LOD = df_novax.log_vl[1]

for i in unique(df_novax.PersonID)
    v_tmp = df_novax[df_novax.PersonID .== i, "log_vl"]
    t_tmp = df_novax[df_novax.PersonID .== i, "TestDateIndex"]
    t_tmp = t_tmp[v_tmp .> LOD]

    len = t_tmp[end]

    if len > curr_max
        curr_max = len
        max_i = i
    end
end

tmp = df_novax[df_novax.PersonID .== max_i, "log_vl"]
tmp[tmp .> LOD]
t_tmp = df_novax[df_novax.PersonID .== max_i, "TestDateIndex"]
t_tmp[tmp .> LOD]

plot(
    df_novax[df_novax.PersonID .== max_i, "TestDateIndex"],
    df_novax[df_novax.PersonID .== max_i, "log_vl"]
)

## Estimate proportion of days missing for each individual

p = zeros(length(unique(df_novax.PersonID)))
x_total = 0
n_total = 0

for (i, id) in enumerate(unique(df_novax.PersonID))
    if i == 17
        println(id)
    end
    v_tmp = df_novax[df_novax.PersonID .== id, "log_vl"]
    t_tmp = df_novax[df_novax.PersonID .== id, "TestDateIndex"]
    t_tmp = t_tmp[v_tmp .> LOD]
    t_tmp_diff = diff(t_tmp)
    p[i] = sum(t_tmp_diff .> 1.3) / length(t_tmp)
    n_total += length(t_tmp)
    x_total += sum(t_tmp_diff .> 1.3)
end

mean(p)

sum(p .> 0) / length(p)

## Estimate how many individuals have missing data in the growth phase (i.e. minimum 2 observations needed)

fig = Figure(size = (1600, 1200))
axs = [Axis(fig[i, j]) for i in 1:13, j in 1:13]
for (i, id) in enumerate(unique(df_novax.PersonID))
    i > length(axs) && break
    tmp = df_novax[df_novax.PersonID .== id, :]
    tmp = tmp[(tmp.TestDateIndex .>= -14) .&& (tmp.TestDateIndex .<= 14), :]
    # tmp = tmp[tmp.log_vl .> LOD, :]
    scatter!(axs[i], tmp.TestDateIndex, tmp.log_vl)
end
display(fig)

##

fig = Figure()
ax = Axis(fig[1, 1])
id = 219
tmp = df_novax[df_novax.PersonID .== id, :]
tmp = tmp[(tmp.TestDateIndex .>= -14) .&& (tmp.TestDateIndex .<= 14), :]
# tmp = tmp[tmp.log_vl .> LOD, :]
scatter!(ax, tmp.TestDateIndex[tmp.log_vl .> LOD], tmp.log_vl[tmp.log_vl .> LOD])
scatter!(ax, tmp.TestDateIndex[tmp.log_vl .<= LOD], tmp.log_vl[tmp.log_vl .<= LOD])

display(fig)

save("figures/zitzmann_data.pdf", fig, pt_per_unit = 1)
