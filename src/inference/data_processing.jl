"""
This file contains all functions dedicated to processing data for the within-host model. This includes
loading and cleaning data for both the simulated and NBA datasets. There are some supporting functions
for processes like the conversion of CT values to viral loads.
"""

using CSV, DataFrames

include("data_structures.jl")
include("../io.jl")

function ct_to_vl(c; intercept = 40.93733, slope = -3.60971)
    """
    Applies the mapping CT -> VL which is ripped straight from Kissler paper.
    """
    return (c - intercept) / slope * log10(10) + log10(250)
end

function load_sim_data()
    """
    This function loads the data for the within-host model. It returns the
    experiment data, the true infection times, the viral load data, and the
    latest observation time.
    """
    df_params = CSV.read(data_dir("sims/parameters.csv"), DataFrame)
    df = CSV.read(data_dir("sims/data.csv"), DataFrame)

    # Now collect the data into a simple format
    vl = Vector{Vector{Float64}}()
    vl_noisy = Vector{Vector{Float64}}()
    obs_times = Vector{Vector{Float64}}()

    for id in unique(df.ID)
        tmp = df[df.ID .== id, :]
        push!(obs_times, tmp.t)
        push!(vl, tmp.log_vl)
        push!(vl_noisy, tmp.noisy_log_vl)
    end

    N = length(vl)
    true_infection_times = df_params.infection_time
    data = Vector{IndividualData}()

    latest_obs_time = maximum([maximum(x) for x in obs_times])

    for i in 1:N
        push!(data, IndividualData(obs_times = obs_times[i], vl = vl_noisy[i]))
    end

    return data, true_infection_times, vl, latest_obs_time
end

function load_zitzmann_data()
    """
    This function loads the data for the within-host model. It returns the
    experiment data, the true infection times, the viral load data, and the
    latest observation time.
    """
    df = CSV.read(data_dir("zitzmann/processed_data.csv"), DataFrame)

    # Now collect the data into a simple format
    vl_noisy = Vector{Vector{Float64}}()
    obs_times = Vector{Vector{Float64}}()

    for id in unique(df.id)
        tmp = df[df.id .== id, :]
        push!(obs_times, tmp.obs_times)
        push!(vl_noisy, tmp.vl_noisy)
    end

    N = length(vl_noisy)
    data = Vector{IndividualData}()

    for i in 1:N
        push!(data, IndividualData(obs_times[i], vl_noisy[i]))
    end

    return data
end

function load_all_nba_data(; lod = 2.6576090679593496, keep_vaccinated = false, keep_lod = false)
    """
    This function loads the data for the within-host model. It returns the
    experiment data, the true infection times, the viral load data, and the
    latest observation time.
    """
    df = CSV.read(data_dir("nba/nba_data.csv"), DataFrame)

    # Now collect the data into a simple format
    vl_noisy = Vector{Vector{Float64}}()
    obs_times = Vector{Vector{Float64}}()
    # is_sparse = Vector{Bool}()
    id_mapping = Vector{Int}()

    for id in unique(df.id)
        # vaccinated = any(df[df.id .== id, :].vax_status)
        # if !keep_vaccinated && vaccinated
        #     continue
        # end
        tmp = df[df.id .== id, :]
        if !keep_lod
            keep_inds = tmp.vl .> lod
        else
            keep_inds = trues(length(tmp.vl))
        end

        # keep_inds = tmp.vl_noisy .> lod
        obs_times_tmp = tmp.time[keep_inds]
        vl_noisy_tmp = tmp.vl[keep_inds]
        push!(obs_times, obs_times_tmp)
        push!(vl_noisy, vl_noisy_tmp)
        # push!(is_sparse, sum(tmp.obs_times .> 0) <= 2)
        push!(id_mapping, id)
    end

    N = length(vl_noisy)
    data = Vector{IndividualData}()

    for i in 1:N
        push!(data, IndividualData(obs_times[i], vl_noisy[i]))
    end

    return data, id_mapping
end

function get_vaccination_status()
    df = CSV.read(data_dir("zitzmann/full_processed_data.csv"), DataFrame)
    vax_status = zeros(Bool, length(unique(df.id)))

    for (i, id) in enumerate(unique(df.id))
        tmp = df[df.id .== id, :]
        vax_status[i] = any(tmp.vax_status)
    end

    return vax_status
end

function load_nba_data(lod = 2.6576090679593496)
    # 2.6576090679593496
    df = CSV.read(data_dir("nba/processed_data.csv"), DataFrame)

    # Now collect the data into a simple format
    vl_noisy = Vector{Vector{Float64}}()
    obs_times = Vector{Vector{Float64}}()

    for id in unique(df.id)
        tmp = df[df.id .== id, :]
        # keep only above LOD data
        keep_inds = tmp.vl_noisy .> lod
        obs_times_tmp = tmp.obs_times[keep_inds]
        vl_noisy_tmp = tmp.vl_noisy[keep_inds]
        push!(obs_times, obs_times_tmp)
        push!(vl_noisy, vl_noisy_tmp)
    end

    N = length(vl_noisy)
    data = Vector{IndividualData}()

    for i in 1:N
        push!(data, IndividualData(obs_times[i], vl_noisy[i]))
    end

    return data
end

function get_cleaned_data(filepath)
    """
    Read in the data from a CSV file `filepath` and return it in the correct format for the inference:
    a vector of `IndividualData` objects and a vector of the unique IDs.
    This function will check that the data is in the correct format and will return an error if the
    correct headings are missing.
    """

    df_data = CSV.read(filepath, DataFrame)

    col_names = names(df_data)

    # Check that data has the correct format
    if !("id" in col_names && "time" in col_names && "vl" in col_names)
        error("DataFrame should have three variables: id, time, vl.")
    end

    data = Vector{IndividualData}()
    ids = unique(df_data.id)

    for id in ids
        df_id = filter(row -> row.id == id, df_data)
        obs_times = df_id.time
        vl = df_id.vl
        push!(data, IndividualData(obs_times, vl))
    end

    return (data, ids)
end

function drop_lod_data(data, id_mapping; lod = 2.6576090679593496)
    """
    Drops the data below the LOD.
    """
    data_tmp = Vector{IndividualData}()
    id_mapping_tmp = Vector{Int}()

    for (i, dat) in enumerate(data)
        keep_inds = dat.vl .> lod
        obs_times_tmp = dat.obs_times[keep_inds]
        vl_tmp = dat.vl[keep_inds]

        push!(data_tmp, IndividualData(obs_times_tmp, vl_tmp))
        push!(id_mapping_tmp, id_mapping[i])
    end

    return data_tmp, id_mapping_tmp
end

function drop_lod_data(data; lod = 2.6576090679593496)
    """
    Drops the data below the LOD.
    """
    data_tmp = Vector{IndividualData}()

    for (i, dat) in enumerate(data)
        keep_inds = dat.vl .> lod
        obs_times_tmp = dat.obs_times[keep_inds]
        vl_tmp = dat.vl[keep_inds]

        push!(data_tmp, IndividualData(obs_times_tmp, vl_tmp))
    end

    return data_tmp
end

function truncate_to_first_and_last_lod(data::IndividualData, lod = 2.6576090679593496)
    first_idx = -Inf
    last_idx = Inf

    v = data.vl
    M = length(v) - 1

    above_lod_idxs = findall(x -> x > lod, v)

    for i in 1:M
        # Check for the first time the viral load goes above the LOD and record the previous index
        if v[i] <= lod && v[i + 1] > lod && i < above_lod_idxs[1]
            if isinf(first_idx)
                first_idx = i
            end
        end

        if (v[i] > lod && v[i + 1] <= lod && i + 1 > above_lod_idxs[end])
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

function trim_long_tails(data::IndividualData; lod = 2.6576090679593496)
    times = data.obs_times
    vl = data.vl

    for i in 1:(length(vl) - 2)
        if vl[i] == vl[i + 1] == vl[i + 2] == lod
            return IndividualData(times[1:i], vl[1:i])
        end
    end

    return data
end

function truncate_data(data, id_mapping; truncation_time_a = -14.0, truncation_time_b = 14.0)
    data_tmp = Vector{IndividualData}()
    id_mapping_tmp = Vector{Int}()

    for (i, dat) in enumerate(data)

        # Only keep individuals with data within the truncation window
        if any(dat.obs_times .< truncation_time_a) || any(dat.obs_times .> truncation_time_b)
            continue
        end
        obs_times_tmp = dat.obs_times
        vl_tmp = dat.vl

        push!(data_tmp, IndividualData(obs_times_tmp, vl_tmp))
        push!(id_mapping_tmp, id_mapping[i])
    end

    return data_tmp, id_mapping_tmp
end

function truncate_data(data; truncation_time_a = -21.0, truncation_time_b = 21.0)
    data_tmp = Vector{IndividualData}()

    for (i, dat) in enumerate(data)
        obs_times_tmp = dat.obs_times[(dat.obs_times .>= truncation_time_a) .& (dat.obs_times .<= truncation_time_b)]
        vl_tmp = dat.vl[(dat.obs_times .>= truncation_time_a) .& (dat.obs_times .<= truncation_time_b)]

        push!(data_tmp, IndividualData(obs_times_tmp, vl_tmp))
    end

    return data_tmp
end

function init_infection_times(data; earliest_time = -5)
    """
    Create an initial guess for the infection times based on the data or a fixed time.
    """

    N = length(data)
    t0 = zeros(N)

    for (i, dat) in enumerate(data)
        # obs_peak_timing = dat.obs_times[findmax(dat.vl)[2]]
        # earliest_timing = obs_peak_timing - 20
        # latest_timing = obs_peak_timing
        earliest_non_LOD = dat.obs_times[findfirst(dat.vl .> LOD)]
        t0[i] = min(earliest_time, earliest_non_LOD)
    end

    return t0
end
