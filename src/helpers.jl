"""
This file contains some simple helper functions for doing various things — like saving
covariance matrices, reading in covariance matrices, etc.
"""

using CSV, DataFrames

"""
Simple helper function to save the tuned covariance matrices.
"""
function save_Σs(Σs; path = "")
    for i in 1:N
        CSV.write(results_dir(path * "Σs_$i.csv"), DataFrame(Σs[i], :auto))
    end

    CSV.write(results_dir(path * "Σs_shared.csv"), DataFrame(Σs[end], :auto))

    return nothing
end

function read_df_to_mat(path)
    return Matrix(CSV.read(path, DataFrame))
end

"""
Simple helper function to read in the tuned covariance matrices given a population
size N.
"""
function read_Σs(N; path = "")
    Σs = Vector{Matrix{Float64}}(undef, N + 1)
    for i in 1:N
        Σs[i] = read_df_to_mat(results_dir(path * "Σs_$i.csv"))
    end

    Σs[end] = read_df_to_mat(results_dir(path * "Σs_shared.csv"))

    return Σs
end
