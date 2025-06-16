function check_exist_mkdir(dir)
    if !isdir(dir)
        mkdir(dir)
    end

    return nothing
end

function data_dir(args...)
    data_dir = "data/"
    dir = joinpath(args...)
    final_dir = joinpath(data_dir, dir)

    return final_dir
end

function results_dir(args...)
    results_dir = "results/"
    dir = joinpath(args...)
    final_dir = joinpath(results_dir, dir)

    return final_dir
end

function figs_dir(args...)
    figures_dir = "figures/"
    dir = joinpath(args...)
    final_dir = joinpath(figures_dir, dir)

    return final_dir
end

function make_all_dirs()
    check_exist_mkdir("data")
    check_exist_mkdir("results")
    check_exist_mkdir("figures")

    return nothing
end
