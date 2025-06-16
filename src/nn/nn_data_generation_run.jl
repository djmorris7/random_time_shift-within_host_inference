include("../../pkgs.jl")
include("../io.jl")
include("../plotting.jl")

include("nn_data_generation.jl")
include("../tcl/tcl_simulation.jl")

## FLAGS

make_all_dirs()

GENERATE_DATA = true

Random.seed!(2023)
n_samples = 100_000

##

if GENERATE_DATA
    generate_data(n_samples)
end

## Visualise the data to make sure it looks okay

data = read_data()

input_par_labels = [L"R_0", L"\delta", L"\pi_v"]
output_par_labels = [L"a", L"p", L"d"]

label_idxs = 1

fig = Figure()
axs = [Axis(fig[i, j]) for i in 1:2, j in 1:3]
for i in 1:2
    for j in 1:3
        ax = axs[i, j]
        if i == 1
            hist!(ax, data["input"][:, j])
            ax.xlabel = input_par_labels[j]
        else
            hist!(ax, data["output"][:, j])
            ax.xlabel = output_par_labels[j]
        end
    end
end
display(fig)
