include("../../pkgs.jl")
include("../io.jl")
include("../plotting.jl")

include("nn_data_generation.jl")
include("../tcl/tcl_simulation.jl")

## FLAGS

GENERATE_DATA = false

##

Random.seed!(2023)
n_samples = 100_000

##

if GENERATE_DATA
    generate_data(n_samples)
end

generate_data(n_samples)

## Visualise the data to make sure it looks okay

data = read_data()

fig = Figure()
axs = [Axis(fig[i, j]) for i in 1:2, j in 1:3]
for i in 1:2
    for j in 1:3
        ax = axs[i, j]
        if i == 1
            hist!(ax, data["input"][:, j])
        else
            hist!(ax, data["output"][:, j])
        end
    end
end
display(fig)
