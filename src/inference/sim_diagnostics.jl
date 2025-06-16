include("within_host_inference.jl")
include("mcmc.jl")
include("../plotting.jl")
include("../io.jl")

##

N = 100
N_total_params =
    sum(1 - v for v in values(fixed_individual_params)) * N +
    sum(1 - v for v in values(fixed_shared_params))

samples = Array{Float64}(undef, 100000, N_total_params, 4)
for i in 1:4
    file = CSV.File(results_dir("samples_sim_$i.csv"))
    samples[:, :, i] = Matrix(DataFrame(file))
end

burnin = 10000
samples = samples[burnin:end, :, :]
samples = permutedims(samples, (1, 3, 2))
diagnostics = ess_rhat(samples)
min_ess = minimum(diagnostics.ess)
max_rhat = maximum(diagnostics.rhat)
println("Minimum ESS: ", min_ess)
println("Maximum Rhat: ", max_rhat)

##

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, dfs[1][:, end])
plot!(ax, dfs[2][:, end])
plot!(ax, dfs[3][:, end])
plot!(ax, dfs[4][:, end])
display(fig)
