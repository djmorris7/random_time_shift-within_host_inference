include("within_host_inference.jl")
include("mcmc.jl")
include("../plotting.jl")
include("../io.jl")

##

(data, id_mapping) = get_cleaned_data("data/nba/nba_data_clean.csv")
N = length(data)

N_total_params =
    sum(1 - v for v in values(fixed_individual_params)) * N +
    sum(1 - v for v in values(fixed_shared_params))

samples = Array{Float64}(undef, 36_001, N_total_params, 3)
for i in 1:3
    file = CSV.File(results_dir("samples_nba_$i.csv"))
    samples[:, :, i] = Matrix(DataFrame(file))
end

# burnin = 10_000
# samples = samples[burnin:end, :, :]
samples = permutedims(samples, (1, 3, 2))
diagnostics = ess_rhat(samples)
min_ess = minimum(diagnostics.ess)
max_rhat = maximum(diagnostics.rhat)

print(findmax(diagnostics.rhat))

println("Minimum ESS: ", min_ess)
println("Maximum Rhat: ", max_rhat)

##

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, samples[:, :, 656])

display(fig)
