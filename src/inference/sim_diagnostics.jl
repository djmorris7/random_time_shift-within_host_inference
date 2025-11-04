include("within_host_inference.jl")
include("mcmc.jl")
include("../plotting.jl")
include("../io.jl")

##

N = 100
N_total_params =
    sum(1 - v for v in values(fixed_individual_params)) * N +
    sum(1 - v for v in values(fixed_shared_params))

dataset_id = 20

dfs = [
    DataFrame(CSV.File(results_dir("sim_samples/dataset_$dataset_id/samples_$i.csv"))) for i in 1:3
]
for df in dfs
    df.σ_R₀ = exp.(df.σ_R₀)
    df.σ_δ = exp.(df.σ_δ)
    df.σ_πv = exp.(df.σ_πv)
end

samples = Array{Float64}(undef, size(dfs[1], 1), size(dfs[1], 2), 3)
for i in 1:3
    samples[:, :, i] = Matrix(dfs[i])
end

# burnin = 20000
# thin = 4
# samples = samples[begin:thin:end, :, :]
samples = permutedims(samples, (1, 3, 2))
diagnostics = ess_rhat(samples)
min_ess = minimum(diagnostics.ess)
max_rhat = maximum(diagnostics.rhat)
findall(diagnostics.rhat .> 1.01)
println("Minimum ESS: ", min_ess)
println("Maximum Rhat: ", max_rhat)

##

fig = Figure()
ax = Axis(fig[1, 1])
density!(ax, dfs[1][burnin:end, 404])
density!(ax, dfs[2][burnin:end, 404])
density!(ax, dfs[3][burnin:end, 404])
display(fig)

fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, dfs[1][burnin:end, :σ_πv], bins = 30, normalization = :pdf)
hist!(ax, dfs[2][burnin:end, :σ_πv], bins = 30, normalization = :pdf)
hist!(ax, dfs[3][burnin:end, :σ_πv], bins = 30, normalization = :pdf)
display(fig)

##

lines(dfs[2][burnin:end, :σ_πv])
lines(dfs[3][burnin:end, :σ_πv])
# hist(dfs[1][burnin:end, :σ_πv])

##

hexbin(dfs[3][burnin:end, :σ_πv], dfs[3][burnin:end, :z_πv_1], bins = 40)
