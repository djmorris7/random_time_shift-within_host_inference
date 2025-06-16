include("../inference_centered/within_host_inference.jl")
include("results.jl")

##

Random.seed!(2023)

n_samples = 10000

##

hyper_prior_samples = Dict()

for (k, v) in hyper_priors
    hyper_prior_samples[k] = rand(v, n_samples)
end

hyper_prior_df = DataFrame(hyper_prior_samples)

##

param_sigmas = CSV.read(data_dir("param_sigma.csv"), DataFrame)
σ_β, σ_δ, σ_πv = param_sigmas[1, ["β", "δ", "πv"]]

prior_samples = Dict()

prior_samples["β"] = [rand(priors[:β](m, σ_β)) for m in hyper_prior_df.μ_β]
prior_samples["δ"] = [
    rand(priors[:δ](m, s)) for (m, s) in zip(hyper_prior_df.μ_δ, hyper_prior_df.σ_δ)
]
prior_samples["πv"] = [rand(priors[:πv](m, σ_πv)) for m in hyper_prior_df.μ_πv]

fig = Figure()
axs = [Axis(fig[i, j]) for i in 1:1, j in 1:3]
hist!(axs[1], prior_samples["β"], bins = 20)
hist!(axs[2], prior_samples["δ"], bins = 20)
hist!(axs[3], prior_samples["πv"], bins = 20)
display(fig)

##

fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, hyper_priors[:μ_δ])
hist!(ax, df_samples.μ_δ, bins = 50, normalization = :pdf)
display(fig)

minimum(df_samples.δ_122)

z = rand(Normal(0, 1), length(df_samples.μ_δ))
x = df_samples.μ_δ .+ df_samples.σ_δ .* z

fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, x, bins = 50, normalization = :pdf)
hist!(ax, prior_samples["δ"], bins = 50, normalization = :pdf)
display(fig)

1 / maximum(df_samples.μ_δ)

##

df = hcat(hyper_prior_df, prior_df)
df.β = df.μ_β .+ df.σ_β .* df.z_β
df.δ = df.μ_δ .+ df.σ_δ .* df.z_δ
df.πv = df.μ_πv .+ df.σ_πv .* df.z_πv

##

fig = Figure()
axs = [Axis(fig[i, j]) for i in 1:2, j in 1:2]
hist!(axs[1, 1], df.β, bins = 20)
axs[1, 1].xlabel = "β"
hist!(axs[1, 2], df.δ, bins = 20)
axs[1, 2].xlabel = "δ"
hist!(axs[2, 1], df.πv, bins = 20)
axs[2, 1].xlabel = "πv"
hist!(axs[2, 2], df.infection_time, bins = 20)
axs[2, 2].xlabel = "infection_time"
display(fig)

##

πv_new = df_samples.μ_πv .+ df_samples.σ_πv .* rand(Normal(0, 1), length(df_samples.μ_πv))

fig = Figure()
ax = Axis(fig[1, 1])
density!(ax, df.πv, color = (:blue, 0.5))
density!(ax, πv_new, color = (:orange, 0.5))
density!(ax, df_samples.πv_1, color = (:pink, 0.5))
density!(ax, df_samples.πv_2, color = (:grey, 0.5))
display(fig)

##
