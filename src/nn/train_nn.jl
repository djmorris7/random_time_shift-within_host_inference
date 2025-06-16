include("../../pkgs.jl")
include("../plotting.jl")
include("../io.jl")

include("../tcl/tcl_simulation.jl")
include("nn.jl")
include("../tcl/timeshift.jl")
include("../tcl/default.jl")

##

Random.seed!(2023)

## FLAGS for training

TRAIN = true

##

data = read_data()
input = Float32.(data["input"]')
output = Float32.(data["output"]')

test_train_prop = 0.9
n_input_size, n_examples = size(input)
n_train = Int(floor(n_examples * test_train_prop))

train_X = input[:, 1:n_train]
train_Y = output[:, 1:n_train]
test_X = input[:, (n_train + 1):end]
test_Y = output[:, (n_train + 1):end]

train_loader = Flux.DataLoader((train_X, train_Y), batchsize = 128, shuffle = true)
model = Flux.Chain(Dense(3 => 64, relu), Dense(64 => 64, relu), Dense(64 => 3), softplus)

lr0 = 1e-3
lr1 = 1e-4
schedule = ParameterSchedulers.Exp(start = lr0, decay = 0.995)
opt = Flux.setup(Flux.Adam(lr0), model)

n_epochs = 1000
losses = Dict("train" => zeros(n_epochs), "test" => zeros(n_epochs))

best_test_loss = Inf
patience = 30
epoch_since_improved = 0
final_epoch = 1
best_model = deepcopy(model)

"""
The main training loop. Not put in a function to allow for easy access to the variables.
"""

if TRAIN
    for (eta, epoch) in zip(schedule, 1:n_epochs)
        loss = 0.0

        Flux.adjust!(opt, eta)

        for (x, y) in train_loader
            # Compute the loss and the gradients:
            l, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                Flux.mse(y_hat, y)
            end
            # Update the model parameters (and the Adam momenta):
            Flux.update!(opt, model, grads[1])
            # Accumulate the mean loss, just for logging:
            loss += l / length(train_loader)
        end

        # Report on train and test loss
        train_loss = Flux.mse(model(train_X), train_Y)
        test_loss = Flux.mse(model(test_X), test_Y)
        losses["train"][epoch] = train_loss
        losses["test"][epoch] = test_loss

        if mod(epoch, 10) == 1
            @info "After epoch = $epoch" loss train_loss test_loss
        end

        if (test_loss < best_test_loss) && (epoch_since_improved <= patience)
            best_model = deepcopy(model)
            epoch_since_improved = 0
            best_test_loss = test_loss
        else
            epoch_since_improved += 1
        end

        if epoch_since_improved > patience
            # Truncate the losses to the last epoch
            losses["train"] = losses["train"][1:epoch]
            losses["test"] = losses["test"][1:epoch]
            break
        end
    end

    model_state = Flux.state(best_model)
    jldsave("data/tcl_timeshift_nn.jld2"; model_state)
end

##

# Load the saved model for some basic analysis
model_state = JLD2.load("data/tcl_timeshift_nn.jld2", "model_state")
model = Flux.Chain(Dense(3 => 64, relu), Dense(64 => 64, relu), Dense(64 => 3), softplus)
Flux.loadmodel!(model, model_state)

##

errors = [Flux.mse(model(train_X[:, i]), train_Y[:, i]) for i in 1:n_train]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"\textrm{Example} \, i", ylabel = L"\textrm{MSE}")
lines!(ax, errors)
display(fig)

max_error, max_error_ind = findmax(errors)

train_Y[:, max_error_ind]
model(train_X[:, max_error_ind])

## Check the model for the maximal error parameters

test_pars = deepcopy(train_X[:, max_error_ind])
test_pars = [8.64475, 1.266, 3.020]

predicted_pars = model(test_pars)

sim_settings = load_default_pars()

insert!(test_pars, 2, sim_settings["pars"][2])
insert!(test_pars, 5, sim_settings["pars"][5])

##

R₀, k, δ, πv, c = test_pars
β_bp = get_bp_β(R₀, k, δ, πv, c)

# Calculate omega matrix and artefacts from that.
Ω = [
    -k k 0
    0 -δ πv
    β_bp 0 -c
]

λ, u_norm, v_norm = calculate_BP_contributions(Ω)

# Calc expected value of W
μ_w = dot([1, 0, 0], u_norm)

# Get the first 5 moments
num_moments = 5
lifetimes = (k, δ + πv, c + β_bp)

αs = Dict(1 => Dict([1, 2] => k))
βs = Dict(2 => Dict([2, 2, 3] => πv), 3 => Dict([3, 1, 3] => β_bp))

moments = calculate_moments_generic(Ω, αs, βs, lifetimes)

q = calculate_extinction_prob(test_pars)

# Using the moments, calculate the parameters of the GG distribution and sample
# pars_m3, _ = minimise_loss(moments, q)
pars_m3 = minimise_loss(moments, q)

# get the number of loss functions to construct
num_init_conds = size(moments, 2)

pars = [zeros(Float64, 3) for _ in 1:num_init_conds]
# Wide bounds on the parameters
lower_bd = [0.0, 0.0, 0.0]
upper_bd = [50.0, 50.0, 50.0]
# Initial guess is uninformative and reflects a boring distribution
x0 = [1.0, 1.0, 1.0]

# Wide bounds on the parameters
lower_bd = 1e-4 * ones(3)
upper_bd = 50 * ones(3)
# Initial guess is uninformative and reflects a boring distribution
x0 = [1.0, 1.0, 1.0]

inner_optimizer = BFGS()
i = 1

function loss_func(pars, moments, q; num_moments_loss = 5)
    a, d, p = pars
    loss = 0.0
    for i in 1:num_moments_loss
        y1 = moments[i] / (1 - q)
        y2 = moments_gengamma(i, a, d, p)
        η = round(y1; sigdigits = 1)
        loss += ((y2 - y1))^2 / η
    end
    return loss
end
l = x -> loss_func(x, moments[:, i], q[i])
sol = Optim.optimize(l, lower_bd, upper_bd, x0)
sol = Optim.optimize(l, lower_bd, upper_bd, x0; autodiff = :forward)

##

μ_w, λ, pars_m3, q = get_gg_pars(test_pars, sim_settings["Z0_bp"])
# samples = τ_dist_from_w(pars_m3, sim_settings["Z0_bp"], q, μ_w, λ)
# samples = τ_dist_from_w(pars_m3, sim_settings["Z0_bp"], q, μ_w, λ)

τ_range = -7:0.01:5
# τ_vals = [τ_prior(τ_i, pars_m3[1], μ_w, λ) for τ_i in τ_range]
τ_vals = [τ_prior(τ_i, sol.minimizer, μ_w, λ) for τ_i in τ_range]
# samples = τ_dist_from_w(pars_m3, sim_settings["Z0_bp"], q, μ_w, λ)

τ_range = -7:0.01:5
pred_τ_vals = [τ_prior(τ_i, predicted_pars, μ_w, λ) for τ_i in τ_range]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"\tau", ylabel = L"\textrm{PDF}")
scatter!(ax, τ_range, τ_vals, color = :red)
lines!(ax, τ_range, pred_τ_vals, color = :green)
# hist!(ax, samples, normalization = :pdf, bins = 50)
display(fig)
