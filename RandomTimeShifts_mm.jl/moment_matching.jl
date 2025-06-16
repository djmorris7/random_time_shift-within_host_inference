"""
    moments_gengamma(n, a, d, p)

Calculates the analytical moments for a generalised Gamma distribution with
scale parameter a, and two shape parameters d, p.

# Arguments
    - n: the moment number
    - a: scale parameter
    - d: shape parameter 1
    - p: shape parameter 2

# Outputs
    - The moment
"""
function moments_gengamma(n, a, d, p)
    # if (gamma((d + n) / p) == 0) || !isfinite(gamma(d / p)) || (gamma(d / p) == 0)
    # if isfinite(gamma(d / p))
    if a > 0 && d > 0 && p > 0
        return exp(n * log(a) + (loggamma((d + n) / p) - loggamma(d / p)))
    else
        return Inf
    end
end

"""
    loss_func(x, moments, q; num_moments_loss = 5)

Calculates the squared loss function (normalised) to be optimised.

# Arguments
    - pars: the parameters (a, d, p) of the GG
    - moments: the vector of moments
    - q: the extinction probability
    - num_moments_loss: (default = 5) the number of moments

# Outputs
    - loss: the value of the loss function for a particular moment
"""
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

"""
    sample_generalized_gamma(pars)

Sample a generalised gamma random variable with parameters (a, d, p) using
the CDF method.

# Arguments
    - pars: parameters in form (a, d, p)

# Outputs
    - x: a sample from GG(a, d, p)
"""
function sample_generalized_gamma(pars)
    a, d, p = pars
    u = rand()
    # Use the inverse of the incomplete Gamma function
    y = gamma_inc_inv(d / p, u, 1 - u)
    # Apply transformation to obtain Generalized Gamma random variables
    x = a * y^(1 / p)
    return x
end

"""
    sample_W(n, pars, q1, Z0; no_extinction = true)

Sample realisations of W given pars, extinction probabilities.

# Arguments
    - n: the number of samples to draw
    - pars: list of pars (a, d, p) for each of the Wi
    - q1: vector of extinction probabilities for each Wi
    - Z0: the initial conditions for the branching process
    - no_extinction: (default = true) whether to condition on non-extinction

# Outputs
    - w: vector of samples of w
"""
function sample_W(n, pars, q1, Z0; no_extinction = true)
    w = zeros(Float64, n)
    i = 1
    while i <= n
        for j in eachindex(Z0)
            Z0[j] == 0 && continue
            for _ in 1:Z0[j]
                # run a Bernoulli trial to see whether the jth process fades out
                if rand(Bernoulli(1 - q1[j]))
                    w[i] += sample_generalized_gamma(pars[j])
                end
            end
        end
        if (no_extinction && (w[i] > 0)) || !no_extinction
            i += 1
        end
    end
    return w
end

"""
    sample_W(pars, q1, Z0)

Sample a single realisation of W given pars, extinction probabilities.

# Arguments
    - pars: list of pars (a, d, p) for each of the Wi
    - q1: vector of extinction probabilities for each Wi
    - Z0: the initial conditions for the branching process

# Outputs
    - w: samples of w
"""
function sample_W(pars, q1, Z0)
    w = 0.0

    while iszero(w)
        for j in eachindex(Z0)
            Z0[j] == 0 && continue
            for _ in 1:Z0[j]
                # run a Bernoulli trial to see whether the jth process fades out
                if rand(Bernoulli(1 - q1[j]))
                    w += sample_generalized_gamma(pars[j])
                end
            end
        end
    end
    return w
end

"""
    sample_W(pars)

Sample a single realisation of W given pars, extinction probabilities.

# Arguments
    - pars: list of pars (a, d, p) for each of the Wi

# Outputs
    - w: samples of w
"""
function sample_W(pars)
    w = sample_generalized_gamma(pars)
    return w
end

"""
    minimise_loss(moments, q1; num_moments_loss = 5, iterations = 10^5)

Minimises sum of moments - (analytical moments).

# Arguments
    - moments: an array of shape (5, number types) with the moments estimated using the methods
               from section 3.3 of the paper
    - q1: vector of extinction probabilities starting with an individual of type i
    - num_moments_loss: (default = 5) number of moments to use in the expansion. Setting a
                        default here lets us calculate more moments for other methods.
    - iterations: (default = 10^5) max number of iterations to run in the optimisation.

# Outputs
    - pars: an array of parameters with length number of types corresponding to each Wi
"""
function minimise_loss(moments, q1)
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

    for i in eachindex(pars)
        l = x -> loss_func(x, moments[:, i], q1[i])
        sol = Optim.optimize(l, lower_bd, upper_bd, x0; autodiff = :forward)
        pars[i] = sol.minimizer
    end

    return pars
end

"""
    log_loss_func(pars, moments, q; num_moments_loss = 5)

Calculates the squared loss function (normalised) to be optimised.

# Arguments
    - pars: the parameters (a, d, p) of the GG
    - moments: the vector of moments
    - q: the extinction probability
    - num_moments_loss: (default = 5) the number of moments

# Outputs
    - loss: the value of the loss function for a particular moment
"""
function log_loss_func(pars, moments, q; num_moments_loss = 5)
    x, y, z = pars

    a = exp(x)
    d = exp(y)
    p = exp(z)

    loss = 0.0

    for i in 1:num_moments_loss
        μ_n = moments[i] / (1 - q)
        pred = moments_gengamma(i, a, d, p)
        loss += ((pred - μ_n) / μ_n)^2
    end

    return log(loss)
end

function loss_func(pars, moments, q; num_moments_loss = 5)
    x, y, z = pars

    loss = 0.0

    for i in 1:num_moments_loss
        μ_n = moments[i] / (1 - q)
        pred = moments_gengamma(i, x, y, z)
        loss += ((pred - μ_n) / μ_n)^2
    end

    return log(loss)
end

"""
    minimise_log_loss(moments, q1; num_moments_loss = 5, iterations = 10^5)

Minimises sum of moments - (analytical moments). We optimise in log-space to preserve pars > 0.

# Arguments
    - moments: an array of shape (5, number types) with the moments estimated using the methods
               from section 3.3 of the paper
    - q1: vector of extinction probabilities starting with an individual of type i
    - num_moments_loss: (default = 5) number of moments to use in the expansion. Setting a
                        default here lets us calculate more moments for other methods.
    - iterations: (default = 10^5) max number of iterations to run in the optimisation.

# Outputs
    - pars: an array of parameters with length number of types corresponding to each Wi
"""
function minimise_log_loss(moments, q)
    # Initial guess reflects an exponential distribution p = d = 1 with rate 1
    x0 = ones(3)
    l = x -> log_loss_func(x, moments[:, 1], q[1])
    # Tolerance of 1e-14 needs to be really high to get the right answer
    # sol = Optim.optimize(l, x0, NelderMead(), Optim.Options(g_tol = 1e-12); autodiff = :forward)
    sol = Optim.optimize(l, x0)

    x = exp.(sol.minimizer)

    return x
end

"""
    minimise_log_loss(moments, q1; num_moments_loss = 5, iterations = 10^5)

Minimises sum of moments - (analytical moments). We optimise in log-space to preserve pars > 0.

# Arguments
    - moments: an array of shape (5, number types) with the moments estimated using the methods
               from section 3.3 of the paper
    - q1: vector of extinction probabilities starting with an individual of type i
    - num_moments_loss: (default = 5) number of moments to use in the expansion. Setting a
                        default here lets us calculate more moments for other methods.
    - iterations: (default = 10^5) max number of iterations to run in the optimisation.

# Outputs
    - pars: an array of parameters with length number of types corresponding to each Wi
"""
function minimise_loss(moments, q)
    # Initial guess reflects an exponential distribution p = d = 1 with rate 1
    x0 = exp(1) * ones(3)
    l = x -> loss_func(x, moments[:, 1], q[1])
    # Tolerance of 1e-14 needs to be really high to get the right answer
    # sol = Optim.optimize(l, x0, NelderMead(), Optim.Options(g_tol = 1e-12); autodiff = :forward)
    sol = Optim.optimize(l, x0)

    x = sol.minimizer

    return x
end

"""
    generate_partitions(k::Int, m::Int)

Generates all partitions of k into m parts.

# Arguments
    - k: the number to partition
    - m: the number of parts

# Outputs
    - partitions: a vector of vectors of integers
"""
function generate_partitions(k::Int, m::Int)
    partitions = Vector{Vector{Int}}()
    generate_partitions_recursive(partitions, [], k, m)
    return partitions
end

"""
    generate_partitions_recursive(
        partitions, current_partition, remaining_sum, remaining_integers
    )

Generates all partitions of k into m parts.

# Arguments
    - partitions: the vector of partitions
    - current_partition: the current partition
    - remaining_sum: the remaining sum
    - remaining_integers: the remaining integers

# Outputs
    - partitions: a vector of vectors of integers
"""
function generate_partitions_recursive(
    partitions, current_partition, remaining_sum, remaining_integers
)
    if remaining_integers == 0
        if remaining_sum == 0
            push!(partitions, copy(current_partition))
        end
        return nothing
    end
    for x in 0:remaining_sum
        new_partition = copy(current_partition)
        push!(new_partition, x)
        generate_partitions_recursive(
            partitions, new_partition, remaining_sum - x, remaining_integers - 1
        )
    end
end

"""
    compute_W_moments(moments; num_moments = 5)

Computes the moments of W using the moments of the individual types.

# Arguments
    - moments: the moments of the individual types
    - num_moments: (default = 5) the number of moments to compute

# Outputs
    - W_moments: the moments of W
"""
function compute_W_moments(moments, Z0_bp, q_star; num_moments = 5)
    W_moments = zeros(Float64, num_moments)
    Z0_bp_cumsum = cumsum(Z0_bp)
    m = sum(Z0_bp)
    for k in 1:num_moments
        A = generate_partitions(k, m)
        for l in A
            C = multinomial(l...)
            internal_term = 1.0
            j = 1
            for i in 1:m
                if l[i] > 0
                    internal_term *= moments[l[i], j]
                else
                    internal_term *= 1.0
                end
                # Move to the next type if we have exhausted the current type
                if i >= Z0_bp_cumsum[j]
                    j += 1
                end
            end
            W_moments[k] += C * internal_term
        end
    end

    W_moments = W_moments / (1 - q_star)

    return W_moments
end

function loss_func(pars, W_moments)
    a, d, p = pars

    loss = 0.0

    for (i, W_i) in enumerate(W_moments)
        y2 = moments_gengamma(i, a, d, p)
        η = round(W_i; sigdigits = 1)
        loss += ((y2 - W_i) / W_i)^2
    end

    return loss
end

function minimise_loss(W_moments)
    pars = zeros(Float64, 3)

    # Initial guess is uninformative and reflects a boring distribution but is not (a, d, p) = (1, 1, 1)
    # which can break gradients
    x0 = ones(Float64, 3)
    lower_bd = 1e-4 * ones(3)
    upper_bd = 50 * ones(3)

    l = x -> loss_func(x, W_moments)
    sol = Optim.optimize(
        l, lower_bd, upper_bd, x0, Optim.Options(g_tol = 1e-12); autodiff = :forward
    )
    pars = sol.minimizer

    return pars
end
