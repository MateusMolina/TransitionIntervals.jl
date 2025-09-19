using Pkg
# Ensure local example environment (uncomment if running standalone)
# Pkg.activate(@__DIR__)
# Pkg.instantiate()

using Distributions
using LinearAlgebra
using DecisionTree
using TransitionIntervals

include("utilities.jl")

# == 1. Generate synthetic data for an unknown latent nonlinear system == #
# Latent system (not revealed to abstraction): x_{k+1} = f(x_k) + noise
# We will learn f with a random forest regressor (separate model per dimension).

function latent_dynamics(x::Vector{Float64})
    # mildly nonlinear twist on linear dynamics
    A = [0.85 0.15; -0.10 0.9]
    y = A * x
    y[1] += 0.05 * sin(3 * x[2])
    y[2] += 0.05 * (x[1]^2)
    return y
end

noise_dist = MvNormal([0.0, 0.0], 0.005I(2))

# Data collection domain
x_lower = [-1.0, -1.0]
x_upper = [ 1.0,  1.0]

function sample_uniform_box(lower, upper)
    return lower .+ rand(length(lower)) .* (upper .- lower)
end

Ntrain = 4000
X = Matrix{Float64}(undef, 2, Ntrain)
Y = Matrix{Float64}(undef, 2, Ntrain)
for i in 1:Ntrain
    x = sample_uniform_box(x_lower, x_upper)
    y = latent_dynamics(x) + rand(noise_dist)
    X[:, i] = x
    Y[:, i] = y
end

# == 2. Train random forest regressors for each output dimension == #
# DecisionTree.jl expects features in columns? Actually, DecisionTree.regression_forest
# expects features as a matrix with each row an observation. We'll transpose.

Xtrain = permutedims(X)  # Ntrain x 2
Y1 = Y[1, :]
Y2 = Y[2, :]

# Hyperparameters kept modest for speed
# build_forest(y, X, n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, min_samples_split, min_purity_increase)
rf1 = build_forest(Y1, Xtrain, 2, 100, 0.7, 8, 5, 2, 0.0)
rf2 = build_forest(Y2, Xtrain, 2, 100, 0.7, 8, 5, 2, 0.0)

# == 3. Calibrate an empirical uniform error bound == #
# We compute residuals on the training (or a holdout) set and choose a quantile.

function predict_point(rf, x::AbstractVector)
    # DecisionTree expects row vector like features -> convert to 1xD matrix
    return apply_forest(rf, reshape(x, 1, :))[1]
end

residuals = Matrix{Float64}(undef, 2, Ntrain)
for i in 1:Ntrain
    x = X[:, i]
    yhat1 = predict_point(rf1, x)
    yhat2 = predict_point(rf2, x)
    residuals[1, i] = Y[1, i] - yhat1
    residuals[2, i] = Y[2, i] - yhat2
end

abs_res = abs.(residuals)
# pick high quantile to get (approx) 99% coverage per-dimension
q = 0.995
err_bound = map(i -> quantile(abs_res[i, :], q), 1:2)
@info "Per-dimension symmetric error bounds (approx)" err_bound

# Wrap this into a state-dependent (optional) function returning a UniformError-like distribution.
# For simplicity we use provided GaussianUniformError as a symmetric envelope proxy, scaling sigma so that ~3*sigma ≈ bound.

function uniform_error_dist(lower::Vector{Float64}, upper::Vector{Float64}; thread_idx=1)
    # ignore state; constant bound for demo. Could make it depend on (lower, upper) if we tracked heteroscedasticity.
    # Choose sigma so that 3*sigma*scale ≈ max(err_bound)
    maxb = maximum(err_bound)
    sigma = maxb / 3
    return GaussianUniformError(sigma, 1.0)
end

# == 4. Build an image map function for abstraction == #
# Given a state (box), we need to produce an over-approximated image box. We:
#  - evaluate RF at each corner of the box
#  - take min/max per dimension
#  - (Optionally) expand by learned error bound

function box_corners(lower::Vector{Float64}, upper::Vector{Float64})
    return [[l1, l2] for l1 in (lower[1], upper[1]), l2 in (lower[2], upper[2])]
end

function rf_image_map(lower::Vector{Float64}, upper::Vector{Float64}; thread_idx=1)
    corners = box_corners(lower, upper)
    preds = Matrix{Float64}(undef, 2, length(corners))
    for (i, c) in enumerate(corners)
        preds[1, i] = predict_point(rf1, c)
        preds[2, i] = predict_point(rf2, c)
    end
    img_lower = map(i -> minimum(view(preds, i, :)), 1:2)
    img_upper = map(i -> maximum(view(preds, i, :)), 1:2)
    # Expand by empirical error bound to cover residual uncertainty
    img_lower .-= err_bound
    img_upper .+= err_bound
    return (img_lower, img_upper)
end

# == 5. Discretization and abstraction construction == #
discretization = UniformDiscretization(DiscreteState(x_lower, x_upper), [0.4, 0.4])
abstraction = transition_intervals(discretization, rf_image_map, uniform_error_dist)

# == 6. Verification spec == #
# Reuse existing spec; interpret labels over state means.
spec_filename = joinpath(@__DIR__, "specs", "bdd_until2.toml")
amb_states = verify_and_plot(abstraction, spec_filename)

# == 7. Iterative refinement == #
refinement_steps = 3
for step in 1:refinement_steps
    @info "Refinement step" step num_ambiguous=length(amb_states)
    global abstraction = refine_abstraction(abstraction, rf_image_map, discretization.compact_space, amb_states, uniform_error_dist)
    global amb_states = verify_and_plot(abstraction, spec_filename)
    if isempty(amb_states)
        @info "All states classified; stopping early." step
        break
    end
end

@info "Finished random forest data-driven verification example." total_states=length(abstraction.states)
