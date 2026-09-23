"""
Configuration for the original stochastic double-integrator transcription.

All modes default to uniform +1e-5 gain entries, repropagated UT–ShARK
covariances, one stochastic solve, and alpha_min_frac=0.05. Initialization
options do not constrain optimized gains. See the README for optional modes.
"""
Base.@kwdef struct Options
    execution_mode::Symbol = :cpu # :cpu, :gpu_condensed, :gpu_hybrid
    dt::Float64 = 0.25
    tf::Float64 = 5.0
    n_arcs::Int = 20
    x0_mean::Vector{Float64} = [2.0, 4.0, 3.0, 2.0]
    x0_covariance::Matrix{Float64} = diagm([0.1, 0.1, 0.02, 0.02])
    xf_mean::Vector{Float64} = [8.0, 2.0, 0.0, 0.0]
    xf_covariance::Matrix{Float64} = diagm([0.06, 0.06, 0.006, 0.006])
    a1::Vector{Float64} = [1.0, 1.0, 0.0, 0.0]
    b1::Float64 = 12.75
    a2::Vector{Float64} = [1.0, 0.1, 0.0, 0.0]
    b2::Float64 = 8.75
    path_confidence::Float64 = 0.9973
    u_max::Float64 = 2.0
    control_confidence::Float64 = 0.9973
    Q::Matrix{Float64} = 0.01 * Matrix{Float64}(I, NX, NX)
    R::Matrix{Float64} = Matrix{Float64}(I, NU, NU)
    scaling_parameter::Float64 = 0.0
    spectral_eigenvalue_smoothing::Float64 = 1e-5
    terminal_margin_floor::Float64 = 1e-7
    control_norm_epsilon::Float64 = 1e-6
    warm_start_gains::Bool = true
    # Initialization only: the stochastic objective continues to use R above.
    tvlqr_control_weight_scale::Float64 = execution_mode == :gpu_hybrid ? 1000.0 : 1.0
    # Default in every execution mode: every entry at every stage is +1e-5.
    # Set nothing to select TVLQR; warm_start_gains=false retains the zero seed.
    initial_gain_uniform::Union{Nothing,Float64} = warm_start_gains ? 1e-5 : nothing
    # Optionally rescale the TVLQR seed so its largest gain entry has this magnitude.
    # Covariances are propagated again with the resulting gains.
    initial_gain_max_abs::Union{Nothing,Float64} = nothing
    # SNOPT's major/minor settings do not map one-to-one to an interior-point solver.
    max_iter::Int = 10000
    max_consecutive_restoration::Int = 2
    tol::Float64 = 1e-8
    feasibility_tol::Float64 = 1e-9
    initial_barrier::Float64 = 0.01
    alpha_min_frac::Float64 = 0.05
    # Explicit relaxation for Lifted-KKT; final residuals always use original bounds.
    condensed_relaxation::Float64 = 1e-9
    condensed_continuation::Bool = false
    hybrid_gamma::Float64 = 1e7
    print_level::Int = 1
    monte_carlo_samples::Int = 8192
    monte_carlo_seed::Int = 42
    run_monte_carlo::Bool = true
    make_plots::Bool = true
    output_dir::String = joinpath(PROJECT_ROOT, "output", String(execution_mode))
end

function validate_options(o::Options)
    o.execution_mode in (:cpu, :gpu_condensed, :gpu_hybrid) ||
        throw(ArgumentError("execution_mode must be :cpu, :gpu_condensed, or :gpu_hybrid"))
    o.n_arcs >= 2 && o.dt > 0 && isfinite(o.dt) || throw(ArgumentError("invalid time grid"))
    isapprox(o.tf, o.dt * o.n_arcs; atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("tf must equal dt * n_arcs"))
    for v in (o.x0_mean, o.xf_mean, o.a1, o.a2)
        length(v) == NX && all(isfinite, v) || throw(ArgumentError("state vectors must have 4 finite entries"))
    end
    for P in (o.x0_covariance, o.xf_covariance)
        size(P) == (NX, NX) && issymmetric(P) && isposdef(P) ||
            throw(ArgumentError("initial and target covariances must be symmetric positive definite 4x4 matrices"))
    end
    # The reference terminal constraint scales by diagonal target variances.
    isdiag(o.xf_covariance) || throw(ArgumentError("the Python terminal formulation requires a diagonal target covariance"))
    for (W, n) in ((o.Q, NX), (o.R, NU))
        size(W) == (n, n) && issymmetric(W) && all(isfinite, W) &&
            eigmin(Symmetric(W)) >= 0 || throw(ArgumentError("invalid cost matrix"))
    end
    0 < o.path_confidence < 1 && 0 < o.control_confidence < 1 ||
        throw(ArgumentError("confidence levels must be in (0, 1)"))
    all(x -> isfinite(x) && x > 0, (o.u_max, o.control_norm_epsilon,
        o.spectral_eigenvalue_smoothing, o.terminal_margin_floor, o.tol,
        o.feasibility_tol, o.condensed_relaxation, o.initial_barrier, o.hybrid_gamma,
        o.tvlqr_control_weight_scale)) || throw(ArgumentError("invalid positive option"))
    NAUG + o.scaling_parameter > 0 || throw(ArgumentError("UT scale must be positive"))
    o.spectral_eigenvalue_smoothing <= 1e-4 ||
        throw(ArgumentError("spectral_eigenvalue_smoothing must not exceed the requested limit of 1e-4"))
    o.max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
    (o.initial_gain_max_abs === nothing ||
        (isfinite(o.initial_gain_max_abs) && o.initial_gain_max_abs >= 0)) ||
        throw(ArgumentError("initial_gain_max_abs must be nothing or finite and nonnegative"))
    (o.initial_gain_uniform === nothing || isfinite(o.initial_gain_uniform)) ||
        throw(ArgumentError("initial_gain_uniform must be nothing or finite"))
    (o.initial_gain_uniform === nothing || o.initial_gain_max_abs === nothing) ||
        throw(ArgumentError("uniform initialization and gain rescaling are mutually exclusive"))
    o.max_consecutive_restoration in 0:2 ||
        throw(ArgumentError("max_consecutive_restoration must be 0, 1, or 2"))
    isfinite(o.alpha_min_frac) && 0 <= o.alpha_min_frac <= 1 ||
        throw(ArgumentError("alpha_min_frac must be between 0 and 1"))
    if o.run_monte_carlo
        o.monte_carlo_samples >= 2 && ispow2(o.monte_carlo_samples) ||
            throw(ArgumentError("monte_carlo_samples must be a power of two and at least 2"))
    end
    return o
end
