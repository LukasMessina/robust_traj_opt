function execution_backend(o::Options)
    o.execution_mode == :cpu && return nothing
    # Load GPU packages only when requested. invokelatest in solve_problem handles
    # newly loaded extension methods without forcing CUDA initialization on CPU.
    @eval import CUDA, CUDSS, MadNLPGPU
    if o.execution_mode == :gpu_hybrid
        @eval import HybridKKT
        if !isdefined(@__MODULE__,:RegularizedHybridKKTSystem)
            Base.include(@__MODULE__,joinpath(@__DIR__,"regularized_hybrid_kkt.jl"))
        end
    end
    return Base.invokelatest(configure_cuda_backend)
end

function configure_cuda_backend()
    CUDA.functional() || error("CUDA is unavailable; use execution_mode=:cpu")
    CUDA.allowscalar(false)
    return CUDA.CUDABackend()
end

function solver_options(o::Options; nominal=false)
    # Keep the energy seed accurate when the stochastic stopping tolerance is relaxed.
    tolerance = nominal ? min(o.tol,1e-8) : o.tol
    common = (; max_iter=o.max_iter, tol=tolerance, acceptable_tol=tolerance,
        intermediate_callback=RestorationGuard(limit=o.max_consecutive_restoration),
        # Use the same line-search threshold in all execution modes.
        alpha_min_frac=o.alpha_min_frac,
        barrier=MadNLP.MonotoneUpdate{Float64}(mu_init=nominal ? 0.1 : o.initial_barrier,
            mu_min=min(1e-4,tolerance)/11),
        acceptable_iter=0, print_level=o.print_level > 0 ? MadNLP.INFO : MadNLP.ERROR,
        # The 504 original variables are unbounded. These settings only move
        # MadNLP's internal slacks into the interior, preserving the model seed.
        bound_push=1e-2, bound_fac=1e-2)
    if o.execution_mode == :cpu || nominal
        return merge(common, (; linear_solver=MadNLP.MumpsSolver,
            kkt_system=MadNLP.SparseKKTSystem, equality_treatment=MadNLP.EnforceEquality,
            bound_relax_factor=0.0))
    elseif o.execution_mode == :gpu_hybrid
        return merge(common, (; linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL, kkt_system=RegularizedHybridKKTSystem,
            inertia_correction_method=MadNLP.InertiaBased,
            equality_treatment=MadNLP.EnforceEquality,
            fixed_variable_treatment=MadNLP.MakeParameter, bound_relax_factor=0.0))
    else
        return merge(common, (; linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL, kkt_system=MadNLP.SparseCondensedKKTSystem,
            equality_treatment=MadNLP.RelaxEquality,
            fixed_variable_treatment=MadNLP.MakeParameter,
            bound_relax_factor=o.condensed_relaxation))
    end
end

function run_madnlp(model,o::Options,settings)
    # ExaModels encodes large expression graphs in its types. A setup-time
    # function barrier avoids propagating them through the entire driver.
    solver=Base.invokelatest(MadNLP.MadNLPSolver,model;settings...)
    # HybridKKT exposes gamma on the constructed KKT system, rather than as a
    # MadNLP keyword. Use the scale exercised in HybridKKT's GPU examples/tests.
    if o.execution_mode == :gpu_hybrid && settings.kkt_system == RegularizedHybridKKTSystem
        solver.kkt.gamma[] = o.hybrid_gamma
    end
    return Base.invokelatest(MadNLP.solve!,solver)
end

function relaxation_schedule(o::Options)
    o.execution_mode == :gpu_condensed || return [0.0]
    initial = o.condensed_continuation ? [r for r in (1e-4,1e-6) if r > o.condensed_relaxation] : Float64[]
    return [initial; o.condensed_relaxation]
end

"Solve once by default; optionally use explicit condensed relaxation continuation."
function solve_stochastic_model(model,o::Options)
    stages = Dict{String,Any}[]
    total_iterations = 0
    schedule = relaxation_schedule(o)
    stats = nothing
    guard = RestorationGuard(limit=o.max_consecutive_restoration)
    for (stage,relaxation) in enumerate(schedule)
        restoration_before = guard.restoration_iterations
        guard.iteration_offset = total_iterations
        tolerance = max(o.tol,relaxation)
        settings = merge(solver_options(o),(;tol=tolerance,acceptable_tol=tolerance,
            bound_relax_factor=relaxation,max_iter=o.max_iter-total_iterations,
            intermediate_callback=guard))
        length(schedule) > 1 && @printf("Equality relaxation stage %d/%d: %.1e\n",stage,length(schedule),relaxation)
        seconds = @elapsed stats = run_madnlp(model,o,settings)
        guard = settings.intermediate_callback
        total_iterations += stats.iter
        push!(stages,Dict("equality_relaxation"=>relaxation,"solver_tolerance"=>tolerance,
            "iterations"=>stats.iter,"seconds"=>seconds,"status"=>string(stats.status),
            "objective"=>stats.objective,"original_constraint_violation"=>original_violation(model,stats.solution),
            "restoration_iterations"=>guard.restoration_iterations-restoration_before,
            "max_consecutive_restoration"=>guard.max_consecutive,
            "blocked_restoration_request"=>String(guard.blocked_phase)))
        (stats.status != MadNLP.SOLVE_SUCCEEDED || stage == length(schedule)) && break
        if total_iterations >= o.max_iter
            stats.status = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
            break
        end
        # Only the primal point is carried between ordinary interior-point solves.
        # Each stage retains the same variables, derivatives, and condensed KKT.
        copyto!(model.meta.x0,stats.solution)
    end
    return (;stats,guard,stages,total_iterations)
end
