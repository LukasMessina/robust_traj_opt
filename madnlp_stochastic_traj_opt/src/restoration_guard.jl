# MadNLP 0.10 reports soft restoration after an accepted step and robust
# restoration before a step. Account for that difference when counting runs.
# Dispatch is specialized only for solvers carrying this application's guard.
Base.@kwdef mutable struct RestorationGuard <: MadNLP.AbstractUserCallback
    limit::Int = 2
    regular_iterations::Int = 0
    restoration_iterations::Int = 0
    consecutive::Int = 0
    max_consecutive::Int = 0
    last_restoration_iteration::Int = -2
    iteration_offset::Int = 0
    blocked_phase::Symbol = :none
    limit_exceeded::Bool = false
end

next_restoration_count(g::RestorationGuard,k) =
    k == g.last_restoration_iteration+1 ? g.consecutive+1 : 1

function allow_restoration!(g::RestorationGuard,k,phase)
    if next_restoration_count(g,k) > g.limit
        g.blocked_phase = phase
        g.limit_exceeded = true
        return false
    end
    return true
end

function record_restoration!(g::RestorationGuard,k,phase)
    allow_restoration!(g,k,phase) || return false
    g.consecutive = next_restoration_count(g,k)
    g.max_consecutive = max(g.max_consecutive,g.consecutive)
    g.last_restoration_iteration = k
    g.restoration_iterations += 1
    return true
end

function (g::RestorationGuard)(solver,::MadNLP.UserCallbackRegular)
    g.regular_iterations = solver.cnt.k
    # A callback precedes the regular step, which could fail without advancing
    # k. Only an actual gap in iteration indices breaks a restoration run.
    return true
end

function (g::RestorationGuard)(solver,::MadNLP.UserCallbackRestore)
    k = g.iteration_offset + solver.cnt.k - 1
    record_restoration!(g,k,:soft) || return false
    if g.consecutive == g.limit
        # The native routine checks this exact condition immediately after its
        # callback. Permit its return to REGULAR, but never its next loop step.
        theta = MadNLP.get_theta(MadNLP.get_c(solver))
        varphi = MadNLP.get_varphi(MadNLP.get_obj_val(solver),
            MadNLP.get_x_lr(solver),MadNLP.get_xl_r(solver),
            MadNLP.get_xu_r(solver),MadNLP.get_x_ur(solver),MadNLP.get_mu(solver))
        if !MadNLP.is_filter_acceptable(MadNLP.get_filter(solver),theta,varphi)
            g.blocked_phase = :soft
            g.limit_exceeded = true
            return false
        end
    end
    return true
end

function (g::RestorationGuard)(solver,::MadNLP.UserCallbackRobust)
    return record_restoration!(g,g.iteration_offset+solver.cnt.k,:robust)
end

const GuardedMadNLPSolver = MadNLP.MadNLPSolver{T,VT,VI,KKT,Model,CB,Iterator,IC,KV,RestorationGuard} where
    {T,VT,VI,KKT,Model,CB,Iterator,IC,KV}

# Keep native restoration compilation lazy: most runs never need these large
# routines, especially their GPU specializations. This is a function barrier,
# not a different restoration algorithm.
native_soft_restoration!(solver) = invoke(MadNLP.restore!,Tuple{MadNLP.AbstractMadNLPSolver},solver)
native_robust_restoration!(solver) = invoke(MadNLP.robust!,Tuple{MadNLP.AbstractMadNLPSolver},solver)

function MadNLP.restore!(solver::GuardedMadNLPSolver)
    g = solver.intermediate_callback
    allow_restoration!(g,g.iteration_offset+solver.cnt.k,:soft) || return MadNLP.USER_REQUESTED_STOP
    return Base.invokelatest(native_soft_restoration!,solver)::MadNLP.Status
end

function MadNLP.robust!(solver::GuardedMadNLPSolver)
    g = solver.intermediate_callback
    allow_restoration!(g,g.iteration_offset+solver.cnt.k,:robust) || return MadNLP.USER_REQUESTED_STOP
    return Base.invokelatest(native_robust_restoration!,solver)::MadNLP.Status
end
