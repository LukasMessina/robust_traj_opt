# Small synthetic CPU workloads only. No optimiser, SPICE, saved trajectory,
# output writes, random state, or GPU context is used during precompilation.
using PrecompileTools: @setup_workload, @compile_workload

function precompile_single_phase_workload(id)
    o=Options(case_id=id,verbose=false)
    c=case_data(id);norm=build_normalization(c,o);p=ut_params(c,o,norm)
    ref=ReferenceTraj([0.,.001,.002],[.001,.001],repeat(collect(c.x0),1,3),fill(c.tmax*.01,3,2),0.)
    # the model's parameters as `_solve_problem` forms them (the defaults: throttle controls)
    o.control_parameterization==:throttle && (p=model_params(o,p))
    seed=build_initial_guess(c,o,norm,p,ref);sc=covariance_scaling(seed,norm);p=model_params(o,p)
    b=build_stochastic_model(c,o,ref,seed,sc,p,nothing);m=b.model;x=m.meta.x0
    NLPModels.obj(m,x);NLPModels.cons(m,x);NLPModels.jac_coord(m,x)
    NLPModels.hess_coord(m,x,ones(m.meta.ncon))
    return nothing
end

@setup_workload begin
    @compile_workload begin
        parse_cli(["--case=psyche","--mode=cpu"])
        precompile_single_phase_workload("lyapunov_l1_to_l2")
        precompile_single_phase_workload("earth_mars")
    end
end
