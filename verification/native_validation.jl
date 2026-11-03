# Post-solve Monte Carlo validation: the Python/JAX validation entry point, run on
# the device of the solve (cpu: JAX on the CPU cores, gpu: JAX on CUDA).

"JAX device of the validation: the execution mode's (cpu or gpu)."
validation_device(o::Options) = o.execution_mode

"Solution file the validator reads."
function validation_input(o::Options)
    directory = output_directory(o)
    return joinpath(directory, o.case_id == "psyche" ? "psyche_solution.npz" : o.case_id * ".npz")
end

function validation_command(o::Options)
    return Cmd(`$(o.validation_python) -m verification $(validation_device(o)) --case=$(o.case_id) --directory=$(output_directory(o))`; dir=REPOSITORY_ROOT)
end

"""
Fail before any solve when the validation Python cannot import the validators'
dependencies (the validator imports the JAX reference, hence CasADi too).
"""
function check_validation_python(runs)
    for python in unique(o.validation_python for o in runs if o.validate_after_solve)
        probe = `$python -c "import jax, diffrax, casadi, scipy, matplotlib"`
        usable = try
            success(pipeline(probe; stdout = devnull, stderr = devnull))
        catch   # missing executable
            false
        end
        usable && continue
        throw(ArgumentError("validation Python $python cannot import jax, casadi, scipy and matplotlib. " *
            "Set MADNLP_VALIDATION_PYTHON or --validation-python to a Python that can, or pass --no-validation."))
    end
    return nothing
end

"Free this process's cached device memory before JAX starts on the same GPU."
function release_gpu_memory()
    GC.gc(true)
    isdefined(@__MODULE__, :CUDA) && Base.invokelatest(() -> CUDA.reclaim())
    return nothing
end

"""
Run the Monte Carlo validation of a saved solve. A failure is reported, never
thrown: the solve and its files are kept, and the manual command is printed.
"""
function validate_solution(o::Options)
    input = validation_input(o)
    if !isfile(input)
        println("Validation skipped: no saved solution ($input).")
        return false
    end
    o.execution_mode != :cpu && release_gpu_memory()
    command = setenv(validation_command(o), merge(ENV, Dict("PYTHONUNBUFFERED" => "1")))
    println("Monte Carlo validation on the $(o.execution_mode): $(validation_command(o))")
    try
        run(command)
        return true
    catch err
        @warn "Monte Carlo validation failed; the solve is saved. Rerun it with the command above " *
              "(set MADNLP_VALIDATION_PYTHON or --validation-python to a Python with JAX)." exception = err
        return false
    end
end
