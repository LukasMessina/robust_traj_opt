# Deterministic energy-optimal reference: loading and uniform resampling
# (`load_ref_traj` / `_build_uniform_ref_traj` of the JAX script).

struct ReferenceTraj
    node_times::Vector{Float64}   # nondimensional, first entry zero
    steps::Vector{Float64}        # arc durations
    states::Matrix{Float64}       # 7 x (N+1)
    controls::Matrix{Float64}     # 3 x N thrust (nondimensional), zero-order hold
    fuel_consumed::Float64        # [kg]
end
n_arcs(r::ReferenceTraj) = length(r.steps)

uniform_arc_count(case::CaseData, o::Options) = something(o.uniform_mesh_arcs, case.uniform_arcs)

"numpy.linspace: i * ((stop - start) / (num - 1)) + start, last entry pinned."
function numpy_linspace(start, stop, num)
    step = (stop - start) / (num - 1)
    y = [i * step + start for i in 0:num-1]
    y[end] = stop
    return y
end

"numpy.interp for increasing sample points, including its exact-hit and NaN branches."
function numpy_interp(x::AbstractVector, xp::AbstractVector, fp::AbstractVector)
    result = similar(x, Float64)
    last = length(xp)
    for (i, xv) in enumerate(x)
        if xv <= xp[1]
            result[i] = fp[1]
        elseif xv >= xp[last]
            result[i] = fp[last]
        else
            j = searchsortedlast(xp, xv)
            if xp[j] == xv
                result[i] = fp[j]
            else
                slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])
                result[i] = slope * (xv - xp[j]) + fp[j]
                if isnan(result[i])
                    result[i] = slope * (xv - xp[j+1]) + fp[j+1]
                    isnan(result[i]) && fp[j] == fp[j+1] && (result[i] = fp[j])
                end
            end
        end
    end
    return result
end

function load_reference(case::CaseData, o::Options)
    path = joinpath(REFERENCE_DIR, case.id * ".npz")
    isfile(path) || error("missing deterministic reference $path; run cr3bp_deterministic_traj_opt.py first")
    data = NPZ.npzread(path, ["t_dense_days", "x_dense", "u_dense", "mesh_fraction", "x"])
    return build_uniform_reference(case, o, data)
end

function build_uniform_reference(case::CaseData, o::Options, data)
    n = uniform_arc_count(case, o)
    n >= 1 || throw(ArgumentError("uniform_mesh_arcs must be positive"))
    truncated = o.truncated_uniform_mesh_arcs
    truncated === nothing || 1 <= truncated <= n ||
        throw(ArgumentError("truncated_uniform_mesh_arcs must be between 1 and $n"))
    dense_days = vec(Float64.(data["t_dense_days"]))
    dense_states = Float64.(data["x_dense"])
    dense_controls = Float64.(data["u_dense"])
    mesh = vec(Float64.(data["mesh_fraction"]))
    stored_states = Float64.(data["x"])
    # np.unique(..., return_index=True): sorted unique values, first occurrences.
    unique_index = Int[]
    for i in sortperm(dense_days; alg=MergeSort)
        (isempty(unique_index) || dense_days[unique_index[end]] != dense_days[i]) && push!(unique_index, i)
    end
    dense_times = dense_days[unique_index] .* 86400.0 ./ case.time_unit
    unique_states = dense_states[:, unique_index]
    horizon = mesh[end] * case.tof_nd
    dense_times[end] >= horizon - 1e-12 || error("the dense reference stops before the requested horizon")
    node_times = numpy_linspace(0.0, horizon, n + 1)
    if truncated !== nothing
        node_times = node_times[1:truncated+1]
        n = truncated
    end
    node_states = Matrix{Float64}(undef, NX, n + 1)
    for row in 1:NX
        node_states[row, :] = numpy_interp(node_times, dense_times, unique_states[row, :])
    end
    # Pin the endpoints to the deterministic solver's own boundary states.
    node_states[:, 1] = stored_states[:, 1]
    truncated === nothing && (node_states[:, end] = stored_states[:, end])
    # Each arc takes the duration-weighted mean of the dense zero-order-hold controls it spans.
    edge_times = dense_days .* 86400.0 ./ case.time_unit
    starts, ends = edge_times[1:end-1], edge_times[2:end]
    segment_controls = dense_controls[:, 1:end-1]
    node_controls = Matrix{Float64}(undef, NU, n)
    for k in 1:n
        overlap = max.(min.(ends, node_times[k+1]) .- max.(starts, node_times[k]), 0.0)
        node_controls[:, k] = segment_controls * overlap / sum(overlap)
    end
    fuel = case.m0_wet * (node_states[7, 1] - node_states[7, end])
    return ReferenceTraj(node_times, diff(node_times), node_states, node_controls, fuel)
end

"Terminal mean target: the full-transfer target or the truncated relative node."
terminal_mean_target(case::CaseData, o::Options, ref::ReferenceTraj) =
    o.truncated_uniform_mesh_arcs === nothing ? Vector(case.xf) : ref.states[1:NP, end]
