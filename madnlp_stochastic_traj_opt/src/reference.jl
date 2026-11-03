# Deterministic fuel-optimal reference: loading and resampling on its own mesh with every interval split
# into REFERENCE_MESH_REFINEMENT equal pieces (`load_ref_traj` of the JAX script).

struct ReferenceTraj
    node_times::Vector{Float64}   # nondimensional, first entry zero
    steps::Vector{Float64}        # arc durations
    states::Matrix{Float64}       # 7 x (N+1)
    controls::Matrix{Float64}     # 3 x N thrust (nondimensional), zero-order hold
    fuel_consumed::Float64        # [kg]
end
n_arcs(r::ReferenceTraj) = length(r.steps)

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

"Directory of the deterministic fuel-optimal reference (the fuel-optimal sibling of the exported energy-optimal one)."
reference_directory(case::CaseData) = replace(case.reference_dir, "energy_optimal" => "fuel_optimal")

function load_reference(case::CaseData, o::Options)
    path = joinpath(reference_directory(case), case.id * ".npz")
    isfile(path) || error("missing deterministic reference $path; run deterministic_traj_opt.py first")
    data = NPZ.npzread(path, ["t_dense_days", "x_dense", "u_dense", "mesh_fraction", "x"])
    case.id == "earth_mars" && check_earth_mars_reference(case, path, data["x"])
    return build_reference(case, o, data)
end

"The JAX script's checks of the archived Earth--Mars reference: same units and ephemeris boundary states."
function check_earth_mars_reference(case::CaseData, path, states)
    units = NPZ.npzread(path, ["length_unit_km", "time_unit_s", "mass_unit_kg"])
    for (key, expected) in (("length_unit_km", case.length_unit), ("time_unit_s", case.time_unit),
                            ("mass_unit_kg", case.m0_wet))
        isapprox(Float64(units[key][]), expected; rtol = 1e-13, atol = 0) ||
            error("Reference normalization mismatch: $key")
    end
    # numpy.allclose(rtol=0, atol=1e-12) is elementwise.
    (all(abs.(states[1:NP, 1] .- case.x0[1:NP]) .<= 1e-12) && all(abs.(states[1:NP, end] .- case.xf) .<= 1e-12)) ||
        error("Reference boundary states do not match archived ephemerides")
end

function build_reference(case::CaseData, o::Options, data)
    truncated = o.truncated_arcs
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
    # the reference's own nodes, every interval split into r equal pieces
    r = REFERENCE_MESH_REFINEMENT
    node_times = vcat([mesh[k] .+ (mesh[k+1] - mesh[k]) .* (0:r-1) ./ r for k in 1:length(mesh)-1]..., mesh[end]) .* case.tof_nd
    n = length(node_times) - 1
    if truncated !== nothing
        truncated <= n || throw(ArgumentError("truncated_arcs must be at most $n"))
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

"Arc cost weights N dt_k / (t_f - t_0) of the reference mesh."
single_phase_arc_weights(ref::ReferenceTraj) = n_arcs(ref) .* ref.steps ./ sum(ref.steps)

"Terminal mean target: the full-transfer target or the truncated relative node."
terminal_mean_target(case::CaseData, o::Options, ref::ReferenceTraj) =
    o.truncated_arcs === nothing ? Vector(case.xf) : ref.states[1:NP, end]
