# Case data and the normalization of the JAX script.

"Case-dependent constants (exported bitwise from the Python case registry)."
struct CaseData
    id::String
    display_name::String
    mu::Float64
    length_unit::Float64
    time_unit::Float64
    velocity_unit::Float64
    thrust_unit::Float64
    m0_wet::Float64
    tmax::Float64                 # max_thrust_nd
    max_thrust_n::Float64
    ve::Float64                   # exhaust_velocity_nd
    tof_nd::Float64
    x0::SVector{7,Float64}        # x0_augmented_state
    xf::SVector{6,Float64}        # xf_state
    uniform_arcs::Int
end

function case_data(case_id::AbstractString)
    c = CASE_DATA[case_id]
    return CaseData(c.id, c.display_name, c.mu, c.length_unit, c.time_unit, c.velocity_unit,
        c.thrust_unit, c.m0_wet, c.max_thrust_nd, c.max_thrust_n, c.exhaust_velocity_nd,
        c.tof_nd, SVector{7}(c.x0_augmented), SVector{6}(c.xf_state), c.uniform_arcs)
end

"""
`D0 = diag(scale)`: the NLP carries P = D0^-1 Sigma D0^-1, K_tilde = K D0'/T_max and
S_tilde = S/T_max. `initial_std` equals `scale` except possibly on the mass.
"""
struct Normalization
    scale::SVector{7,Float64}
    initial_std::SVector{7,Float64}
end

function build_normalization(case::CaseData)
    pv = [fill(INITIAL_POSITION_STD_KM / case.length_unit, 3);
          fill(INITIAL_VELOCITY_STD_KM_S / case.velocity_unit, 3)]
    return Normalization(SVector{7}([pv; MASS_SCALE / case.m0_wet]),
                         SVector{7}([pv; INITIAL_MASS_STD / case.m0_wet]))
end

initial_covariance(n::Normalization) = Matrix(Diagonal(Vector((n.initial_std ./ n.scale) .^ 2)))

"sigma_nd = sigma_dim T^(3/2) / L."
acceleration_diffusion_nd(case::CaseData, o::Options) =
    o.acceleration_diffusion_km_s32 * case.time_unit^1.5 / case.length_unit

"Normalized navigation variances (the diagonal of `navigation_covariance`)."
function navigation_variances(case::CaseData, o::Options, n::Normalization)
    std_nd = [fill(o.navigation_position_std_km / case.length_unit, 3);
              fill(o.navigation_velocity_std_km_s / case.velocity_unit, 3)]
    return (std_nd ./ n.scale[1:NP]) .^ 2
end

"Terminal-covariance targets: 1 / sqrt(target ratio) per position/velocity channel."
terminal_inverse_std(o::Options) =
    sqrt.([fill(o.position_covariance_reduction, 3); fill(o.velocity_covariance_reduction, 3)])
