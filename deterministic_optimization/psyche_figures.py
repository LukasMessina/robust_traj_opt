"""Psyche figures in the original Plotter style, including oriented cartography."""

from pathlib import Path
import json

import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
from PIL import Image

from .plotting import Plotter, plt, save
from .plot_assets import ASSETS, EARTH_PCK, digest


def surface_geometry(body, rotation, resolution=360):
    """Texture cell edges follow map edges, avoiding longitude flips/half-pixel shifts."""
    filename = "earth_october_5400.jpg" if body == "Earth" else "mars_viking_8192.jpg"
    with Image.open(ASSETS / filename) as image:
        colors = (
            np.array(
                image.convert("RGB").resize((2 * resolution, resolution), Image.Resampling.LANCZOS)
            )
            / 255.0
        )
    longitude, latitude = np.meshgrid(
        np.linspace(-np.pi, np.pi, 2 * resolution + 1),
        np.linspace(np.pi / 2, -np.pi / 2, resolution + 1),
    )
    if body == "Earth":
        a, c = 6378.137, 6356.752314245  # WGS84 geodetic latitude in BMNG.
        e2 = 1 - (c / a) ** 2
        n = a / np.sqrt(1 - e2 * np.sin(latitude) ** 2)
        fixed = np.array(
            [
                n * np.cos(latitude) * np.cos(longitude),
                n * np.cos(latitude) * np.sin(longitude),
                n * (1 - e2) * np.sin(latitude),
            ]
        )
    else:
        a, c = 3396.19, 3376.20
        radius = 1 / np.sqrt((np.cos(latitude) / a) ** 2 + (np.sin(latitude) / c) ** 2)
        fixed = np.array(
            [
                radius * np.cos(latitude) * np.cos(longitude),
                radius * np.cos(latitude) * np.sin(longitude),
                radius * np.sin(latitude),
            ]
        )
    inertial = np.einsum("ij,jkl->ikl", rotation, fixed)
    return inertial, colors, (a, a, c)


def body_orientation(body, epoch):
    """Post-processing SPICE only; the NLP remains entirely cache based."""
    import spiceypy as spice

    manifest = json.loads((ASSETS / "manifest.json").read_text())
    record = dict(manifest[body.lower()])
    record["sha256"] = digest(ASSETS / record["file"])
    record["orientation_kernel_sha256"] = digest(ASSETS / record["orientation_kernel"])
    paths = [
        ASSETS / "pck00011.tpc",
        ASSETS / EARTH_PCK,
        Path(__file__).resolve().parents[1] / "data/jpl_kernels/naif0012.tls",
    ]
    for item in manifest["source_files"]:
        if item["file"] in ("pck00011.tpc", EARTH_PCK, "earth_october_5400.jpg"):
            if digest(ASSETS / item["file"]) != item["sha256"]:
                raise ValueError("Changed plot asset: " + item["file"])
    if digest(ASSETS / manifest["mars"]["file"]) != manifest["mars"]["sha256"]:
        raise ValueError("Changed Mars map")
    loaded = []
    try:
        for path in paths:
            spice.furnsh(str(path))
            loaded.append(path)
        rotation = spice.pxform(record["frame"], "J2000", float(epoch))
        utc = spice.et2utc(float(epoch), "ISOC", 3)
    finally:
        for path in reversed(loaded):
            spice.unload(str(path))
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    if np.linalg.det(rotation) < 0:
        raise ValueError("Orientation reverses handedness")
    return rotation, utc, record


def occulted(points, viewer, rotation, axes):
    """Exact ray/ellipsoid test for an orthographic camera looking towards origin."""
    p = (rotation.T @ points.T).T / axes
    ray = (rotation.T @ viewer) / axes
    a, b, c = np.dot(ray, ray), p @ ray, np.sum(p * p, axis=1) - 1
    discriminant = b * b - a * c
    near = (-b - np.sqrt(np.maximum(discriminant, 0))) / a
    return (discriminant > 0) & (near > 0)


def plot_earth_departure(plotter, solution, mission):
    """Textured Earth with the first solar phase from the fixed Earth SOI exit.

    Optimised Sun-centred states are translated to Earth-centred J2000 with the
    cached Earth ephemeris at each plotted epoch. The globe is enlarged for
    display only; trajectory coordinates are not scaled.
    """
    from .psyche import evaluate_phase

    phase = solution["phases"][0]

    def evaluate(tau):
        state = evaluate_phase(solution, 0, tau, mission).copy()
        state[:6] -= mission.cache.state(399, phase["start_et"] + tau * phase["duration_s"])
        return state

    radius = mission.settings.earth_radius_m / 1000
    anchor = evaluate(0.0)
    epoch = phase["start_et"]
    rotation, utc, cartography = body_orientation("Earth", epoch)
    # Window width retained from the original passage views (3.381 times the
    # original close-up half-width).
    limit = 3.381 * max(2.4 * radius, 1.45 * np.linalg.norm(anchor[:3]) / 1000)
    arrow_length = 0.10 * limit
    coarse = np.unique(np.r_[np.linspace(0, 1, 1601), phase["mesh"], 0.0])
    positions = np.array([evaluate(tau)[:3] / 1000 for tau in coarse])
    if not np.any(np.linalg.norm(positions, axis=1) < limit * 1.7):
        raise ValueError("Earth departure not in plotting window")
    outside = np.flatnonzero(np.linalg.norm(positions, axis=1) >= limit * 1.7)
    lo, hi = 0.0, coarse[outside[0]] if len(outside) else 1.0
    tau = np.unique(np.r_[np.linspace(lo, hi, 801), 0.0])
    positions = np.array([evaluate(s)[:3] / 1000 for s in tau])
    k = np.clip(np.searchsorted(phase["mesh"], tau, side="right") - 1, 0, len(phase["mesh"]) - 2)
    thrust = phase["thrust_n"][k]
    visible_window = np.linalg.norm(positions, axis=1) < limit
    max_thrust = np.linalg.norm(thrust[visible_window], axis=1).max()
    # Match the main heliocentric view, including which sides carry the axis labels.
    elevation, azimuth = np.radians([30, -60])
    viewer = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    fig = plt.figure(figsize=plotter.THREE_D_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")
    ax.set(
        xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
        xlabel="x [km]", ylabel="y [km]", zlabel="z [km]",
    )
    ax.set_box_aspect((1, 1, 1))
    plotter._style_3d_axis(ax)
    ax.view_init(azim=-60, elev=30)
    fig.subplots_adjust(left=0.02, right=0.82, bottom=0.10, top=0.73)
    ax.apply_aspect()
    geometry, colors, axes = surface_geometry("Earth", rotation)
    # Display-only magnification: the textured globe has twice the former
    # centre marker's diameter. Preserve its orientation and oblateness.
    screen_right = np.array([-viewer[1], viewer[0], 0.0])
    screen_right /= np.linalg.norm(screen_right)
    edge = axes[0] * screen_right
    px, py, _ = proj3d.proj_transform(
        [0.0, edge[0]], [0.0, edge[1]], [0.0, edge[2]], ax.get_proj()
    )
    screen_points = ax.transData.transform(np.column_stack((px, py)))
    radius_pixels = np.linalg.norm(screen_points[1] - screen_points[0])
    target_radius_pixels = np.sqrt(plotter.SMALL_MARKER_SIZE) * fig.dpi / 72.0
    surface_scale = target_radius_pixels / radius_pixels
    geometry = geometry * surface_scale
    displayed_axes = np.asarray(axes) * surface_scale
    ax.plot_surface(
        *geometry,
        rstride=1,
        cstride=1,
        facecolors=colors,
        shade=False,
        linewidth=0,
        antialiased=False,
        rasterized=True,
        zorder=2,
        label="Earth",
    )
    hidden = occulted(positions, viewer, rotation, displayed_axes)
    shown = positions.copy()
    shown[hidden | (np.max(abs(shown), axis=1) > limit)] = np.nan
    ax.plot(
        *shown.T,
        color=plotter.BLACK,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        label="Departure",
        zorder=4,
    )
    selection = np.unique(np.linspace(0, len(tau) - 1, 12).astype(int))
    selection = selection[(~hidden[selection]) & visible_window[selection]]
    # Same coast cutoff as the main 3D plot.
    arrow_cutoff = 1e-6 * mission.settings.thrust_max_n
    selection = selection[np.linalg.norm(thrust[selection], axis=1) > arrow_cutoff]
    if max_thrust > 0 and len(selection):
        arrows = arrow_length * thrust[selection] / max_thrust
        ax.quiver(
            *positions[selection].T,
            *arrows.T,
            color=plotter.THRUST_ARROW_3D_COLOR,
            arrow_length_ratio=0.22,
            linewidths=1.6,
            label="Thrust",
            zorder=5,
        )
    ax.scatter(
        *(anchor[:3] / 1000),
        color=plotter.ENDPOINT_COLOR,
        edgecolor=plotter.BLACK,
        s=plotter.SMALL_MARKER_SIZE,
        label="Earth SOI exit",
        zorder=6,
    )
    projected = positions.copy()
    projected[np.max(abs(projected), axis=1) > limit] = np.nan
    plotter._plot_trajectory_projections_3d(ax, projected.T, xz_far_side=True)
    # Plane projections lie behind the textured body and visible trajectory.
    for line in ax.lines[-3:]:
        line.set_zorder(1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(3, prune="both"))
    ax.tick_params(axis="z", pad=5)
    ax.zaxis.labelpad = 8
    plotter._heliocentric_legend(fig, ax, gap=0.003)
    save(plotter, fig, "earth_passage")
    return dict(
        body="Earth",
        trajectory_source="Optimised first solar phase from fixed Earth SOI exit",
        trajectory_frame="Earth-centred J2000",
        trajectory_start_et=float(phase["start_et"] + lo * phase["duration_s"]),
        trajectory_end_et=float(phase["start_et"] + hi * phase["duration_s"]),
        anchor_event="Earth SOI exit",
        surface_epoch_et=float(epoch),
        surface_epoch_utc=utc,
        orientation=cartography,
        rotation_body_fixed_to_j2000=rotation.tolist(),
        ellipsoid_axes_km=axes,
        displayed_ellipsoid_axes_km=displayed_axes.tolist(),
        surface_display_scale=float(surface_scale),
        surface_display_diameter_points=float(2 * np.sqrt(plotter.SMALL_MARKER_SIZE)),
        displayed_half_width_km=float(limit),
        maximum_displayed_thrust_n=float(max_thrust),
        maximum_arrow_length_km=float(arrow_length if len(selection) else 0.0),
        arrow_thrust_cutoff_n=float(arrow_cutoff),
        arrow_linewidth_points=1.6,
        camera_azimuth_degrees=-60,
        camera_elevation_degrees=30,
        anchor_altitude_above_surface_km=float(np.linalg.norm(anchor[:3]) / 1000 - radius),
        note=(
            "Static surface orientation at surface_epoch_et; inertial trajectory spans a time interval. "
            "Earth at SOI exit is visually enlarged to twice the former centre marker diameter. "
            "Visual ellipsoid does not alter trajectory coordinates or spherical collision constraints."
        ),
    )


def plot_psyche_export(prefix, data, meta):
    from .psyche import Mission, Settings, load_solution

    mission = Mission(Settings.from_dict(meta["configuration"]["settings"]))
    solution = load_solution(Path(prefix).with_suffix(".npz"))
    plotter = Plotter(prefix)
    t, x, u = data["t_dense_days"], data["x_dense"], data["u_dense"] * meta["force_unit_n"]
    # Main trajectory arrows retain the 0.48 AU display scale.
    # Normalise by Psyche's own thrust limit so a smaller engine does not shrink arrows.
    au = 149597870700.0
    position = x[:3] * meta["length_unit_km"] * 1000 / au
    index = np.unique(np.searchsorted(t, np.linspace(t[0], t[-1], 60)))
    index = index[np.linalg.norm(u[:, index], axis=0) > 1e-6 * meta["max_thrust_n"]]
    arrows = 0.48 * u[:, index] / meta["max_thrust_n"]
    meta["heliocentric_arrow_length_at_full_throttle_au"] = 0.48
    meta["heliocentric_arrow_linewidth_3d_points"] = 0.75
    meta["projection_arrow_width_axes_fraction"] = 0.0046
    for dimension in (2, 3):
        fig = plt.figure(
            figsize=plotter.SINGLE_FIGSIZE if dimension == 2 else plotter.THREE_D_FIGSIZE
        )
        ax = fig.add_subplot(
            111, **({"projection": "3d", "computed_zorder": False} if dimension == 3 else {})
        )
        for name, color in (("earth", plotter.EARTH_COLOR), ("mars", plotter.MARS_COLOR)):
            track = data[name + "_track_si"][:3] / au
            ax.plot(
                *track[:dimension],
                "--",
                color=color,
                lw=plotter.REFERENCE_LINE_WIDTH,
                label=name.title() + " orbit",
            )
        ax.plot(
            *position[:dimension],
            color=plotter.BLACK,
            lw=plotter.TRAJECTORY_LINE_WIDTH,
            label="Transfer",
        )
        if dimension == 3:
            ax.quiver(
                *position[:, index],
                *arrows,
                color=plotter.THRUST_ARROW_3D_COLOR,
                arrow_length_ratio=0.18,
                linewidths=0.75,
                label="Thrust",
                zorder=5,
            )
        else:
            ax.quiver(
                *position[:2, index],
                *arrows[:2],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=plotter.THRUST_ARROW_3D_COLOR,
                width=0.0046,
                label="Thrust",
                zorder=5,
            )
        for point, label, color in (
            (np.zeros(3), "Sun", plotter.SUN_COLOR),
            (position[:, 0], mission.initial_event_name.replace("_", " "), plotter.EARTH_COLOR),
            (position[:, -1], "Psyche", plotter.GREEN),
        ):
            ax.scatter(
                *point[:dimension],
                s=plotter.SMALL_MARKER_SIZE,
                color=color,
                edgecolor=plotter.BLACK,
                linewidth=0.6,
                label=label,
                zorder=6,
            )
        ax.set(xlabel="x [AU]", ylabel="y [AU]")
        if dimension == 3:
            ax.set_zlabel("z [AU]")
            plotter._style_3d_axis(ax)
            ax.view_init(azim=-60, elev=30)
            plotter._set_axes_equal_3d(ax)
            plotter._plot_trajectory_projections_3d(ax, position, xz_far_side=True)
            fig.subplots_adjust(left=0.02, right=0.82, bottom=0.10, top=0.73)
        else:
            plotter._style_2d_axis(ax, equal_axis=True)
            ax.margins(0.12)
            fig.tight_layout(rect=(0, 0, 1, 0.76))
        plotter._heliocentric_legend(fig, ax, gap=0.003 if dimension == 3 else 0.02)
        save(plotter, fig, "3d" if dimension == 3 else "projections")
    plotter.plot_inertial_control(
        t, u, meta["max_thrust_n"],
        angle_mask_thrust_n=1.5e-4 if meta["configuration"]["objective"] == "fuel" else 1e-5,
    )
    plotter.plot_interval_defect(
        data["mesh_fraction"],
        meta["tof_days"],
        data["interval_defect_scaled"],
        meta["defect_tolerance"],
    )
    plotter.plot_collocation_approx_errors(
        t,
        data["position_error_m"],
        data["velocity_error_m_per_s"],
        data["mass_consumption_abs_difference_kg"],
    )
    # The Mars 3D passage figure is intentionally not generated.
    meta["passage_figures"] = [plot_earth_departure(plotter, solution, mission)]
