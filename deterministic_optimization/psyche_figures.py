"""Psyche figures in the original Plotter style, including oriented cartography."""

from pathlib import Path
import json

import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy.optimize import minimize_scalar

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


def plot_passage(plotter, solution, mission, index):
    from .psyche import evaluate_phase

    body = "Earth" if index == 0 else "Mars"
    phase = solution["phases"][index]
    radius = (
        mission.settings.earth_radius_m if index == 0 else mission.settings.mars_radius_m
    ) / 1000
    if index == 0:
        closest = 0.0
    else:
        candidates = [
            (float(tau), np.linalg.norm(state[:3]))
            for tau, state in zip(phase["mesh"], phase["states_si"])
        ]
        for a, b in zip(phase["mesh"][:-1], phase["mesh"][1:]):
            answer = minimize_scalar(
                lambda tau: np.linalg.norm(evaluate_phase(solution, index, tau, mission)[:3]),
                bounds=(a, b),
                method="bounded",
                options={"xatol": 1e-14},
            )
            candidates.append((answer.x, answer.fun))
        closest = min(candidates, key=lambda v: v[1])[0]
    anchor = evaluate_phase(solution, index, closest, mission)
    epoch = phase["start_et"] + closest * phase["duration_s"]
    rotation, utc, cartography = body_orientation(body, epoch)
    # Widen the physical window and resample its longer trajectory segment below.
    # This is 1.75 times the preceding 1.932-factor passage view.
    limit = 3.381 * max(2.4 * radius, 1.45 * np.linalg.norm(anchor[:3]) / 1000)
    coarse = np.unique(np.r_[np.linspace(0, 1, 1601), phase["mesh"], closest])
    positions = np.array(
        [evaluate_phase(solution, index, tau, mission)[:3] / 1000 for tau in coarse]
    )
    inside = np.flatnonzero(np.linalg.norm(positions, axis=1) < limit * 1.7)
    if not len(inside):
        raise ValueError("Close approach not in plotting window")
    lo, hi = coarse[max(0, inside[0] - 1)], coarse[min(len(coarse) - 1, inside[-1] + 1)]
    tau = np.unique(np.r_[np.linspace(lo, hi, 801), closest])
    positions = np.array([evaluate_phase(solution, index, s, mission)[:3] / 1000 for s in tau])
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
    geometry, colors, axes = surface_geometry(body, rotation)
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
    )
    hidden = occulted(positions, viewer, rotation, axes)
    shown = positions.copy()
    shown[hidden | (np.max(abs(shown), axis=1) > limit)] = np.nan
    ax.plot(
        *shown.T,
        color=plotter.BLACK,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        label="Departure" if index == 0 else "Flyby",
        zorder=4,
    )
    selection = np.unique(np.linspace(0, len(tau) - 1, 12).astype(int))
    selection = selection[(~hidden[selection]) & visible_window[selection]]
    if max_thrust > 0 and len(selection):
        arrows = 1.20 * radius * thrust[selection] / max_thrust
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
        label="TIP" if index == 0 else "Closest approach",
        zorder=6,
    )
    ax.set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        zlim=(-limit, limit),
        xlabel="x [km]",
        ylabel="y [km]",
        zlabel="z [km]",
    )
    ax.set_box_aspect((1, 1, 1))
    plotter._style_3d_axis(ax)
    # The shared style sets its own camera; apply the common Psyche view last.
    ax.view_init(azim=-60, elev=30)
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
    fig.subplots_adjust(left=0.02, right=0.82, bottom=0.10, top=0.73)
    plotter._heliocentric_legend(fig, ax, gap=0.003)
    save(plotter, fig, body.lower() + "_passage")
    return dict(
        body=body,
        surface_epoch_et=float(epoch),
        surface_epoch_utc=utc,
        orientation=cartography,
        rotation_body_fixed_to_j2000=rotation.tolist(),
        ellipsoid_axes_km=axes,
        displayed_half_width_km=float(limit),
        maximum_displayed_thrust_n=float(max_thrust),
        maximum_arrow_length_km=float(1.20 * radius),
        arrow_linewidth_points=1.6,
        camera_azimuth_degrees=-60,
        camera_elevation_degrees=30,
        closest_approach_altitude_above_constraint_sphere_km=float(
            np.linalg.norm(anchor[:3]) / 1000 - radius
        ),
        note="Static surface orientation at surface_epoch_et; inertial trajectory spans a time interval. Visual ellipsoid does not alter spherical collision constraints.",
    )


def plot_psyche_export(prefix, data, meta):
    from .psyche import Mission, Settings, load_solution

    mission = Mission(Settings(**meta["configuration"]["settings"]))
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
            (position[:, 0], "TIP", plotter.EARTH_COLOR),
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
    plotter.plot_inertial_control(t, u, meta["max_thrust_n"])
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
    meta["passage_figures"] = [plot_passage(plotter, solution, mission, i) for i in (0, 2)]
