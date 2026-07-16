"""Plotting utilities for CR3BP transfer results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.usetex": True,
    }
)


class Plotter:
    AXIS_LABELS = ("x [LU]", "y [LU]", "z [LU]")
    PROJECTION_SUFFIXES = {
        (0, 1): "xy",
        (0, 2): "xz",
        (1, 2): "yz",
    }

    BLACK = "#000000"
    BLUE = "#1f77b4"
    ORANGE = "#ff7f0e"
    GREEN = "#2ca02c"
    RED = "#d62728"
    PURPLE = "#9b6fd3"
    GREY = "#7f7f7f"
    LIGHT_GREY = "#d9d9d9"

    TRAJECTORY_COLOR = BLACK
    DEPARTURE_COLOR = BLUE
    TARGET_COLOR = GREEN
    THRUST_COLOR = RED
    THRUST_ARROW_3D_COLOR = PURPLE
    ENDPOINT_COLOR = ORANGE
    MOON_COLOR = GREY
    LAGRANGE_COLOR = BLACK
    L1_COLOR = BLUE
    L2_COLOR = GREEN

    FIGURE_DPI = 240
    SINGLE_FIGSIZE = (4.7, 4.3)
    TRIPLE_FIGSIZE = (7.4, 2.85)
    WIDE_FIGSIZE = (5.8, 3.35)
    SQUARE_DIAGNOSTIC_FIGSIZE = (4.2, 4.2)
    THREE_PANEL_FIGSIZE = (7.4, 2.65)
    THREE_D_FIGSIZE = (5.6, 5.1)

    LABEL_SIZE = 11
    TICK_SIZE = 9
    DIAGNOSTIC_LABEL_SIZE = 10
    DIAGNOSTIC_TICK_SIZE = 8
    VERIFICATION_LABEL_SIZE = 9.5
    VERIFICATION_TICK_SIZE = 8
    PROJECTION_TICK_SIZE = 8
    THREE_D_TICK_SIZE = 8
    LEGEND_SIZE = 8.5
    SPINE_WIDTH = 1.05
    MAJOR_TICK_WIDTH = 1.0
    MINOR_TICK_WIDTH = 0.8
    TRAJECTORY_LINE_WIDTH = 1.7
    REFERENCE_LINE_WIDTH = 1.15
    GUIDE_LINE_WIDTH = 1.05
    MARKER_SIZE = 26
    SMALL_MARKER_SIZE = 16

    def __init__(self, output_prefix: Path):
        self.output_prefix = Path(output_prefix)
        self.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    def _path(self, suffix: str) -> Path:
        return self.output_prefix.with_name(f"{self.output_prefix.name}_{suffix}").with_suffix(".png")

    def _style_2d_axis(
        self,
        ax,
        *,
        equal_axis: bool = False,
        grid: bool = True,
        grid_which: str = "major",
        tick_size: float | None = None,
        label_size: float | None = None,
    ) -> None:
        tick_size = self.TICK_SIZE if tick_size is None else tick_size
        label_size = self.LABEL_SIZE if label_size is None else label_size
        ax.set_facecolor("white")
        ax.minorticks_on()
        if grid:
            ax.grid(True, which=grid_which, color=self.LIGHT_GREY, alpha=0.8, linewidth=0.55)
        else:
            ax.grid(False)
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            top=False,
            right=False,
            colors=self.BLACK,
            labelsize=tick_size,
            width=self.MAJOR_TICK_WIDTH,
            length=5.0,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            top=False,
            right=False,
            colors=self.BLACK,
            width=self.MINOR_TICK_WIDTH,
            length=2.8,
        )
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)
        ax.title.set_size(label_size)
        ax.xaxis.label.set_color(self.BLACK)
        ax.yaxis.label.set_color(self.BLACK)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(self.BLACK)
            spine.set_linewidth(self.SPINE_WIDTH)
        if equal_axis:
            ax.set_aspect("equal", adjustable="box")

    def _legend(self, ax, **kwargs) -> None:
        handles, labels = ax.get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_") and label not in unique:
                unique[label] = handle
        if unique:
            kwargs.setdefault("frameon", False)
            kwargs.setdefault("fontsize", self.LEGEND_SIZE)
            legend = ax.legend(
                unique.values(),
                unique.keys(),
                **kwargs,
            )
            if kwargs.get("frameon", False):
                legend.get_frame().set_linewidth(0.8)

    def _libration_point_from_label(self, label: str) -> str | None:
        normalized = label.replace("_", "").replace(" ", "").upper()
        if "L1" in normalized:
            return "L1"
        if "L2" in normalized:
            return "L2"
        return None

    def _libration_color(self, name: str | None) -> str:
        if name == "L1":
            return self.L1_COLOR
        if name == "L2":
            return self.L2_COLOR
        return self.ENDPOINT_COLOR

    def _figure_legend(self, fig, axes: Iterable[Any], *, ncol: int, **kwargs) -> None:
        unique: dict[str, Any] = {}
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label and not label.startswith("_") and label not in unique:
                    unique[label] = handle
        if unique:
            kwargs.setdefault("loc", "lower center")
            kwargs.setdefault("bbox_to_anchor", (0.5, 0.01))
            kwargs.setdefault("frameon", False)
            kwargs.setdefault("fontsize", self.LEGEND_SIZE)
            fig.legend(
                unique.values(),
                unique.keys(),
                ncol=ncol,
                **kwargs,
            )

    def _projection_axes_for_case(self, case: Any) -> tuple[tuple[int, int], ...]:
        if case.test_case_id == "lyapunov_l1_to_l2":
            return ((0, 1),)
        return ((0, 1), (0, 2), (1, 2))

    def _plot_system_points_2d(
        self,
        ax,
        case: Any,
        axis_0: int,
        axis_1: int,
        lagrange: Mapping[str, float],
    ) -> None:
        moon = np.array([1.0 - case.mu, 0.0, 0.0])
        ax.scatter(
            moon[axis_0],
            moon[axis_1],
            color=self.MOON_COLOR,
            edgecolor=self.BLACK,
            linewidth=0.6,
            marker="o",
            s=58,
            zorder=6,
            label="Moon",
        )
        for name in ("L1", "L2"):
            point = np.array([lagrange[name], 0.0, 0.0])
            ax.scatter(
                point[axis_0],
                point[axis_1],
                color=self._libration_color(name),
                marker="o",
                s=self.SMALL_MARKER_SIZE,
                zorder=7,
                label=rf"${name[0]}_{name[1]}$",
            )

    def _plot_thrust_arrows_2d(
        self,
        ax,
        case: Any,
        x_dense: np.ndarray,
        u_dense: np.ndarray,
        axis_0: int,
        axis_1: int,
    ) -> None:
        u_norm = np.linalg.norm(u_dense, axis=0)
        active = np.flatnonzero(u_norm > 0.03 * case.max_thrust_nd)
        if not active.size:
            return
        chosen = active[np.unique(np.linspace(0, active.size - 1, min(26, active.size)).astype(int))]
        direction = u_dense[:, chosen] / np.maximum(u_norm[chosen], 1e-14)
        thrust_ratio = u_norm[chosen] / max(case.max_thrust_nd, float(np.max(u_norm)))
        arrow_scale_lu = 0.025
        arrow_width = 0.0060 if case.test_case_id == "halo_l2_to_halo_l1" else 0.0046
        ax.quiver(
            x_dense[axis_0, chosen],
            x_dense[axis_1, chosen],
            direction[axis_0] * thrust_ratio * arrow_scale_lu,
            direction[axis_1] * thrust_ratio * arrow_scale_lu,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=self.THRUST_ARROW_3D_COLOR,
            width=arrow_width,
            headwidth=3.0,
            headlength=3.8,
            alpha=0.82,
            zorder=5,
            label="Thrust direction",
        )

    def _plot_projection(
        self,
        ax,
        case: Any,
        x_dense: np.ndarray,
        u_dense: np.ndarray,
        departure_orbit: np.ndarray | None,
        target_orbit: np.ndarray | None,
        lagrange: Mapping[str, float],
        axis_0: int,
        axis_1: int,
    ) -> None:
        departure_point = self._libration_point_from_label(case.departure_label)
        target_point = self._libration_point_from_label(case.target_label)
        departure_color = self._libration_color(departure_point)
        target_color = self._libration_color(target_point)
        if departure_orbit is not None:
            ax.plot(
                departure_orbit[axis_0],
                departure_orbit[axis_1],
                color=departure_color,
                linestyle="-",
                lw=self.REFERENCE_LINE_WIDTH,
                label=case.departure_label,
            )
        if target_orbit is not None:
            ax.plot(
                target_orbit[axis_0],
                target_orbit[axis_1],
                color=target_color,
                linestyle="-",
                lw=self.REFERENCE_LINE_WIDTH,
                label=case.target_label,
            )
        ax.plot(
            x_dense[axis_0],
            x_dense[axis_1],
            color=self.TRAJECTORY_COLOR,
            lw=self.TRAJECTORY_LINE_WIDTH,
            zorder=4,
            label="Transfer",
        )
        self._plot_thrust_arrows_2d(ax, case, x_dense, u_dense, axis_0, axis_1)
        self._plot_system_points_2d(ax, case, axis_0, axis_1, lagrange)
        ax.scatter(
            x_dense[axis_0, 0],
            x_dense[axis_1, 0],
            color=departure_color,
            edgecolor=self.BLACK,
            linewidth=0.45,
            marker="D",
            s=0.72 * self.MARKER_SIZE,
            zorder=8,
            label=r"$x_0$",
        )
        ax.scatter(
            x_dense[axis_0, -1],
            x_dense[axis_1, -1],
            color=target_color,
            edgecolor=self.BLACK,
            linewidth=0.45,
            marker="D",
            s=0.72 * self.MARKER_SIZE,
            zorder=8,
            label=r"$x_f$",
        )
        ax.set_xlabel(self.AXIS_LABELS[axis_0])
        ax.set_ylabel(self.AXIS_LABELS[axis_1])
        self._style_2d_axis(ax, equal_axis=True, tick_size=self.PROJECTION_TICK_SIZE)

    def save_projection_figure(
        self,
        case: Any,
        x_dense: np.ndarray,
        u_dense: np.ndarray,
        departure_orbit: np.ndarray | None,
        target_orbit: np.ndarray | None,
        lagrange: Mapping[str, float],
    ) -> Path:
        axes_to_plot = self._projection_axes_for_case(case)
        if len(axes_to_plot) == 1:
            fig, ax = plt.subplots(figsize=self.SINGLE_FIGSIZE, dpi=self.FIGURE_DPI)
            axes_array = [ax]
        else:
            fig, axes = plt.subplots(1, len(axes_to_plot), figsize=self.TRIPLE_FIGSIZE, dpi=self.FIGURE_DPI)
            axes_array = list(np.atleast_1d(axes))

        for ax, (axis_0, axis_1) in zip(axes_array, axes_to_plot):
            self._plot_projection(
                ax,
                case,
                x_dense,
                u_dense,
                departure_orbit,
                target_orbit,
                lagrange,
                axis_0,
                axis_1,
            )

        is_single_projection = len(axes_to_plot) == 1
        legend_anchor_y = 0.90 if is_single_projection else 0.995
        layout_top = 0.76 if is_single_projection else 0.82

        self._figure_legend(
            fig,
            axes_array,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_anchor_y),
            frameon=True,
            fancybox=False,
            edgecolor=self.BLACK,
            facecolor="white",
            framealpha=1.0,
            fontsize=7.4,
            columnspacing=1.0,
            handlelength=1.7,
            borderpad=0.35,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, layout_top))
        suffix = self.PROJECTION_SUFFIXES[axes_to_plot[0]] if len(axes_to_plot) == 1 else "projections"
        output_path = self._path(suffix)
        fig.savefig(output_path, dpi=self.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def _set_axes_equal_3d(self, ax) -> None:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        ranges = np.array(
            [
                x_limits[1] - x_limits[0],
                y_limits[1] - y_limits[0],
                z_limits[1] - z_limits[0],
            ]
        )
        centers = np.array([sum(x_limits), sum(y_limits), sum(z_limits)]) * 0.5
        radius = 0.5 * max(ranges)
        ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
        ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
        ax.set_zlim3d(centers[2] - radius, centers[2] + radius)

    def _style_3d_axis(self, ax) -> None:
        ax.set_facecolor("white")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            axis.pane.set_edgecolor(self.BLACK)
            axis.label.set_size(self.LABEL_SIZE)
            axis.label.set_color(self.BLACK)
        ax.tick_params(axis="both", which="major", colors=self.BLACK, labelsize=self.THREE_D_TICK_SIZE, width=self.MAJOR_TICK_WIDTH)
        ax.tick_params(axis="z", which="major", pad=12.0)
        ax.zaxis.labelpad = 14.0
        ax.grid(True, color=self.LIGHT_GREY, linewidth=0.5)
        ax.view_init(azim=78, elev=15)

    def _plot_thrust_arrows_3d(self, ax, case: Any, x_dense: np.ndarray, u_dense: np.ndarray) -> None:
        u_norm = np.linalg.norm(u_dense, axis=0)
        active = np.flatnonzero(u_norm > 0.03 * case.max_thrust_nd)
        if not active.size:
            return
        chosen = active[np.unique(np.linspace(0, active.size - 1, min(30, active.size)).astype(int))]
        direction = u_dense[:, chosen] / np.maximum(u_norm[chosen], 1e-14)
        thrust_ratio = u_norm[chosen] / max(case.max_thrust_nd, float(np.max(u_norm)))
        arrows = direction * thrust_ratio * 0.032
        ax.quiver(
            x_dense[0, chosen],
            x_dense[1, chosen],
            x_dense[2, chosen],
            arrows[0],
            arrows[1],
            arrows[2],
            color=self.THRUST_ARROW_3D_COLOR,
            arrow_length_ratio=0.18,
            linewidths=0.75,
            alpha=0.82,
            label="Thrust direction",
        )

    def save_3d_plot(
        self,
        case: Any,
        x_dense: np.ndarray,
        u_dense: np.ndarray,
        departure_orbit: np.ndarray | None,
        target_orbit: np.ndarray | None,
        lagrange: Mapping[str, float],
    ) -> Path:
        fig = plt.figure(figsize=self.THREE_D_FIGSIZE, dpi=self.FIGURE_DPI)
        ax = fig.add_subplot(111, projection="3d")
        departure_point = self._libration_point_from_label(case.departure_label)
        target_point = self._libration_point_from_label(case.target_label)
        departure_color = self._libration_color(departure_point)
        target_color = self._libration_color(target_point)
        if departure_orbit is not None:
            ax.plot(
                departure_orbit[0],
                departure_orbit[1],
                departure_orbit[2],
                color=departure_color,
                linestyle="-",
                lw=self.REFERENCE_LINE_WIDTH,
                label=case.departure_label,
            )
        if target_orbit is not None:
            ax.plot(
                target_orbit[0],
                target_orbit[1],
                target_orbit[2],
                color=target_color,
                linestyle="-",
                lw=self.REFERENCE_LINE_WIDTH,
                label=case.target_label,
            )
        ax.plot(x_dense[0], x_dense[1], x_dense[2], color=self.TRAJECTORY_COLOR, lw=self.TRAJECTORY_LINE_WIDTH, label="Transfer")
        self._plot_thrust_arrows_3d(ax, case, x_dense, u_dense)
        ax.scatter(x_dense[0, 0], x_dense[1, 0], x_dense[2, 0], color=departure_color, edgecolor=self.BLACK, linewidth=0.45, marker="D", s=0.72 * self.MARKER_SIZE, label=r"$x_0$")
        ax.scatter(x_dense[0, -1], x_dense[1, -1], x_dense[2, -1], color=target_color, edgecolor=self.BLACK, linewidth=0.45, marker="D", s=0.72 * self.MARKER_SIZE, label=r"$x_f$")
        moon = np.array([1.0 - case.mu, 0.0, 0.0])
        ax.scatter(moon[0], moon[1], moon[2], color=self.MOON_COLOR, edgecolor=self.BLACK, linewidth=0.6, marker="o", s=58, label="Moon")
        for name in ("L1", "L2"):
            ax.scatter(lagrange[name], 0.0, 0.0, color=self._libration_color(name), marker="o", s=self.SMALL_MARKER_SIZE, label=rf"${name[0]}_{name[1]}$")
        ax.set_xlabel(self.AXIS_LABELS[0])
        ax.set_ylabel(self.AXIS_LABELS[1])
        ax.set_zlabel(self.AXIS_LABELS[2])
        self._style_3d_axis(ax)
        self._set_axes_equal_3d(ax)
        self._legend(
            ax,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=3,
            frameon=True,
            fancybox=False,
            edgecolor=self.BLACK,
            facecolor="white",
            framealpha=1.0,
            columnspacing=1.0,
            handlelength=1.7,
            borderpad=0.35,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), pad=0.2)
        output_path = self._path("3d")
        fig.savefig(output_path, dpi=self.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def save_thrust_plot(self, t_days: np.ndarray, u_norm_n: np.ndarray, max_thrust_n: float) -> Path:
        fig, ax = plt.subplots(figsize=self.SQUARE_DIAGNOSTIC_FIGSIZE, dpi=self.FIGURE_DPI)
        ax.plot(t_days, u_norm_n, color=self.TRAJECTORY_COLOR, lw=self.TRAJECTORY_LINE_WIDTH, label="_nolegend_")
        ax.axhline(max_thrust_n, color=self.RED, linestyle=":", lw=self.GUIDE_LINE_WIDTH, label="Maximum thrust")
        ax.set_xlabel("time [days]")
        ax.set_ylabel("thrust [N]")
        self._style_2d_axis(ax, tick_size=self.DIAGNOSTIC_TICK_SIZE, label_size=self.DIAGNOSTIC_LABEL_SIZE)
        self._legend(ax, loc="best", frameon=True, fancybox=False, edgecolor=self.BLACK, facecolor="white", framealpha=1.0)
        fig.tight_layout()
        output_path = self._path("thrust")
        fig.savefig(output_path, dpi=self.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def save_moon_distance_plot(self, t_dense_days: np.ndarray, moon_distance_km: np.ndarray) -> Path:
        fig, ax = plt.subplots(figsize=self.SQUARE_DIAGNOSTIC_FIGSIZE, dpi=self.FIGURE_DPI)
        ax.plot(t_dense_days, moon_distance_km, color=self.BLUE, lw=self.TRAJECTORY_LINE_WIDTH, label="_nolegend_")
        ax.axhline(0.0, color=self.RED, linestyle=":", lw=self.GUIDE_LINE_WIDTH, label="_nolegend_")
        ax.set_xlabel("time [days]")
        ax.set_ylabel("distance to Moon surface [km]")
        self._style_2d_axis(ax, tick_size=self.DIAGNOSTIC_TICK_SIZE, label_size=self.DIAGNOSTIC_LABEL_SIZE)
        fig.tight_layout()
        output_path = self._path("moon_distance")
        fig.savefig(output_path, dpi=self.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def save_verification_errors(
        self,
        t_dense_days: np.ndarray,
        position_error_m: np.ndarray,
        velocity_error_m_per_s: np.ndarray,
        mass_consumption_abs_difference_kg: np.ndarray,
    ) -> Path:
        specs = [
            ("position", "position error [m]", np.maximum(position_error_m, 1e-16), self.BLACK),
            ("velocity", "velocity error [m/s]", np.maximum(velocity_error_m_per_s, 1e-16), self.BLUE),
            ("mass", "mass-consumption error [kg]", np.maximum(mass_consumption_abs_difference_kg, 1e-16), self.RED),
        ]
        fig, axes = plt.subplots(1, 3, figsize=self.THREE_PANEL_FIGSIZE, dpi=self.FIGURE_DPI)
        for ax, (_, ylabel, values, color) in zip(axes, specs):
            max_error = float(np.max(values))
            ax.semilogy(t_dense_days, values, color=color, lw=self.TRAJECTORY_LINE_WIDTH, label=f"max = {max_error:.3e}")
            ax.set_xlabel("time [days]")
            ax.set_ylabel(ylabel)
            self._style_2d_axis(
                ax,
                grid=True,
                grid_which="both",
                tick_size=self.VERIFICATION_TICK_SIZE,
                label_size=self.VERIFICATION_LABEL_SIZE,
            )
            self._legend(
                ax,
                loc="lower right",
                fontsize=7.2,
                frameon=True,
                fancybox=False,
                edgecolor=self.BLACK,
                facecolor="white",
                framealpha=1.0,
                borderpad=0.28,
                handlelength=1.2,
            )
        fig.tight_layout()
        output_path = self._path("verification_errors")
        fig.savefig(output_path, dpi=self.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def save_defect_plot(
        self,
        mesh: np.ndarray,
        tof_days: float,
        scaled_defects: np.ndarray,
        defect_tolerance: float,
    ) -> Path:
        centers = 0.5 * (mesh[:-1] + mesh[1:]) * tof_days
        scaled = np.maximum(scaled_defects, 1e-16)
        fig, ax = plt.subplots(figsize=self.SQUARE_DIAGNOSTIC_FIGSIZE, dpi=self.FIGURE_DPI)
        ax.semilogy(
            centers,
            scaled,
            linestyle="none",
            marker="o",
            markersize=2.2,
            markerfacecolor=self.BLACK,
            markeredgecolor=self.BLACK,
            label="_nolegend_",
        )
        ax.axhline(defect_tolerance, color=self.RED, linestyle=":", lw=self.GUIDE_LINE_WIDTH, label="Tolerance")
        ax.set_xlabel("time [days]")
        ax.set_ylabel("scaled endpoint defect")
        self._style_2d_axis(ax, grid=False, tick_size=self.DIAGNOSTIC_TICK_SIZE, label_size=self.DIAGNOSTIC_LABEL_SIZE)
        self._legend(ax, loc="best", frameon=True, fancybox=False, edgecolor=self.BLACK, facecolor="white", framealpha=1.0)
        fig.tight_layout()
        output_path = self._path("defect")
        fig.savefig(output_path, dpi=self.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path
