"""Base class of the aortic blood-flow workflows.

:class:`AortaWorkflowSimulateFlowBase` holds what the 1D, 3D rigid and 3D
fluid-structure workflows share: the patient physiology, the geometry from a
labelmap or a prepared geometry, the calibration of the outlets to the input
pressures, the selection of the last beat, the root-pressure summary and the
pressure-and-flow plot.

Every workflow result holds ``"geometry"``, ``"physiology"``, ``"series"``
and ``"summary"``.  ``"series"`` holds one value per simulated sample:
``"times_s"``, ``"inlet_flow_m3_s"``, ``"root_pressure_pa"`` and the outlet
name -> array dicts ``"outlet_flow_m3_s"`` and ``"outlet_pressure_pa"``, plus
each model's own series.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Optional

import itk
import numpy as np

from ..monai_physio_base import MONAIPhysioBase
from .flow_physiology import AortaFlowPhysiology
from .process_geometry import AortaProcessGeometry


class AortaWorkflowSimulateFlowBase(MONAIPhysioBase):
    """Shared inputs, calibration, summary and plots of the aorta flow workflows.

    ``physiology`` supplies the patient's input blood pressure and heart
    rate, the blood properties, the inflow smoothing and the outlet
    Windkessels; see :class:`AortaFlowPhysiology`.

    **Input pressures**

    The patient's systolic and diastolic pressures are inputs, not results.
    By default ``process`` calibrates a copy of the outlets to them and to
    the given inflow (:meth:`AortaFlowPhysiology.calibrated`), leaving
    :attr:`physiology` unchanged; the simulated root pressure then differs
    from the inputs by the model's pressure drop along the aorta.
    ``calibrate_outlets=False`` uses the outlet totals of ``physiology`` as
    given.

    **Inflow**

    Any sampled periodic inflow works: ``inflow_m3_s`` holds the mean flow
    of each interval of ``inflow_interval_s``, for example from
    :meth:`AortaFlowPhysiology.lv_inflow` or a measured flow curve, and must
    hold at least one ejection (a run of positive flow).  It is smoothed by
    :meth:`AortaFlowPhysiology.inlet_flow`.

    **Beats**

    The simulation covers ``beats`` heart beats of ``60 / heart_rate_bpm``
    seconds.  Time 0 is the middle of the filling gap before the first
    ejection of the inflow series, so every beat window holds one whole
    ejection.  The summary reports the last beat; the outlet capacitors start
    at :attr:`AortaFlowPhysiology.initial_pressure_pa` (the calibrated
    periodic pressure by default), and the change of mean root pressure from
    the beat before shows how periodic the last beat is.

    **I/O contract**

    ``process`` performs *no* file I/O.  Use ``save_results`` and
    :meth:`save_plots` to write results to disk.
    """

    #: Whether :meth:`save_plots` adds a third panel drawn by
    #: :meth:`_extra_plot_panel`.
    EXTRA_PLOT_PANEL: ClassVar[bool] = False

    def __init__(
        self,
        physiology: Optional[AortaFlowPhysiology] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        """Initialize the workflow.

        Args:
            physiology: Blood, inflow, outlet and wall parameters.  Defaults
                to :class:`AortaFlowPhysiology` ``()``.
            log_level: Logging level.  Default: ``logging.INFO``.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self.physiology = (
            physiology if physiology is not None else AortaFlowPhysiology()
        )
        self._geometry_tools = AortaProcessGeometry(log_level=log_level)

    # ─────────────────────────── Shared steps ──────────────────────────────

    def _resolve_geometry(
        self,
        labelmap_image: Optional[itk.Image],
        geometry: Optional[Mapping[str, Any]],
        aorta_label_ids: Sequence[int],
        branch_label_ids: Optional[Mapping[str, int]],
        **prepare_options: Any,
    ) -> Mapping[str, Any]:
        """Return *geometry*, or prepare it from *labelmap_image*.

        Args:
            labelmap_image: Labelmap to prepare with
                :meth:`AortaProcessGeometry.prepare`.
            geometry: A prepared geometry, returned as given.
            aorta_label_ids: Labels whose union is the aortic lumen.
            branch_label_ids: Branch name -> label.
            prepare_options: Keyword arguments of
                :meth:`AortaProcessGeometry.prepare`.

        Raises:
            ValueError: If not exactly one of *labelmap_image* and *geometry*
                is given.
        """
        if (labelmap_image is None) == (geometry is None):
            raise ValueError("Give exactly one of labelmap_image and geometry")
        if geometry is not None:
            return geometry
        return self._geometry_tools.prepare(
            labelmap_image, aorta_label_ids, branch_label_ids, **prepare_options
        )

    def _calibrate(
        self,
        inflow_m3_s: np.ndarray,
        interval_s: float,
        dt_s: float,
        calibrate: bool,
        wall_compliance_m3_pa: float = 0.0,
    ) -> AortaFlowPhysiology:
        """Return :attr:`physiology`, calibrated to this inflow if *calibrate*.

        See :meth:`AortaFlowPhysiology.calibrated`; *wall_compliance_m3_pa*
        is the compliance of an elastic aorta in front of the outlets.
        """
        physiology = self.physiology
        if calibrate:
            physiology = physiology.calibrated(
                inflow_m3_s,
                interval_s,
                dt_s,
                wall_compliance_m3_pa=wall_compliance_m3_pa,
            )
        self.log_info(
            "Input patient pressure %.0f/%.0f mmHg (mean %.1f); outlets %s: "
            "R %.4g Pa s/m^3, C %.4g m^3/Pa, initial %.1f mmHg",
            physiology.systolic_pressure_mmhg,
            physiology.diastolic_pressure_mmhg,
            physiology.mean_pressure_mmhg,
            "calibrated to it" if calibrate else "as given",
            physiology.total_resistance_pa_s_m3,
            physiology.total_compliance_m3_pa,
            physiology.initial_pressure_pa / AortaFlowPhysiology.MMHG_PA,
        )
        if wall_compliance_m3_pa > 0:
            self.log_info(
                "Compliance total %.4g = wall %.4g + outlets %.4g m^3/Pa",
                physiology.total_compliance_m3_pa + wall_compliance_m3_pa,
                wall_compliance_m3_pa,
                physiology.total_compliance_m3_pa,
            )
        return physiology

    @staticmethod
    def _last_beat(times_s: np.ndarray, beat_s: float) -> np.ndarray:
        """Return which samples fall in the last *beat_s* of the run.

        The samples are evenly spaced, one per time step, so these are the
        last ``round(beat_s / step)`` of them.
        """
        times_s = np.asarray(times_s)
        count = round(beat_s / float(times_s[1] - times_s[0]))
        last = np.zeros(len(times_s), dtype=bool)
        last[max(0, len(times_s) - count) :] = True
        return last

    def _pressure_summary(
        self, physiology: AortaFlowPhysiology, series: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return the input pressures, outlet totals and last-beat root metrics.

        The keys are ``heart_rate_bpm``, the input patient pressures
        ``input_systolic_mmhg``, ``input_diastolic_mmhg`` and
        ``input_mean_mmhg`` (:attr:`AortaFlowPhysiology.mean_pressure_mmhg`),
        the outlet totals simulated ``outlet_total_resistance_pa_s_m3`` and
        ``outlet_total_compliance_m3_pa``; then, over the last beat, the root's
        ``root_systolic_mmhg``, ``root_diastolic_mmhg``, ``root_mean_mmhg``
        and ``root_pulse_pressure_mmhg``, the largest and smallest root minus
        distal-outlet pressure ``root_to_distal_max_mmhg`` and
        ``root_to_distal_min_mmhg``, the change of mean root pressure from
        the beat before ``mean_pressure_change_mmhg``, ``peak_inflow_ml_s``,
        ``stroke_volume_ml`` and ``cardiac_output_l_min``.
        """
        beat_s = 60.0 / physiology.heart_rate_bpm
        times = np.asarray(series["times_s"])
        dt_s = float(times[1] - times[0])
        last = self._last_beat(times, beat_s)
        previous = self._last_beat(times, 2 * beat_s) & ~last
        root = np.asarray(series["root_pressure_pa"]) / AortaFlowPhysiology.MMHG_PA
        distal = (
            np.asarray(series["outlet_pressure_pa"]["distal"])
            / AortaFlowPhysiology.MMHG_PA
        )
        difference = root[last] - distal[last]
        inflow = np.asarray(series["inlet_flow_m3_s"])
        stroke_ml = float(inflow[last].sum() * dt_s * 1e6)
        return {
            "heart_rate_bpm": physiology.heart_rate_bpm,
            "input_systolic_mmhg": physiology.systolic_pressure_mmhg,
            "input_diastolic_mmhg": physiology.diastolic_pressure_mmhg,
            "input_mean_mmhg": physiology.mean_pressure_mmhg,
            "outlet_total_resistance_pa_s_m3": physiology.total_resistance_pa_s_m3,
            "outlet_total_compliance_m3_pa": physiology.total_compliance_m3_pa,
            "root_systolic_mmhg": float(root[last].max()),
            "root_diastolic_mmhg": float(root[last].min()),
            "root_mean_mmhg": float(root[last].mean()),
            "root_pulse_pressure_mmhg": float(np.ptp(root[last])),
            "root_to_distal_max_mmhg": float(difference.max()),
            "root_to_distal_min_mmhg": float(difference.min()),
            "mean_pressure_change_mmhg": float(
                root[last].mean() - root[previous].mean()
            ),
            "peak_inflow_ml_s": float(inflow[last].max() * 1e6),
            "stroke_volume_ml": stroke_ml,
            "cardiac_output_l_min": stroke_ml * 60.0 / beat_s / 1000.0,
        }

    @staticmethod
    def _array_items(arrays: Mapping[str, Any]) -> dict[str, Any]:
        """Return *arrays* flat for ``np.savez``: ``key_name`` for nested dicts."""
        flat: dict[str, Any] = {}
        for key, value in arrays.items():
            if isinstance(value, Mapping):
                for name, values in value.items():
                    flat[f"{key}_{name}"] = np.asarray(values)
            else:
                flat[key] = np.asarray(value)
        return flat

    # ─────────────────────────── I/O helpers ───────────────────────────────

    @staticmethod
    def _extra_plot_panel(
        axis: Any, result: Mapping[str, Any], shown: np.ndarray
    ) -> None:
        """Draw the third panel of :meth:`save_plots` (see EXTRA_PLOT_PANEL)."""

    @classmethod
    def save_plots(cls, result: Mapping[str, Any], output_dir: str) -> str:
        """Plot the last two beats of pressure and flow to ``pressure_flow.png``.

        The top panel shows the root and distal-outlet pressures, the next
        one the inflow and each outlet's flow; with :attr:`EXTRA_PLOT_PANEL`
        a third shows the model's own series (the FSI workflow's wall).

        Args:
            result: The result of ``process``.
            output_dir: Directory to write into (created if absent).

        Returns:
            Path of the saved PNG.
        """
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        series = result["series"]
        times = np.asarray(series["times_s"])
        beat_s = 60.0 / float(result["summary"]["heart_rate_bpm"])
        shown = cls._last_beat(times, 2 * beat_s)
        root = (
            np.asarray(series["root_pressure_pa"])[shown] / AortaFlowPhysiology.MMHG_PA
        )
        distal = (
            np.asarray(series["outlet_pressure_pa"]["distal"])[shown]
            / AortaFlowPhysiology.MMHG_PA
        )
        path = os.path.join(output_dir, "pressure_flow.png")
        rows = 3 if cls.EXTRA_PLOT_PANEL else 2
        fig, axes = plt.subplots(rows, 1, figsize=(8.0, 2.0 + 2.0 * rows), sharex=True)
        try:
            top, middle = axes[0], axes[1]
            top.plot(times[shown], root, label="root")
            top.plot(times[shown], distal, label="distal", linestyle="--")
            top.set_ylabel("Pressure (mmHg)")
            top.legend(loc="upper right")
            middle.plot(
                times[shown],
                np.asarray(series["inlet_flow_m3_s"])[shown] * 1e6,
                label="inflow",
                color="black",
            )
            for name, values in series["outlet_flow_m3_s"].items():
                middle.plot(times[shown], np.asarray(values)[shown] * 1e6, label=name)
            middle.set_ylabel("Flow (mL/s)")
            middle.legend(loc="upper right")
            if cls.EXTRA_PLOT_PANEL:
                cls._extra_plot_panel(axes[2], result, shown)
            axes[-1].set_xlabel("Time (s)")
            fig.tight_layout()
            fig.savefig(path, dpi=100)
        finally:
            plt.close(fig)
        return path
