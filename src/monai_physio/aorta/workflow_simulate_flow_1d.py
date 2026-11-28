"""Workflow for 1D rigid-wall blood flow along the centerline of an aorta.

The workflow prepares an aorta labelmap with :class:`AortaProcessGeometry`
(or takes a prepared geometry) and evaluates the 1D rigid-tube model of
``experiments/BloodFlow-1D-Rigid-Warp`` along its centerline.  In a rigid
tube incompressible flow is fixed by the inflow, so no equation is solved in
time: the section-mean velocity is ``Q(s, t) / A(s)`` and the pressure is

    p(s, t) = P_out(t) + dQ_in/dt int_s^L rho f / A ds
                       + Q_in int_s^L 8 pi mu f / A^2 ds,

where ``s`` is the arc length from the root (0) to the distal end (L),
``Q(s, t) = f(s) Q_in(t)`` and ``f(s)`` is the fraction of the inflow still
in the aorta at ``s``: one at the root, dropping at each branch port's arc
length by that outlet's share of the flow.  The shares are the
:attr:`AortaFlowPhysiology.flow_fractions` renormalized over the outlets
present, and each outlet ends in the Windkessel
:meth:`AortaFlowPhysiology.windkessels` builds for that share.  Those
Windkessels all scale with their share, so every outlet carries the same
pressure ``P_out``, that of one whole-body Windkessel fed by ``Q_in`` (the
model of the experiment); the imposed split neglects the few-mmHg trunk
pressure drop between the branches and the distal end.  Convective
acceleration and wall motion are not modelled.  The integrals are evaluated
by a float64 Warp kernel with the trapezoidal rule.

Typical usage::

    import itk
    from monai_physio.aorta import AortaFlowPhysiology, AortaWorkflowSimulateFlow1D

    # The patient's measured blood pressure, an input of the simulation.
    physiology = AortaFlowPhysiology(
        systolic_pressure_mmhg=124.0, diastolic_pressure_mmhg=82.0
    )
    # The generator reads one frame at a time: a long series of labelmaps
    # may not fit in memory.
    lv_volume_ml = AortaFlowPhysiology.lv_volumes_ml(
        (itk.imread(f) for f in frame_files), [151]
    )
    inflow, interval_s = physiology.lv_inflow(lv_volume_ml)
    workflow = AortaWorkflowSimulateFlow1D(physiology)
    result = workflow.process(
        itk.imread("labelmap.mha"),
        aorta_label_ids=(6,),
        branch_label_ids={"BCT": 109, "LCCA": 112, "LSA": 123},
        inflow_m3_s=inflow,
        inflow_interval_s=interval_s,
    )
    AortaWorkflowSimulateFlow1D.save_results(result, "./out")
    AortaWorkflowSimulateFlow1D.save_plots(result, "./out")
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import itk
import numpy as np
import warp as wp

from .flow_physiology import AortaFlowPhysiology
from .workflow_simulate_flow_base import AortaWorkflowSimulateFlowBase


@wp.kernel
def _trunk_fields(
    arc: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
    area: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
    fraction: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
    flow: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
    derivative: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
    density: wp.float64,
    drag: wp.float64,
    velocity: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
    drop: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
) -> None:
    """Mean velocity and the pressure drop from a section to the distal end."""
    t, x = wp.tid()  # type: ignore[misc]
    inertance = wp.float64(0.0)
    resistance = wp.float64(0.0)
    for j in range(x, arc.shape[0] - 1):
        length = arc[j + 1] - arc[j]
        a = fraction[j] / area[j]
        b = fraction[j + 1] / area[j + 1]
        inertance += density * length * (a + b) / wp.float64(2.0)
        resistance += drag * length * (a / area[j] + b / area[j + 1]) / wp.float64(2.0)
    velocity[t, x] = flow[t] * fraction[x] / area[x]
    drop[t, x] = inertance * derivative[t] + resistance * flow[t]


class AortaWorkflowSimulateFlow1D(AortaWorkflowSimulateFlowBase):
    """Simulate rigid-wall 1D blood flow along an aorta's centerline.

    The inputs, outlet calibration, beats and I/O contract are those of
    :class:`AortaWorkflowSimulateFlowBase`.  The flow split is imposed: each
    outlet takes its renormalized :attr:`AortaFlowPhysiology.flow_fractions`
    share of the inflow, and every outlet, branch or distal, carries the
    distal outlet's Windkessel pressure (see the module docstring).
    """

    # ─────────────────────────── Internal helpers ──────────────────────────

    def _evaluate(
        self,
        arc_m: np.ndarray,
        area_m2: np.ndarray,
        fraction: np.ndarray,
        flow: np.ndarray,
        derivative: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the mean velocity and pressure drop (time x section)."""
        wp.init()
        inputs: list[Any] = [
            wp.array(np.ascontiguousarray(values, dtype=np.float64), dtype=wp.float64)
            for values in (arc_m, area_m2, fraction, flow, derivative)
        ]
        velocity = wp.empty((len(flow), len(arc_m)), dtype=wp.float64)
        drop = wp.empty_like(velocity)
        # Formed in Python: Warp would round a float literal to float32.
        drag = wp.float64(8.0 * math.pi * self.physiology.viscosity_pa_s)
        density = wp.float64(self.physiology.density_kg_m3)
        wp.launch(
            _trunk_fields,
            dim=velocity.shape,
            inputs=[*inputs, density, drag, velocity, drop],
        )
        self.log_info("Evaluated the trunk fields on %s", wp.get_device().name)
        return velocity.numpy(), drop.numpy()

    # ─────────────────────────── Main workflow ─────────────────────────────

    def process(
        self,
        labelmap_image: Optional[itk.Image] = None,
        aorta_label_ids: Sequence[int] = (1,),
        branch_label_ids: Optional[Mapping[str, int]] = None,
        *,
        inflow_m3_s: np.ndarray,
        inflow_interval_s: float,
        geometry: Optional[Mapping[str, Any]] = None,
        beats: int = 3,
        dt_s: float = 0.003,
        calibrate_outlets: bool = True,
    ) -> dict[str, Any]:
        """Simulate 1D flow in an aorta labelmap or a prepared geometry.

        Args:
            labelmap_image: Labelmap holding the aorta and, optionally, its
                branches; prepared by :meth:`AortaProcessGeometry.prepare`
                with its defaults.  Give this or *geometry*.
            aorta_label_ids: Labels whose union is the aortic lumen.
            branch_label_ids: Branch name -> label.  Each branch found becomes
                an outlet named by its lowercased key, which must be a key of
                :attr:`AortaFlowPhysiology.flow_fractions`.  ``None`` gives a
                tube whose whole inflow leaves at the distal end.
            inflow_m3_s: Mean inflow of each interval of a periodic series,
                for example from :meth:`AortaFlowPhysiology.lv_inflow`.
            inflow_interval_s: Duration of each inflow interval in seconds.
            geometry: A :meth:`AortaProcessGeometry.prepare` result, used in
                place of preparing *labelmap_image*.  Only its
                ``"centerline"`` (point data ``arc_length_mm`` and
                ``area_mm2``), its ``"ports"`` (``role`` and
                ``arc_length_mm``) and, for :meth:`save_results`, its
                ``"surface"`` are read.
            beats: Number of heart beats to simulate, at least two.
            dt_s: Time step of the outlet Windkessels and the saved fields.
            calibrate_outlets: Calibrate the outlet totals and initial
                pressure to the input pressures of :attr:`physiology` and
                this inflow (:meth:`AortaFlowPhysiology.calibrated`).
                ``False`` uses its outlet totals as given.

        Returns:
            ``dict`` with the following keys:

            - ``'geometry'`` - the prepared geometry.
            - ``'physiology'`` - the :class:`AortaFlowPhysiology` simulated:
              :attr:`physiology`, calibrated unless *calibrate_outlets* is
              ``False``.
            - ``'sections'`` - (N,) ``'arc_length_m'``, ``'area_m2'`` and
              ``'flow_fraction'``: the arc length, lumen area and fraction
              of the inflow of each centerline section.
            - ``'series'`` - per time sample: (T,) ``'times_s'`` (the middle
              of each step), ``'inlet_flow_m3_s'`` and
              ``'inlet_flow_derivative_m3_s2'`` (the smoothed inflow and its
              time derivative) and ``'root_pressure_pa'``; outlet name ->
              (T,) ``'outlet_flow_m3_s'`` and ``'outlet_pressure_pa'``
              (Windkessel pressure); and (T, N) ``'flow_m3_s'``,
              ``'velocity_m_s'`` (section mean) and ``'pressure_pa'``.
            - ``'summary'`` - the keys of
              :meth:`AortaWorkflowSimulateFlowBase._pressure_summary`, then
              each outlet's flow share ``outlet_flow_shares`` and stroke
              volume ``outlet_stroke_volume_ml`` (mL), the peak section-mean
              velocity ``peak_velocity_m_s`` (m/s) and the peak Reynolds
              number ``rho V D / mu`` of the equivalent circular section
              ``peak_reynolds``, all over the last beat.

        Raises:
            ValueError: If not exactly one of *labelmap_image* and *geometry*
                is given, if *beats* is below two, if the inflow has no
                ejection, or if an outlet has no flow fraction.
        """
        self.log_section("STARTING 1D AORTA FLOW WORKFLOW")
        # Progress: preparing the geometry takes nearly all the time.
        self._start_progress("1D aorta flow starting")
        if beats < 2:
            raise ValueError(f"beats must be at least 2, not {beats}")
        geometry = self._resolve_geometry(
            labelmap_image, geometry, aorta_label_ids, branch_label_ids
        )
        self._report_progress(0.9, "geometry prepared")
        inflow = np.asarray(inflow_m3_s, dtype=np.float64)
        physiology = self._calibrate(inflow, inflow_interval_s, dt_s, calibrate_outlets)

        centerline = geometry["centerline"]
        arc_m = np.asarray(centerline.point_data["arc_length_mm"]) * 1e-3
        area_m2 = np.asarray(centerline.point_data["area_mm2"]) * 1e-6
        ports = geometry["ports"]
        outlets = [name for name, port in ports.items() if port["role"] == "outlet"]
        windkessels = physiology.windkessels(outlets)
        total = sum(physiology.flow_fractions[name] for name in outlets)
        shares = {name: physiology.flow_fractions[name] / total for name in outlets}
        fraction = np.ones(len(arc_m))
        for name in outlets:
            if name != "distal":
                fraction[arc_m > ports[name]["arc_length_mm"] * 1e-3] -= shares[name]
        self.log_info(
            "Centerline: %d sections over %.1f mm; outlet shares %s",
            len(arc_m),
            arc_m[-1] * 1e3,
            {name: round(share, 3) for name, share in shares.items()},
        )

        beat_s = 60.0 / physiology.heart_rate_bpm
        times = (np.arange(round(beats * beat_s / dt_s)) + 0.5) * dt_s
        flow, derivative = physiology.inlet_flow(
            times + AortaFlowPhysiology.beat_start_s(inflow, inflow_interval_s),
            inflow,
            inflow_interval_s,
        )
        velocity, drop = self._evaluate(arc_m, area_m2, fraction, flow, derivative)
        outlet_flow = {name: shares[name] * flow for name in outlets}
        outlet_pressure = {
            name: np.asarray(
                [windkessels[name].advance(q, dt_s) for q in outlet_flow[name]]
            )
            for name in outlets
        }
        pressure = drop + outlet_pressure["distal"][:, None]

        result: dict[str, Any] = {
            "geometry": geometry,
            "physiology": physiology,
            "sections": {
                "arc_length_m": arc_m,
                "area_m2": area_m2,
                "flow_fraction": fraction,
            },
            "series": {
                "times_s": times,
                "inlet_flow_m3_s": flow,
                "inlet_flow_derivative_m3_s2": derivative,
                "root_pressure_pa": pressure[:, 0],
                "outlet_flow_m3_s": outlet_flow,
                "outlet_pressure_pa": outlet_pressure,
                "flow_m3_s": flow[:, None] * fraction[None, :],
                "velocity_m_s": velocity,
                "pressure_pa": pressure,
            },
        }
        result["summary"] = self._summarize(result, shares)

        self.log_section("1D AORTA FLOW WORKFLOW COMPLETE")
        for key, value in result["summary"].items():
            self.log_info("%-32s %s", key, value)
        self._report_progress(1.0, "1D aorta flow complete")
        return result

    def _summarize(
        self, result: Mapping[str, Any], shares: Mapping[str, float]
    ) -> dict[str, Any]:
        """Return the metrics of the last simulated beat (see :meth:`process`)."""
        physiology = result["physiology"]
        series = result["series"]
        summary = self._pressure_summary(physiology, series)
        stroke_ml = summary["stroke_volume_ml"]
        last = self._last_beat(series["times_s"], 60.0 / physiology.heart_rate_bpm)
        velocity = np.asarray(series["velocity_m_s"])[last]
        diameter = 2.0 * np.sqrt(np.asarray(result["sections"]["area_m2"]) / np.pi)
        reynolds = (
            physiology.density_kg_m3
            * np.abs(velocity)
            * diameter
            / physiology.viscosity_pa_s
        )
        summary.update(
            {
                "outlet_flow_shares": dict(shares),
                "outlet_stroke_volume_ml": {
                    name: share * stroke_ml for name, share in shares.items()
                },
                "peak_velocity_m_s": float(np.abs(velocity).max()),
                "peak_reynolds": float(reynolds.max()),
            }
        )
        return summary

    # ─────────────────────────── I/O helpers ───────────────────────────────

    @classmethod
    def save_results(cls, result: Mapping[str, Any], output_dir: str) -> dict[str, str]:
        """Write the centerline, lumen surface, fields and summary.

        Files written to *output_dir* (created if absent):

        - ``centerline.vtp`` - the centerline with last-beat point data
          ``systolic_pressure_mmhg``, ``diastolic_pressure_mmhg``,
          ``mean_pressure_mmhg``, ``peak_velocity_m_s`` and
          ``flow_fraction``.
        - ``lumen_surface.vtp`` - the tagged lumen surface of the geometry.
        - ``fields.npz`` - every array of ``result["sections"]`` and
          ``result["series"]``, the outlet arrays as
          ``outlet_flow_m3_s_<name>`` and ``outlet_pressure_pa_<name>``.
        - ``summary.json`` - ``result["summary"]``.

        Args:
            result: The result of :meth:`process`.
            output_dir: Directory to write into.

        Returns:
            File kind (``"centerline"``, ``"surface"``, ``"fields"``,
            ``"summary"``) -> path of the saved file.
        """
        os.makedirs(output_dir, exist_ok=True)
        series = result["series"]
        beat_s = 60.0 / float(result["summary"]["heart_rate_bpm"])
        last = cls._last_beat(series["times_s"], beat_s)
        pressure = np.asarray(series["pressure_pa"])[last] / AortaFlowPhysiology.MMHG_PA
        centerline = result["geometry"]["centerline"].copy()
        centerline.point_data["systolic_pressure_mmhg"] = pressure.max(axis=0)
        centerline.point_data["diastolic_pressure_mmhg"] = pressure.min(axis=0)
        centerline.point_data["mean_pressure_mmhg"] = pressure.mean(axis=0)
        centerline.point_data["peak_velocity_m_s"] = np.abs(
            np.asarray(series["velocity_m_s"])[last]
        ).max(axis=0)
        centerline.point_data["flow_fraction"] = np.asarray(
            result["sections"]["flow_fraction"]
        )

        paths = {
            "centerline": os.path.join(output_dir, "centerline.vtp"),
            "surface": os.path.join(output_dir, "lumen_surface.vtp"),
            "fields": os.path.join(output_dir, "fields.npz"),
            "summary": os.path.join(output_dir, "summary.json"),
        }
        centerline.save(paths["centerline"])
        result["geometry"]["surface"].save(paths["surface"])
        np.savez_compressed(
            paths["fields"], **cls._array_items({**result["sections"], **series})
        )
        with open(paths["summary"], "w", encoding="utf-8") as file:
            json.dump(result["summary"], file, indent=2)
        return paths
