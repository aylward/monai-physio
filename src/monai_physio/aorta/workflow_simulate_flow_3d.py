"""Workflow for 3D rigid-wall blood flow in an aorta.

The workflow prepares an aorta labelmap with :class:`AortaProcessGeometry`
(or takes a prepared geometry), meshes its lumen alone with netgen
(:meth:`AortaProcessGeometry.rigid_mesh`) and solves time-dependent
incompressible Navier-Stokes flow in it with :class:`WarpNavierStokes`, the
model of ``experiments/BloodFlow-3D-Rigid-Warp``: quadratic velocity and
linear pressure (P2/P1) on Warp FEM, a blunt inlet profile carrying the given
inflow through the root, and at every outlet the three-element Windkessel
:meth:`AortaFlowPhysiology.windkessels` builds for that outlet's share, with
backflow stabilization.  Unlike the 1D workflow the flow split is computed,
not imposed: each outlet receives what its Windkessel and the lumen let
through.  The wall does not move.

The solver needs a CUDA GPU and NVIDIA cuDSS
(:func:`monai_physio.warp_navier_stokes.find_cudss_library`).

Typical usage::

    import itk
    from monai_physio.aorta import AortaFlowPhysiology, AortaWorkflowSimulateFlow3D

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
    workflow = AortaWorkflowSimulateFlow3D(physiology)
    result = workflow.process(
        itk.imread("labelmap.mha"),
        aorta_label_ids=(6,),
        branch_label_ids={"BCT": 109, "LCCA": 112, "LSA": 123},
        inflow_m3_s=inflow,
        inflow_interval_s=interval_s,
    )
    AortaWorkflowSimulateFlow3D.save_results(result, "./out")
    AortaWorkflowSimulateFlow3D.save_plots(result, "./out")
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Optional

import itk
import numpy as np
import warp as wp
from scipy.spatial import cKDTree

from ..warp_navier_stokes import WarpNavierStokes, tetrahedron_edges
from .flow_physiology import AortaFlowPhysiology
from .workflow_simulate_flow_base import AortaWorkflowSimulateFlowBase


class AortaWorkflowSimulateFlow3D(AortaWorkflowSimulateFlowBase):
    """Simulate rigid-wall 3D blood flow in an aorta.

    The inputs, outlet calibration (here at the solver time step), beats and
    I/O contract are those of :class:`AortaWorkflowSimulateFlowBase`; the
    simulated root pressure adds the 3D pressure drop along the aorta.  The
    run starts from rest, and full velocity and pressure fields are kept for
    the last beat.

    **Outlets**

    Each outlet ends in the object :meth:`AortaFlowPhysiology.windkessels`
    returns for it, a three-element :class:`Windkessel` by default.
    :class:`WarpNavierStokes` uses only its ``implicit(dt)``, which returns
    ``(R, P0)`` such that the outlet pressure over a step is ``R Q + P0``
    for the step's outflow ``Q`` (``R`` must not change between steps),
    ``advance(Q, dt)``, which ends the step and returns that pressure, and
    its ``pressure`` state, which a coupled step restores between
    iterations.  A subclass of :class:`AortaFlowPhysiology` whose
    ``windkessels`` returns other objects with these three members gives
    other outlet models.

    **Steppers**

    :meth:`process` steps the object :meth:`_build_stepper` returns: a
    :class:`WarpNavierStokes`, or an object wrapping one as ``.fluid`` (such
    as :class:`WarpFluidStructure`), whose ``step(inlet_flow)`` returns the
    step result of :meth:`WarpNavierStokes.step` plus its own entries.
    :attr:`STEP_KEYS` names the entries kept as series and
    :meth:`_frame_extras` adds fields to the saved frames.
    """

    #: Model name of the log and progress messages.
    TITLE: ClassVar[str] = "3D aorta flow"
    #: Backflow stabilization coefficient of the outlets.
    BACKFLOW_BETA = 1.0
    #: The inlet speed rises from the wall over this fraction of the root's
    #: equivalent radius and is flat inside it.
    INLET_BLUNT_FRACTION = 0.2
    #: Speeds within this distance of an outlet center are left out of
    #: ``peak_speed_away_from_outlets_m_s``: the short branch stumps end in
    #: small caps whose speeds are not mesh converged.
    OUTLET_EXCLUSION_MM = 10.0
    #: Keys a mesh given to :meth:`process` must hold.
    MESH_KEYS: ClassVar[tuple[str, ...]] = (
        "points_mm",
        "tetrahedra",
        "boundary_faces",
        "boundary_regions",
    )
    #: Series kept from every step: series key -> (step-result key, dtype).
    STEP_KEYS: ClassVar[dict[str, tuple[str, Any]]] = {
        "gmres_iterations": ("iterations", np.int64),
        "residual": ("residual", np.float64),
    }
    #: Steps between progress lines in the log.
    LOG_STEPS: ClassVar[int] = 100

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
        mesh: Optional[Mapping[str, Any]] = None,
        beats: int = 3,
        dt_s: float = 0.005,
        element_size_mm: float = 2.0,
        precision: str = "float32",
        frame_interval_s: float = 0.03,
        calibrate_outlets: bool = True,
    ) -> dict[str, Any]:
        """Simulate 3D flow in an aorta labelmap or a prepared geometry.

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
                place of preparing *labelmap_image*.
            mesh: A lumen mesh with the keys of
                :meth:`AortaProcessGeometry.rigid_mesh`, used in place of
                meshing the geometry: (n, 3) ``"points_mm"``, (m, 4)
                positively oriented ``"tetrahedra"``, (f, 3) outward
                ``"boundary_faces"`` and (f,) ``"boundary_regions"``, 0 on
                the wall and the port's ``label_id`` on each of the
                geometry's ports.  The geometry still supplies the ports.
            beats: Number of heart beats to simulate, at least two.
            dt_s: Solver time step.
            element_size_mm: Target edge length of the lumen mesh away from
                small ports (:meth:`AortaProcessGeometry.rigid_mesh`).
            precision: ``"float32"`` or ``"float64"`` operators and linear
                solves (:class:`WarpNavierStokes`).
            frame_interval_s: Interval of the saved velocity and pressure
                fields over the last beat; rounded to a multiple of *dt_s*.
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
            - ``'mesh'`` - the lumen mesh (millimetres).
            - ``'series'`` - per step, (T,) ``'times_s'`` (end of the step),
              ``'inlet_flow_m3_s'`` (the smoothed inflow imposed at the
              root), ``'root_flow_m3_s'`` (the flow the solution carries in
              through the root), ``'root_pressure_pa'`` (area-weighted mean
              over the root), ``'gmres_iterations'`` and ``'residual'`` (of
              the linear solve) and ``'step_seconds'`` (wall time); and
              outlet name -> (T,) ``'outlet_flow_m3_s'`` (outward) and
              ``'outlet_pressure_pa'`` (Windkessel pressure).
            - ``'frames'`` - the saved last-beat frames: (F,) ``'times_s'``
              and ``'inlet_flow_m3_s'``, (F, nodes, 3) ``'velocity_m_s'`` at
              the P2 nodes (the mesh points, then the midpoints of the
              sorted unique tetrahedron edges, :func:`tetrahedron_edges`)
              and (F, points) ``'pressure_pa'`` at the mesh points.
            - ``'run'`` - solver settings, sizes, device and timings.
            - ``'summary'`` - the keys of
              :meth:`AortaWorkflowSimulateFlowBase._pressure_summary`, with
              the 1D workflow's meanings, and each outlet's computed flow
              share ``outlet_flow_shares`` and stroke volume
              ``outlet_stroke_volume_ml``; then the peak speed anywhere
              ``peak_speed_m_s`` and more than :attr:`OUTLET_EXCLUSION_MM`
              from every outlet center ``peak_speed_away_from_outlets_m_s``
              and its 99.9th percentile ``speed_99_9_percentile_m_s`` (m/s,
              over the saved frames), the peak Reynolds number of the mean
              root velocity ``root_peak_reynolds``, the largest net flow
              through all ports relative to the peak inflow
              ``max_net_port_flow_relative``, and the linear solver's
              ``max_linear_residual``, ``mean_gmres_iterations`` and
              ``unconverged_steps`` over the whole run.

        Raises:
            ValueError: If not exactly one of *labelmap_image* and *geometry*
                is given, if *beats* is below two, if *frame_interval_s* is
                longer than a beat, if *mesh* lacks a key or a port, if the
                inflow has no ejection, or if an outlet has no flow fraction.
            RuntimeError: If a linear solve diverges.
        """
        return self._simulate(
            labelmap_image,
            aorta_label_ids,
            branch_label_ids,
            inflow_m3_s=inflow_m3_s,
            inflow_interval_s=inflow_interval_s,
            geometry=geometry,
            mesh=mesh,
            beats=beats,
            dt_s=dt_s,
            element_size_mm=element_size_mm,
            precision=precision,
            frame_interval_s=frame_interval_s,
            calibrate_outlets=calibrate_outlets,
        )

    def _simulate(
        self,
        labelmap_image: Optional[itk.Image],
        aorta_label_ids: Sequence[int],
        branch_label_ids: Optional[Mapping[str, int]],
        *,
        inflow_m3_s: np.ndarray,
        inflow_interval_s: float,
        geometry: Optional[Mapping[str, Any]],
        mesh: Optional[Mapping[str, Any]],
        beats: int,
        dt_s: float,
        element_size_mm: float,
        precision: str,
        frame_interval_s: float,
        calibrate_outlets: bool,
        prepare_options: Optional[Mapping[str, Any]] = None,
        mesh_options: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run :meth:`process`; the options go to ``prepare`` and :meth:`_mesh`."""
        self.log_section("STARTING %s WORKFLOW", self.TITLE.upper())
        # Progress: geometry 0-0.1, mesh 0.1-0.2, solver setup 0.2-0.25 and
        # the time steps the rest.
        self._start_progress(f"{self.TITLE} starting")
        if beats < 2:
            raise ValueError(f"beats must be at least 2, not {beats}")
        beat_s = 60.0 / self.physiology.heart_rate_bpm
        if round(frame_interval_s / dt_s) > round(beat_s / dt_s):
            raise ValueError(
                f"frame_interval_s {frame_interval_s} s exceeds the {beat_s:.3g} s "
                "beat, so no frame of the last beat would be saved"
            )
        geometry = self._resolve_geometry(
            labelmap_image,
            geometry,
            aorta_label_ids,
            branch_label_ids,
            **(prepare_options or {}),
        )
        self._report_progress(0.1, "geometry prepared")
        inflow_series = np.asarray(inflow_m3_s, dtype=np.float64)
        start_s = AortaFlowPhysiology.beat_start_s(inflow_series, inflow_interval_s)
        ports = geometry["ports"]
        outlets = [name for name, port in ports.items() if port["role"] == "outlet"]
        if mesh is None:
            mesh = self._mesh(geometry, element_size_mm, **(mesh_options or {}))
        else:
            self._check_mesh(mesh, ports)
        self._report_progress(0.2, "mesh ready")

        wp.init()
        started = time.perf_counter()
        stepper, physiology, run_extras = self._build_stepper(
            geometry,
            mesh,
            outlets,
            (inflow_series, inflow_interval_s),
            dt_s=dt_s,
            precision=precision,
            calibrate_outlets=calibrate_outlets,
        )
        setup_seconds = time.perf_counter() - started
        fluid: WarpNavierStokes = getattr(stepper, "fluid", stepper)
        try:
            with self._progress_range(0.25, 1.0):
                series, frames = self._run(
                    stepper,
                    fluid,
                    physiology,
                    ports,
                    outlets,
                    mesh,
                    inflow=(inflow_series, inflow_interval_s, start_s),
                    beats=beats,
                    frame_interval_s=frame_interval_s,
                )
        finally:
            stepper.close()

        result: dict[str, Any] = {
            "geometry": geometry,
            "physiology": physiology,
            "mesh": mesh,
            "series": series,
            "frames": frames,
            "run": {
                "dt_s": dt_s,
                "precision": precision,
                "gmres_tolerance": fluid.tolerance,
                "element_size_mm": element_size_mm,
                **run_extras,
                "steps": len(series["times_s"]),
                "frames": len(frames["times_s"]),
                "velocity_nodes": len(fluid.dof_points),
                "pressure_nodes": len(fluid.points),
                "tetrahedra": len(fluid.tetrahedra),
                "device": wp.get_device().name,
                "setup_seconds": setup_seconds,
                "stepping_seconds": float(np.sum(series["step_seconds"])),
            },
        }
        outlet_distance_mm = cKDTree(
            [ports[name]["center_mm"] for name in outlets]
        ).query(fluid.dof_points * 1e3)[0]
        near = np.asarray(outlet_distance_mm) < self.OUTLET_EXCLUSION_MM
        result["summary"] = self._summarize(result, near)

        self.log_section("%s WORKFLOW COMPLETE", self.TITLE.upper())
        for key, value in result["summary"].items():
            self.log_info("%-38s %s", key, value)
        self._report_progress(1.0, f"{self.TITLE} complete")
        return result

    # ─────────────────────────── Model hooks ───────────────────────────────

    def _mesh(
        self, geometry: Mapping[str, Any], element_size_mm: float, **options: Any
    ) -> dict[str, Any]:
        """Mesh the lumen (:meth:`AortaProcessGeometry.rigid_mesh`)."""
        return self._geometry_tools.rigid_mesh(geometry, element_size_mm, **options)

    def _check_mesh(
        self, mesh: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """Raise ValueError unless *mesh* has :attr:`MESH_KEYS` and every port."""
        missing = [key for key in self.MESH_KEYS if key not in mesh]
        if missing:
            raise ValueError(f"The mesh lacks {missing}; it needs {self.MESH_KEYS}")
        points = np.asarray(mesh["points_mm"])
        tetrahedra = np.asarray(mesh["tetrahedra"])
        faces = np.asarray(mesh["boundary_faces"])
        regions = np.asarray(mesh["boundary_regions"])
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or tetrahedra.ndim != 2
            or tetrahedra.shape[1] != 4
            or faces.ndim != 2
            or faces.shape[1] != 3
            or regions.shape != (len(faces),)
        ):
            raise ValueError(
                "The mesh needs (n, 3) points_mm, (m, 4) tetrahedra, (f, 3) "
                "boundary_faces and (f,) boundary_regions"
            )
        labels = {int(port["label_id"]) for port in ports.values()}
        unknown = sorted(set(np.unique(regions).tolist()) - labels - {0})
        absent = [
            name
            for name, port in ports.items()
            if not np.any(regions == port["label_id"])
        ]
        if unknown or absent:
            raise ValueError(
                f"The mesh's boundary regions do not match the geometry's ports: "
                f"regions {unknown} are no port and ports {absent} have no "
                "boundary triangle"
            )

    def _navier_stokes(
        self,
        mesh: Mapping[str, Any],
        ports: Mapping[str, Mapping[str, Any]],
        outlets: Sequence[str],
        physiology: AortaFlowPhysiology,
        dt_s: float,
        precision: str,
    ) -> WarpNavierStokes:
        """Return the blood solver on the lumen mesh with the outlets' Windkessels."""
        windkessels = physiology.windkessels(outlets)
        return WarpNavierStokes(
            np.asarray(mesh["points_mm"]) * 1e-3,
            mesh["tetrahedra"],
            mesh["boundary_faces"],
            mesh["boundary_regions"],
            inlet_region=ports["root"]["label_id"],
            outlets={ports[name]["label_id"]: windkessels[name] for name in outlets},
            dt=dt_s,
            density=physiology.density_kg_m3,
            viscosity=physiology.viscosity_pa_s,
            backflow=self.BACKFLOW_BETA,
            blunt_fraction=self.INLET_BLUNT_FRACTION,
            precision=precision,
        )

    def _build_stepper(
        self,
        geometry: Mapping[str, Any],
        mesh: Mapping[str, Any],
        outlets: Sequence[str],
        inflow: tuple[np.ndarray, float],
        *,
        dt_s: float,
        precision: str,
        calibrate_outlets: bool,
    ) -> tuple[Any, AortaFlowPhysiology, dict[str, Any]]:
        """Return the stepper, the physiology simulated and extra run entries.

        *inflow* is the interval inflow series and its interval.
        """
        physiology = self._calibrate(*inflow, dt_s, calibrate_outlets)
        solver = self._navier_stokes(
            mesh, geometry["ports"], outlets, physiology, dt_s, precision
        )
        return solver, physiology, {}

    def _frame_extras(self, stepper: Any) -> dict[str, np.ndarray]:
        """Return fields saved with each frame besides velocity and pressure."""
        return {}

    def _step_message(self, step: Mapping[str, Any]) -> str:
        """Return the solver statistics of a step for the log."""
        return f"GMRES={step['iterations']}"

    # ─────────────────────────── Time stepping ─────────────────────────────

    def _run(
        self,
        stepper: Any,
        fluid: WarpNavierStokes,
        physiology: AortaFlowPhysiology,
        ports: Mapping[str, Mapping[str, Any]],
        outlets: Sequence[str],
        mesh: Mapping[str, Any],
        *,
        inflow: tuple[np.ndarray, float, float],
        beats: int,
        frame_interval_s: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Step the solver; return the per-step series and last-beat frames."""
        dt_s = fluid.dt
        beat_s = 60.0 / physiology.heart_rate_bpm
        count = round(beats * beat_s / dt_s)
        times = np.arange(1, count + 1) * dt_s
        series_m3_s, interval_s, start_s = inflow
        flow = physiology.inlet_flow(times + start_s, series_m3_s, interval_s)[0]

        faces = mesh["boundary_faces"]
        regions = mesh["boundary_regions"]
        root_region = ports["root"]["label_id"]
        root = regions == root_region
        root_area = np.linalg.norm(fluid.area_vectors[root], axis=1)
        root_weights = np.zeros(len(fluid.points))
        np.add.at(root_weights, faces[root], root_area[:, None] / 3)
        root_weights /= root_weights.sum()

        stride = max(1, round(frame_interval_s / dt_s))
        last_beat = self._last_beat(times, beat_s)
        series: dict[str, Any] = {
            "times_s": times,
            "inlet_flow_m3_s": flow,
            "root_flow_m3_s": np.zeros(count),
            "root_pressure_pa": np.zeros(count),
            "outlet_flow_m3_s": {name: np.zeros(count) for name in outlets},
            "outlet_pressure_pa": {name: np.zeros(count) for name in outlets},
            **{
                key: np.zeros(count, dtype=kind)
                for key, (_, kind) in self.STEP_KEYS.items()
            },
            "step_seconds": np.zeros(count),
        }
        frame_index: list[int] = []
        fields: dict[str, list[np.ndarray]] = {}
        for index, inlet in enumerate(flow):
            tic = time.perf_counter()
            step = stepper.step(float(inlet))
            if not np.isfinite(step["residual"]):
                raise RuntimeError(f"Linear solve diverged at step {index}: {step}")
            pressure = fluid.pressure()
            series["step_seconds"][index] = time.perf_counter() - tic
            for key, (step_key, _) in self.STEP_KEYS.items():
                series[key][index] = step[step_key]
            series["root_flow_m3_s"][index] = -step["flows_m3_s"][root_region]
            series["root_pressure_pa"][index] = root_weights @ pressure
            for name in outlets:
                region = ports[name]["label_id"]
                series["outlet_flow_m3_s"][name][index] = step["flows_m3_s"][region]
                series["outlet_pressure_pa"][name][index] = step["pressures_pa"][region]
            if last_beat[index] and (index + 1) % stride == 0:
                frame_index.append(index)
                frame = {
                    "velocity_m_s": fluid.velocity(),
                    "pressure_pa": pressure,
                    **self._frame_extras(stepper),
                }
                for key, values in frame.items():
                    fields.setdefault(key, []).append(values)
            self.log_progress(index + 1, count, "Step", log=False)
            if (
                index % self.LOG_STEPS == 0
                or index == count - 1
                or not step.get("converged", True)
            ):
                self.log_info(
                    "t=%.4f s Q=%.1f mL/s root=%.1f mmHg %s %.3f s",
                    times[index],
                    inlet * 1e6,
                    series["root_pressure_pa"][index] / AortaFlowPhysiology.MMHG_PA,
                    self._step_message(step),
                    series["step_seconds"][index],
                )
        frames = {
            "times_s": times[frame_index],
            "inlet_flow_m3_s": flow[frame_index],
            **{key: np.stack(values) for key, values in fields.items()},
        }
        return series, frames

    # ─────────────────────────── Summary ───────────────────────────────────

    def _summarize(
        self, result: Mapping[str, Any], near_outlet: np.ndarray
    ) -> dict[str, Any]:
        """Return the metrics of the last simulated beat (see :meth:`process`)."""
        physiology = result["physiology"]
        series = result["series"]
        summary = self._pressure_summary(physiology, series)
        beat_s = 60.0 / physiology.heart_rate_bpm
        times = np.asarray(series["times_s"])
        dt_s = float(times[1] - times[0])
        last = self._last_beat(times, beat_s)
        inflow = np.asarray(series["inlet_flow_m3_s"])
        outlet_flow = {
            name: np.asarray(values)
            for name, values in series["outlet_flow_m3_s"].items()
        }
        net = np.asarray(series["root_flow_m3_s"]) - sum(outlet_flow.values())
        speed = np.linalg.norm(np.asarray(result["frames"]["velocity_m_s"]), axis=2)
        root_area_m2 = result["geometry"]["ports"]["root"]["area_mm2"] * 1e-6
        root_diameter_m = 2.0 * np.sqrt(root_area_m2 / np.pi)
        residual = np.asarray(series["residual"])
        summary.update(
            {
                "outlet_flow_shares": {
                    name: float(flow[last].sum() / inflow[last].sum())
                    for name, flow in outlet_flow.items()
                },
                "outlet_stroke_volume_ml": {
                    name: float(flow[last].sum() * dt_s * 1e6)
                    for name, flow in outlet_flow.items()
                },
                "peak_speed_m_s": float(speed.max()),
                "peak_speed_away_from_outlets_m_s": float(speed[:, ~near_outlet].max()),
                "speed_99_9_percentile_m_s": float(np.percentile(speed, 99.9)),
                "root_peak_reynolds": float(
                    physiology.density_kg_m3
                    * inflow[last].max()
                    / root_area_m2
                    * root_diameter_m
                    / physiology.viscosity_pa_s
                ),
                "max_net_port_flow_relative": float(
                    np.abs(net[last]).max() / inflow[last].max()
                ),
                "max_linear_residual": float(residual.max()),
                "mean_gmres_iterations": float(np.mean(series["gmres_iterations"])),
                "unconverged_steps": int(
                    np.sum(~(residual <= result["run"]["gmres_tolerance"]))
                ),
            }
        )
        return summary

    # ─────────────────────────── I/O helpers ───────────────────────────────

    @classmethod
    def save_results(cls, result: Mapping[str, Any], output_dir: str) -> dict[str, str]:
        """Write the fields for :class:`VisualizeFlowFields` and the summary.

        Files written to *output_dir* (created if absent):

        - ``mesh.npz`` - ``points_m``, ``tetrahedra``, ``edges`` (the P2 edge
          node order), ``boundary_faces`` and ``boundary_regions`` of the
          lumen mesh.
        - ``time.npz`` - the frames' ``frame_times_s`` and
          ``frame_inlet_flow_m3_s``.
        - ``<field>.npy`` - every other array of ``result["frames"]``:
          ``velocity_m_s.npy`` and ``pressure_pa.npy`` (and the FSI
          workflow's ``displacement_m.npy``).
        - ``steps.npz`` - every array of ``result["series"]``, the outlet
          arrays as ``outlet_flow_m3_s_<name>`` and
          ``outlet_pressure_pa_<name>``.
        - ``run.json`` - ``result["run"]``.
        - ``summary.json`` - ``result["summary"]``.

        Args:
            result: The result of :meth:`process`.
            output_dir: Directory to write into.

        Returns:
            File kind -> path of the saved file: ``"mesh"``, ``"time"``,
            each field's key (``"velocity_m_s"``, ``"pressure_pa"``, ...),
            ``"steps"``, ``"run"`` and ``"summary"``.
        """
        os.makedirs(output_dir, exist_ok=True)
        frames = result["frames"]
        fields = [key for key in frames if key not in ("times_s", "inlet_flow_m3_s")]
        paths = {
            "mesh": os.path.join(output_dir, "mesh.npz"),
            "time": os.path.join(output_dir, "time.npz"),
            **{key: os.path.join(output_dir, f"{key}.npy") for key in fields},
            "steps": os.path.join(output_dir, "steps.npz"),
            "run": os.path.join(output_dir, "run.json"),
            "summary": os.path.join(output_dir, "summary.json"),
        }
        mesh = result["mesh"]
        np.savez_compressed(
            paths["mesh"],
            points_m=np.asarray(mesh["points_mm"]) * 1e-3,
            tetrahedra=mesh["tetrahedra"],
            edges=tetrahedron_edges(np.asarray(mesh["tetrahedra"])),
            boundary_faces=mesh["boundary_faces"],
            boundary_regions=mesh["boundary_regions"],
        )
        np.savez_compressed(
            paths["time"],
            frame_times_s=np.asarray(frames["times_s"]),
            frame_inlet_flow_m3_s=np.asarray(frames["inlet_flow_m3_s"]),
        )
        for key in fields:
            np.save(paths[key], np.asarray(frames[key]))
        np.savez_compressed(paths["steps"], **cls._array_items(result["series"]))
        with open(paths["run"], "w", encoding="utf-8") as file:
            json.dump(result["run"], file, indent=2)
        with open(paths["summary"], "w", encoding="utf-8") as file:
            json.dump(result["summary"], file, indent=2)
        return paths
