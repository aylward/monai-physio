"""Workflow for 3D blood flow in an aorta with a moving elastic wall (FSI).

The workflow prepares an aorta labelmap with :class:`AortaProcessGeometry`
(or takes a prepared geometry), meshes the lumen and a wall shell around it
together with netgen (:meth:`AortaProcessGeometry.fsi_mesh`) and couples the
flow model of :class:`AortaWorkflowSimulateFlow3D` to a quasi-static
Neo-Hookean wall with :class:`WarpFluidStructure`, the model of
``experiments/BloodFlow-3D-FSI-Simpleware-Warp``: the wall is prestressed so
that the imaged lumen is its shape at the outlets' initial pressure,
clamped in a collar around every port and tethered on its outer surface,
and every time step iterates the blood and the wall to agreement with an
interface quasi-Newton method.

The solver needs a CUDA GPU and NVIDIA cuDSS
(:func:`monai_physio.warp_navier_stokes.find_cudss_library`).

Typical usage::

    import itk
    from monai_physio.aorta import AortaFlowPhysiology, AortaWorkflowSimulateFlowFSI

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
    workflow = AortaWorkflowSimulateFlowFSI(physiology)
    result = workflow.process(
        itk.imread("labelmap.mha"),
        aorta_label_ids=(6,),
        branch_label_ids={"BCT": 109, "LCCA": 112, "LSA": 123},
        inflow_m3_s=inflow,
        inflow_interval_s=interval_s,
    )
    AortaWorkflowSimulateFlowFSI.save_results(result, "./out")
    AortaWorkflowSimulateFlowFSI.save_plots(result, "./out")
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Optional

import itk
import numpy as np
from scipy.spatial import cKDTree

from ..process_volumetric_mesh import ProcessVolumetricMesh
from ..warp_fluid_structure import WarpFluidStructure, WarpHyperelasticWall
from ..warp_navier_stokes import WarpNavierStokes
from .flow_physiology import AortaFlowPhysiology
from .workflow_simulate_flow_3d import AortaWorkflowSimulateFlow3D


class AortaWorkflowSimulateFlowFSI(AortaWorkflowSimulateFlow3D):
    """Simulate 3D blood flow in an aorta whose wall moves with the blood.

    The blood model, its inflow, outlets and summary are those of
    :class:`AortaWorkflowSimulateFlow3D`; this class adds the wall.
    ``physiology`` also supplies the wall's Young's modulus, Poisson ratio,
    thickness and tether stiffness.

    **Wall**

    A compressible Neo-Hookean solid of the given thickness around the
    lumen, quasi-static (massless), prestressed so that the imaged geometry
    is its equilibrium at :attr:`AortaFlowPhysiology.initial_pressure_pa`,
    the pressure where the simulation starts (with calibrated outlets, the
    calibrated periodic pressure at the start of the beats), so the run
    starts at rest in equilibrium; clamped within ``collar_mm`` of every
    port and held by springs on its outer surface
    (:class:`WarpHyperelasticWall`).

    Netgen fills a wall shell reliably only when it is at least
    :attr:`MIN_WALL_CELLS` grid spacings thick, so a thinner wall is
    prepared that thick and its Young's modulus scaled by the physiological
    over the meshed thickness.  Keeping modulus times thickness is a
    thin-wall approximation: the Lame thick-wall solution makes the default
    1.5 mm, 1 MPa wall meshed 3.75 mm thick at 0.4 MPa about 20-35% more
    compliant for inner radii of 12-7.5 mm.  The reported wall motion,
    strain, distensibility and wave speed describe that surrogate wall.  The
    run records the thickness and modulus meshed.

    **Input pressures and wall compliance**

    As in the rigid workflows the patient's systolic and diastolic pressures
    are inputs, but here the wall stores part of each stroke itself and
    smooths the root pressure before the outlets see it.  :meth:`process`
    therefore first measures the wall's compliance ``dV/dP`` with a
    quasi-static solve under uniform pressure, prestressed at the diastolic
    input and loaded with the systolic one, and then calibrates the outlets
    with that compliance in front of them
    (:meth:`AortaFlowPhysiology.calibrated` with
    ``wall_compliance_m3_pa``).  The total compliance reported is the
    outlets' plus the wall's.  The lumen is somewhat more compliant in the
    flow than in this static, uniform-pressure estimate (compare the
    summary's ``lumen_compliance_m3_pa``), so the simulated pulse and
    systolic pressure come out a little low.  ``calibrate_outlets=False``
    uses the outlet totals of ``physiology`` as given.
    """

    TITLE: ClassVar[str] = "3D FSI aorta flow"
    #: Interface quasi-Newton settings of :class:`WarpFluidStructure`, those
    #: of ``experiments/BloodFlow-3D-FSI-Simpleware-Warp`` except a looser
    #: absolute tolerance, since float32 blood solves hold the interface
    #: residual above the experiment's 1 nm.
    COUPLING: ClassVar[dict[str, Any]] = {
        "relaxation": 0.1,
        "reuse": 2,
        "tolerance": 1e-3,
        "absolute_tolerance": 1e-8,
        "max_iterations": 20,
    }
    #: Fewest grid spacings across the wall shell that netgen fills reliably.
    MIN_WALL_CELLS = 3.0
    MESH_KEYS: ClassVar[tuple[str, ...]] = (
        *AortaWorkflowSimulateFlow3D.MESH_KEYS,
        "wall_points_mm",
        "wall_tetrahedra",
        "wall_clamped",
        "wall_tether_faces",
    )
    #: ``gmres_iterations`` and ``residual`` are those of the step's last
    #: blood solve; ``total_gmres_iterations`` sums every blood solve.
    STEP_KEYS: ClassVar[dict[str, tuple[str, Any]]] = {
        **AortaWorkflowSimulateFlow3D.STEP_KEYS,
        "total_gmres_iterations": ("gmres_iterations", np.int64),
        "converged": ("converged", bool),
        **{
            key: (key, np.int64)
            for key in (
                "coupling_iterations",
                "wall_iterations",
                "wall_tangent_updates",
            )
        },
        **{
            key: (key, np.float64)
            for key in (
                "coupling_residual",
                "coupling_residual_m",
                "max_displacement_m",
                "mean_displacement_m",
                "volume_m3",
                "volume_rate_m3_s",
            )
        },
    }
    LOG_STEPS: ClassVar[int] = 20
    EXTRA_PLOT_PANEL: ClassVar[bool] = True

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
        dt_s: float = 0.0025,
        element_size_mm: float = 2.0,
        precision: str = "float32",
        frame_interval_s: float = 0.03,
        calibrate_outlets: bool = True,
        spacing_mm: float = 1.25,
        collar_mm: float = 2.0,
    ) -> dict[str, Any]:
        """Simulate coupled flow and wall motion in an aorta.

        Args:
            labelmap_image: Labelmap holding the aorta and, optionally, its
                branches; prepared by :meth:`AortaProcessGeometry.prepare` at
                *spacing_mm* with the wall thickness to mesh (see *Wall*
                above).  Give this or *geometry*.
            aorta_label_ids: Labels whose union is the aortic lumen.
            branch_label_ids: Branch name -> label, as for
                :class:`AortaWorkflowSimulateFlow3D`.
            inflow_m3_s: Mean inflow of each interval of a periodic series,
                for example from :meth:`AortaFlowPhysiology.lv_inflow`.
            inflow_interval_s: Duration of each inflow interval in seconds.
            geometry: A :meth:`AortaProcessGeometry.prepare` result, used in
                place of preparing *labelmap_image*; its spacing replaces
                *spacing_mm*, and the Young's modulus is scaled to its
                ``wall_thickness_mm``.
            mesh: Lumen and wall meshes with the keys of
                :meth:`AortaProcessGeometry.fsi_mesh`, used in place of
                meshing the geometry: the lumen keys of
                :meth:`AortaWorkflowSimulateFlow3D.process` plus
                ``"wall_points_mm"``, ``"wall_tetrahedra"``,
                ``"wall_clamped"`` and ``"wall_tether_faces"``.  Every lumen
                point on a wall triangle (region 0) must be a wall point.
            beats: Number of heart beats to simulate, at least two.
            dt_s: Solver time step.
            element_size_mm: Largest interior edge length of the meshes
                (:meth:`AortaProcessGeometry.fsi_mesh`).
            precision: ``"float32"`` or ``"float64"`` blood operators and
                linear solves (:class:`WarpNavierStokes`); the wall is
                always float64.
            frame_interval_s: Interval of the saved velocity, pressure and
                displacement fields over the last beat; rounded to a
                multiple of *dt_s*.
            calibrate_outlets: Calibrate the outlet totals and the initial
                pressure to the input pressures, this inflow and the wall's
                measured compliance.  ``False`` uses the outlet totals of
                :attr:`physiology` as given.
            spacing_mm: Grid spacing of the prepared geometry, which sets
                the mesh size; the run time grows about as its inverse cube.
            collar_mm: Wall points this close to a port are clamped.

        Returns:
            ``dict`` with the keys of :meth:`AortaWorkflowSimulateFlow3D.process`,
            where ``'mesh'`` holds the lumen and wall meshes (millimetres),
            the velocity is at the P2 nodes of the lumen mesh as given, every
            field belongs to the mesh moved by the frame's displacement, and
            ``'gmres_iterations'`` and ``'residual'`` are those of the last
            blood solve of each step.  Added:

            - ``'series'`` - (T,) ``'total_gmres_iterations'``,
              ``'coupling_iterations'``, ``'converged'``,
              ``'coupling_residual'``, ``'coupling_residual_m'``,
              ``'wall_iterations'`` and ``'wall_tangent_updates'``, the
              coupling statistics of every step
              (:meth:`WarpFluidStructure.step`);
              ``'max_displacement_m'`` and ``'mean_displacement_m'``, the
              largest and mean interface displacement; ``'volume_m3'`` and
              ``'volume_rate_m3_s'``, the lumen volume and its rate of
              change.
            - ``'frames'`` - (F, points, 3) ``'displacement_m'`` of the lumen
              mesh points (the wall's on the wall, its harmonic extension
              inside).
            - ``'summary'`` - the 3D summary plus the total, wall and outlet
              compliances, the wall's metrics over the last beat (peak
              displacement and outward normal displacement in mm, lumen
              volume and mean diameter strain in percent, the lumen's
              compliance fitted to the root pressure, the distensibility
              and the Bramwell-Hill pulse-wave speed) and the coupling
              statistics of the whole run (``max_coupling_residual`` relative
              to the step's interface motion, ``max_coupling_residual_m`` the
              RMS residual in metres).  ``max_net_port_flow_relative``
              counts the lumen volume change as outflow.

        Raises:
            ValueError: If not exactly one of *labelmap_image* and *geometry*
                is given, if *beats* is below two, if *frame_interval_s* is
                longer than a beat, if *mesh* lacks a key or a port, if the
                inflow has no ejection, if an outlet has no flow fraction,
                or if no outlet resistance and compliance give the input
                pressures, as when the wall alone is too compliant.
            RuntimeError: If a linear solve or a coupled step diverges.
        """
        wall_thickness_mm = max(
            self.physiology.wall_thickness_mm, self.MIN_WALL_CELLS * spacing_mm
        )
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
            prepare_options={
                "spacing_mm": spacing_mm,
                "wall_thickness_mm": wall_thickness_mm,
            },
            mesh_options={"collar_mm": collar_mm},
        )

    # ─────────────────────────── Model hooks ───────────────────────────────

    def _mesh(
        self, geometry: Mapping[str, Any], element_size_mm: float, **options: Any
    ) -> dict[str, Any]:
        """Mesh the lumen and wall (:meth:`AortaProcessGeometry.fsi_mesh`)."""
        return self._geometry_tools.fsi_mesh(geometry, element_size_mm, **options)

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
        """Build the wall, measure its compliance, calibrate and couple."""
        given = self.physiology
        thickness_mm = float(geometry["wall_thickness_mm"])
        young_modulus_pa = (
            given.young_modulus_pa * given.wall_thickness_mm / thickness_mm
        )
        if thickness_mm != given.wall_thickness_mm:
            self.log_info(
                "Meshing the %.3g mm wall %.3g mm thick with Young's modulus "
                "%.3g Pa, keeping modulus x thickness",
                given.wall_thickness_mm,
                thickness_mm,
                young_modulus_pa,
            )
        wall = WarpHyperelasticWall(
            np.asarray(mesh["wall_points_mm"]) * 1e-3,
            mesh["wall_tetrahedra"],
            mesh["wall_clamped"],
            young_modulus=young_modulus_pa,
            poisson_ratio=given.poisson_ratio,
            tether_faces=mesh["wall_tether_faces"],
            tether_stiffness=given.tether_stiffness_pa_m,
        )
        solver: Optional[WarpNavierStokes] = None
        try:
            wall_compliance = self._wall_compliance(
                wall,
                np.asarray(mesh["points_mm"]) * 1e-3,
                mesh["boundary_faces"],
                mesh["boundary_regions"],
                (
                    given.diastolic_pressure_mmhg * AortaFlowPhysiology.MMHG_PA,
                    given.systolic_pressure_mmhg * AortaFlowPhysiology.MMHG_PA,
                ),
            )
            physiology = self._calibrate(
                *inflow, dt_s, calibrate_outlets, wall_compliance_m3_pa=wall_compliance
            )
            solver = self._navier_stokes(
                mesh, geometry["ports"], outlets, physiology, dt_s, precision
            )
            coupled = WarpFluidStructure(
                solver,
                wall,
                mesh["boundary_regions"],
                reference_pressure=physiology.initial_pressure_pa,
                **self.COUPLING,
            )
        except BaseException:
            wall.close()
            if solver is not None:
                solver.close()
            raise
        run: dict[str, Any] = {
            "spacing_mm": float(geometry["labelmap"].GetSpacing()[0]),
            "wall_thickness_mm": given.wall_thickness_mm,
            "meshed_wall_thickness_mm": thickness_mm,
            "meshed_young_modulus_pa": young_modulus_pa,
            "reference_pressure_mmhg": physiology.initial_pressure_pa
            / AortaFlowPhysiology.MMHG_PA,
            "coupling": dict(self.COUPLING),
            "wall_nodes": len(wall.dof_points),
            "wall_tetrahedra": len(wall.tetrahedra),
            "wall_clamped_points": int(np.sum(mesh["wall_clamped"])),
            "nonconforming_faces": mesh.get("nonconforming_faces"),
            "total_compliance_m3_pa": physiology.total_compliance_m3_pa
            + wall_compliance,
            "wall_compliance_m3_pa": wall_compliance,
        }
        return coupled, physiology, run

    def _frame_extras(self, stepper: Any) -> dict[str, np.ndarray]:
        """Return the frame's displacement of the lumen mesh points."""
        return {"displacement_m": stepper.mesh_displacement.copy()}

    def _step_message(self, step: Mapping[str, Any]) -> str:
        """Return the wall motion and coupling statistics of a step."""
        return (
            f"wall={step['max_displacement_m'] * 1e3:.3f} mm "
            f"coupling={step['coupling_iterations']}"
            f"{'' if step['converged'] else ' (not converged)'} "
            f"r={step['coupling_residual']:.1e} GMRES={step['gmres_iterations']}"
        )

    def _wall_compliance(
        self,
        wall: WarpHyperelasticWall,
        points_m: np.ndarray,
        faces: np.ndarray,
        regions: np.ndarray,
        pressures_pa: tuple[float, float],
    ) -> float:
        """Return the lumen's dV/dP in m^3/Pa from a quasi-static wall solve.

        The wall is prestressed to the first pressure, so that the imaged
        lumen is its shape there, and solved under the second, uniform over
        the lumen's wall faces; the compliance is the secant between the
        two.  The ports do not move.
        """
        started = time.perf_counter()
        low_pa, high_pa = pressures_pa
        lumen_wall = faces[regions == 0]
        moving = np.unique(lumen_wall)
        distance, to_wall = cKDTree(wall.points).query(points_m[moving])
        if np.max(distance) > 1e-9:
            raise ValueError("Lumen wall points are not wall points")
        index = np.zeros(len(points_m), dtype=np.int64)
        index[moving] = to_wall
        wall_faces = index[lumen_wall]
        wall.prestress(wall.pressure_load(wall_faces, low_pa))
        solve = wall.solve(wall.pressure_load(wall_faces, high_pa))
        moved = points_m.copy()
        moved[moving] += wall.displacement()[to_wall]
        volumes = [
            ProcessVolumetricMesh.enclosed_volume(points_m, faces),
            ProcessVolumetricMesh.enclosed_volume(moved, faces),
        ]
        change_m3 = volumes[1] - volumes[0]
        compliance = change_m3 / (high_pa - low_pa)
        self.log_info(
            "Wall compliance %.4g m^3/Pa: lumen volume %.2f mL grows %.3f mL "
            "from %.1f to %.1f mmHg, displacement up to %.3f mm; %d Newton "
            "iterations, %.1f s",
            compliance,
            volumes[0] * 1e6,
            change_m3 * 1e6,
            low_pa / AortaFlowPhysiology.MMHG_PA,
            high_pa / AortaFlowPhysiology.MMHG_PA,
            float(np.linalg.norm(wall.displacement()[to_wall], axis=1).max()) * 1e3,
            solve["iterations"],
            time.perf_counter() - started,
        )
        return float(compliance)

    # ─────────────────────────── Summary ───────────────────────────────────

    def _summarize(
        self, result: Mapping[str, Any], near_outlet: np.ndarray
    ) -> dict[str, Any]:
        """Return the metrics of the last simulated beat (see :meth:`process`)."""
        summary = super()._summarize(result, near_outlet)
        physiology = result["physiology"]
        series = result["series"]
        last = self._last_beat(series["times_s"], 60.0 / physiology.heart_rate_bpm)
        inflow = np.asarray(series["inlet_flow_m3_s"])
        outflow = sum(np.asarray(v) for v in series["outlet_flow_m3_s"].values())
        net = (
            np.asarray(series["root_flow_m3_s"])
            - outflow
            - np.asarray(series["volume_rate_m3_s"])
        )
        summary["max_net_port_flow_relative"] = float(
            np.abs(net[last]).max() / inflow[last].max()
        )

        # Outward displacement of the lumen's wall points, on their normals.
        mesh = result["mesh"]
        points = np.asarray(mesh["points_mm"])
        wall_faces = np.asarray(mesh["boundary_faces"])[
            np.asarray(mesh["boundary_regions"]) == 0
        ]
        corners = points[wall_faces]
        area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals = np.zeros_like(points)
        for k in range(3):
            np.add.at(normals, wall_faces[:, k], area)
        moving = np.unique(wall_faces)
        normals = normals[moving] / np.linalg.norm(normals[moving], axis=1)[:, None]
        displacement = np.asarray(result["frames"]["displacement_m"])[:, moving]
        outward = np.einsum("fpk,pk->fp", displacement, normals)

        volume = np.asarray(series["volume_m3"])[last]
        root = np.asarray(series["root_pressure_pa"])[last]
        # Lumen compliance: least-squares slope of volume against root pressure.
        lumen_compliance = float(np.polyfit(root, volume, 1)[0])
        distensibility_pa = lumen_compliance / float(volume.mean())
        coupling = np.asarray(series["coupling_iterations"])
        converged = np.asarray(series["converged"], dtype=bool)
        run = result["run"]
        summary.update(
            {
                "total_compliance_m3_pa": run["total_compliance_m3_pa"],
                "wall_compliance_m3_pa": run["wall_compliance_m3_pa"],
                "peak_wall_displacement_mm": float(
                    np.asarray(series["max_displacement_m"])[last].max() * 1e3
                ),
                "peak_outward_wall_displacement_mm": float(outward.max() * 1e3),
                "lumen_volume_ml": float(volume.min() * 1e6),
                "volume_strain_percent": float(np.ptp(volume) / volume.min() * 100),
                "diameter_strain_percent": float(
                    (np.sqrt(volume.max() / volume.min()) - 1) * 100
                ),
                "lumen_compliance_m3_pa": lumen_compliance,
                "distensibility_per_mmhg": distensibility_pa
                * AortaFlowPhysiology.MMHG_PA,
                "bramwell_hill_wave_speed_m_s": float(
                    1.0 / np.sqrt(physiology.density_kg_m3 * distensibility_pa)
                ),
                "mean_coupling_iterations": float(coupling.mean()),
                "max_coupling_iterations": int(coupling.max()),
                "capped_coupling_steps": int(
                    np.sum(coupling > run["coupling"]["max_iterations"])
                ),
                "unconverged_coupling_steps": int(np.sum(~converged)),
                "max_coupling_residual": float(
                    np.max(np.asarray(series["coupling_residual"]))
                ),
                "max_coupling_residual_m": float(
                    np.max(np.asarray(series["coupling_residual_m"]))
                ),
            }
        )
        return summary

    # ─────────────────────────── I/O helpers ───────────────────────────────

    @staticmethod
    def _extra_plot_panel(
        axis: Any, result: Mapping[str, Any], shown: np.ndarray
    ) -> None:
        """Plot the lumen volume change and the largest wall displacement."""
        series = result["series"]
        times = np.asarray(series["times_s"])
        volume = np.asarray(series["volume_m3"])
        axis.plot(
            times[shown],
            (volume[shown] - volume[shown].min()) * 1e6,
            label="lumen volume change (mL)",
        )
        axis.plot(
            times[shown],
            np.asarray(series["max_displacement_m"])[shown] * 1e3,
            label="largest wall displacement (mm)",
            linestyle="--",
        )
        axis.set_ylabel("Wall")
        axis.legend(loc="upper right")
