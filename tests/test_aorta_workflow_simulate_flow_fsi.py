"""Tests for AortaWorkflowSimulateFlowFSI on a synthetic aortic arch.

Requires the ``[aorta]`` extra (``monai-physio[aorta]``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("skimage")

from monai_physio.aorta import (
    AortaFlowPhysiology,
    AortaProcessGeometry,
    AortaWorkflowSimulateFlow3D,
    AortaWorkflowSimulateFlowFSI,
)
from monai_physio.process_volumetric_mesh import ProcessVolumetricMesh
from monai_physio.visualize_flow_fields import VisualizeFlowFields
from monai_physio.warp_navier_stokes import WarpNavierStokes

from .test_aorta_flow_physiology import STROKE_ML, lv_volume_curve
from .test_aorta_process_geometry import AORTA, BRANCH, NETGEN, arch_labelmap
from .test_warp_navier_stokes import require_cudss

DT_S = 0.01
FRAME_INTERVAL_S = 0.05
# The workflow meshes the default 1.5 mm wall three 1 mm grid spacings thick.
SPACING_MM = 1.0
MESHED_WALL_MM = 3.0


def run_fsi(physiology: AortaFlowPhysiology) -> dict[str, Any]:
    """Two beats through the synthetic arch with one branch.

    The arch is the 56 x 24 x 64 labelmap of 1 mm voxels of
    ``test_aorta_process_geometry``; the inflow is the synthetic 36 mL LV
    curve.
    """
    inflow, interval_s = physiology.lv_inflow(lv_volume_curve())
    return AortaWorkflowSimulateFlowFSI(physiology).process(
        arch_labelmap(),
        aorta_label_ids=(AORTA,),
        branch_label_ids={"BCT": BRANCH},
        inflow_m3_s=inflow,
        inflow_interval_s=interval_s,
        beats=2,
        dt_s=DT_S,
        spacing_mm=SPACING_MM,
        frame_interval_s=FRAME_INTERVAL_S,
    )


@pytest.fixture(scope="module")
def fsi_result() -> dict[str, Any]:
    """The arch with the default wall."""
    require_cudss()
    return run_fsi(AortaFlowPhysiology())


@NETGEN
@pytest.mark.requires_gpu
def test_coupling_converges_and_conserves_volume(fsi_result: dict[str, Any]) -> None:
    """Every step converges; inflow - outflow - dV/dt is small in the last beat.

    The blood mesh moves once per step, to the predicted interface, so the
    balance closes to first order in the time step: in a development run on
    this arch the largest last-beat imbalance was 1.28%, 0.63% and 0.27% of
    peak inflow at 10, 5 and 2.5 ms steps.  These 10 ms steps keep the test
    short.
    """
    summary = fsi_result["summary"]
    assert summary["unconverged_coupling_steps"] == 0
    assert np.all(fsi_result["series"]["converged"])
    inflow = fsi_result["series"]["inlet_flow_m3_s"]
    np.testing.assert_allclose(
        fsi_result["series"]["root_flow_m3_s"], inflow, rtol=0, atol=1e-3 * inflow.max()
    )
    assert summary["max_net_port_flow_relative"] < 2e-2
    # The lumen does change volume, and the wall flux accounts for it.
    assert np.ptp(fsi_result["series"]["volume_m3"]) > 0
    outflow = sum(fsi_result["series"]["outlet_flow_m3_s"].values())
    stored = np.sum((inflow - outflow)[1:]) * DT_S
    change = (
        fsi_result["series"]["volume_m3"][-1] - fsi_result["series"]["volume_m3"][0]
    )
    assert abs(stored - change) < 1e-2 * np.sum(inflow) * DT_S


@NETGEN
@pytest.mark.requires_gpu
def test_meshes_conform_and_wall_is_meshed_thick(fsi_result: dict[str, Any]) -> None:
    """Lumen wall points are wall points; the thin wall is meshed 3 mm thick."""
    mesh = fsi_result["mesh"]
    lumen = mesh["points_mm"]
    wall_faces = mesh["boundary_faces"][mesh["boundary_regions"] == 0]
    on_wall = lumen[np.unique(wall_faces)]
    wall_points = {tuple(point) for point in mesh["wall_points_mm"]}
    assert all(tuple(point) in wall_points for point in on_wall)
    # Only port triangles fail to conform.
    assert mesh["nonconforming_faces"] < 0.05 * len(mesh["boundary_faces"])
    run = fsi_result["run"]
    assert run["meshed_wall_thickness_mm"] == pytest.approx(MESHED_WALL_MM)
    physiology = AortaFlowPhysiology()
    assert run["meshed_young_modulus_pa"] == pytest.approx(
        physiology.young_modulus_pa * physiology.wall_thickness_mm / MESHED_WALL_MM
    )
    # The displacement is the wall's on the interface and zero on the caps.
    caps = np.unique(mesh["boundary_faces"][mesh["boundary_regions"] != 0])
    np.testing.assert_array_equal(
        fsi_result["frames"]["displacement_m"][:, np.setdiff1d(caps, wall_faces)], 0
    )


@NETGEN
@pytest.mark.requires_gpu
def test_summary_reports_compliance_and_wall(fsi_result: dict[str, Any]) -> None:
    """The summary has the 3D keys plus the wall's; the compliances add up."""
    summary = fsi_result["summary"]
    wall_keys = {
        "total_compliance_m3_pa",
        "wall_compliance_m3_pa",
        "peak_wall_displacement_mm",
        "peak_outward_wall_displacement_mm",
        "lumen_volume_ml",
        "volume_strain_percent",
        "diameter_strain_percent",
        "lumen_compliance_m3_pa",
        "distensibility_per_mmhg",
        "bramwell_hill_wave_speed_m_s",
        "mean_coupling_iterations",
        "max_coupling_iterations",
        "capped_coupling_steps",
        "unconverged_coupling_steps",
        "max_coupling_residual",
        "max_coupling_residual_m",
    }
    rigid_keys = {
        "heart_rate_bpm",
        "input_systolic_mmhg",
        "input_diastolic_mmhg",
        "input_mean_mmhg",
        "outlet_total_resistance_pa_s_m3",
        "outlet_total_compliance_m3_pa",
        "root_systolic_mmhg",
        "root_diastolic_mmhg",
        "root_mean_mmhg",
        "root_pulse_pressure_mmhg",
        "root_to_distal_max_mmhg",
        "root_to_distal_min_mmhg",
        "mean_pressure_change_mmhg",
        "peak_inflow_ml_s",
        "stroke_volume_ml",
        "cardiac_output_l_min",
        "outlet_flow_shares",
        "outlet_stroke_volume_ml",
        "peak_speed_m_s",
        "peak_speed_away_from_outlets_m_s",
        "speed_99_9_percentile_m_s",
        "root_peak_reynolds",
        "max_net_port_flow_relative",
        "max_linear_residual",
        "mean_gmres_iterations",
        "unconverged_steps",
    }
    assert set(summary) == rigid_keys | wall_keys
    assert summary["stroke_volume_ml"] == pytest.approx(STROKE_ML, rel=1e-3)
    assert 0 < summary["wall_compliance_m3_pa"] < summary["total_compliance_m3_pa"]
    assert summary["outlet_total_compliance_m3_pa"] == pytest.approx(
        summary["total_compliance_m3_pa"] - summary["wall_compliance_m3_pa"]
    )
    assert summary["peak_outward_wall_displacement_mm"] > 0
    assert summary["volume_strain_percent"] > summary["diameter_strain_percent"] > 0
    assert summary["bramwell_hill_wave_speed_m_s"] > 0
    # The lumen's dynamic dV/dP is close to the static wall solve's.
    assert summary["lumen_compliance_m3_pa"] == pytest.approx(
        summary["wall_compliance_m3_pa"], rel=0.3
    )
    assert summary["unconverged_steps"] == 0
    frame_times = fsi_result["frames"]["times_s"]
    assert len(frame_times) == round(1.0 / FRAME_INTERVAL_S)


@NETGEN
@pytest.mark.requires_gpu
def test_wall_compliance_is_the_static_volume_change(
    fsi_result: dict[str, Any],
) -> None:
    """The wall compliance gives a volume change like the simulated one.

    The pulse pressure times the static compliance and the lumen's simulated
    volume range agree within a third.
    """
    summary = fsi_result["summary"]
    mesh = fsi_result["mesh"]
    volume = ProcessVolumetricMesh.enclosed_volume(
        mesh["points_mm"] * 1e-3, mesh["boundary_faces"]
    )
    assert summary["lumen_volume_ml"] == pytest.approx(volume * 1e6, rel=0.05)
    pulse_pa = summary["root_pulse_pressure_mmhg"] * AortaFlowPhysiology.MMHG_PA
    predicted = summary["wall_compliance_m3_pa"] * pulse_pa
    simulated = summary["volume_strain_percent"] / 100 * volume
    assert simulated == pytest.approx(predicted, rel=1 / 3)


@NETGEN
@pytest.mark.requires_gpu
def test_stiff_wall_matches_rigid_flow() -> None:
    """A very stiff wall gives the root pressure of rigid flow.

    On the same lumen mesh the rigid solver, fed the same inflow and
    outlets, gives the same root pressure during the slow flow of the first
    0.4 s, within the solvers' sensitivity to their linear-solve tolerance
    (rtol 1e-4); every coupled step takes a single iteration.  In the fast
    systolic flow after that the two differ more: in development runs on
    this arch the rigid solver alone changed by up to 0.6% between GMRES
    tolerances 1e-6 and 1e-9, and the stiff wall differed from it by up to
    0.5% in float32 and 0.2% in float64, so the 1% bound over the whole run
    reflects the rigid solver's own sensitivity to that tolerance.  The
    rigid 3D workflow's own, coarser mesh adds its mesh dependence to the
    pressure drop along the arch, so its root pressure is compared more
    loosely.
    """
    require_cudss()
    physiology = AortaFlowPhysiology(young_modulus_pa=1e12)
    stiff = run_fsi(physiology)
    summary = stiff["summary"]
    assert summary["peak_wall_displacement_mm"] < 1e-6
    assert summary["wall_compliance_m3_pa"] < 1e-6 * summary["total_compliance_m3_pa"]

    mesh = stiff["mesh"]
    ports = stiff["geometry"]["ports"]
    windkessels = stiff["physiology"].windkessels(["distal", "bct"])
    faces, regions = mesh["boundary_faces"], mesh["boundary_regions"]
    solver = WarpNavierStokes(
        mesh["points_mm"] * 1e-3,
        mesh["tetrahedra"],
        faces,
        regions,
        inlet_region=ports["root"]["label_id"],
        outlets={ports[n]["label_id"]: w for n, w in windkessels.items()},
        dt=DT_S,
        density=physiology.density_kg_m3,
        viscosity=physiology.viscosity_pa_s,
        backflow=AortaWorkflowSimulateFlowFSI.BACKFLOW_BETA,
        blunt_fraction=AortaWorkflowSimulateFlowFSI.INLET_BLUNT_FRACTION,
    )
    root = regions == ports["root"]["label_id"]
    weights = np.zeros(len(solver.points))
    np.add.at(
        weights,
        faces[root],
        np.linalg.norm(solver.area_vectors[root], axis=1)[:, None] / 3,
    )
    weights /= weights.sum()
    try:
        same_mesh = []
        for flow in stiff["series"]["inlet_flow_m3_s"]:
            solver.step(float(flow))
            same_mesh.append(weights @ solver.pressure())
    finally:
        solver.close()
    rigid_pa = np.asarray(same_mesh)
    early = stiff["series"]["times_s"] <= 0.4
    np.testing.assert_allclose(
        stiff["series"]["root_pressure_pa"][early], rigid_pa[early], rtol=1e-4
    )
    np.testing.assert_allclose(
        stiff["series"]["root_pressure_pa"],
        rigid_pa,
        rtol=0,
        atol=1e-2 * np.mean(rigid_pa),
    )

    inflow, interval_s = physiology.lv_inflow(lv_volume_curve())
    rigid = AortaWorkflowSimulateFlow3D(physiology).process(
        arch_labelmap(),
        aorta_label_ids=(AORTA,),
        branch_label_ids={"BCT": BRANCH},
        inflow_m3_s=inflow,
        inflow_interval_s=interval_s,
        beats=2,
        dt_s=DT_S,
        frame_interval_s=FRAME_INTERVAL_S,
    )
    for key in ("root_mean_mmhg", "root_diastolic_mmhg", "root_systolic_mmhg"):
        assert summary[key] == pytest.approx(rigid["summary"][key], rel=1e-2)
    np.testing.assert_allclose(
        stiff["series"]["root_pressure_pa"],
        rigid["series"]["root_pressure_pa"],
        rtol=0,
        atol=3e-2 * np.mean(rigid["series"]["root_pressure_pa"]),
    )


@NETGEN
@pytest.mark.requires_gpu
def test_save_results_open_in_the_viewer(
    fsi_result: dict[str, Any], tmp_path: Path
) -> None:
    """The saved folder opens in VisualizeFlowFields with the displacement."""
    paths = AortaWorkflowSimulateFlowFSI.save_results(fsi_result, str(tmp_path))
    assert set(paths) == {
        "mesh",
        "time",
        "velocity_m_s",
        "pressure_pa",
        "displacement_m",
        "steps",
        "run",
        "summary",
    }
    with open(paths["summary"], encoding="utf-8") as file:
        assert json.load(file) == fsi_result["summary"]
    with np.load(paths["steps"]) as steps:
        np.testing.assert_array_equal(
            steps["coupling_iterations"], fsi_result["series"]["coupling_iterations"]
        )
        # The last blood solve's GMRES iterations, and their sum over the step.
        assert np.all(steps["total_gmres_iterations"] >= steps["gmres_iterations"])
    viewer = VisualizeFlowFields(tmp_path)
    try:
        assert "displacement" in viewer.fields
        assert viewer.grid.n_points == fsi_result["frames"]["velocity_m_s"].shape[1]
        viewer.set_field("displacement")
        low, high = viewer.field_limits("displacement", np.arange(len(viewer.times)))
        assert low == 0 and high > 0
        pixels = viewer.screenshot(tmp_path / "view.png")
        assert pixels.size > 0
    finally:
        viewer.plotter.close()
    plot = Path(AortaWorkflowSimulateFlowFSI.save_plots(fsi_result, str(tmp_path)))
    assert plot.is_file() and plot.stat().st_size > 0


@NETGEN
@pytest.mark.requires_gpu
def test_too_compliant_wall_is_an_error() -> None:
    """No outlets restore the input pressures behind a too-soft wall.

    With a thousandth of the synthetic stroke volume the wall alone damps
    the root pulse pressure far below the input's.
    """
    require_cudss()
    geometry = AortaProcessGeometry().prepare(
        arch_labelmap(),
        aorta_label_ids=(AORTA,),
        branch_label_ids={"BCT": BRANCH},
        spacing_mm=SPACING_MM,
        wall_thickness_mm=MESHED_WALL_MM,
    )
    physiology = AortaFlowPhysiology()
    inflow, interval_s = physiology.lv_inflow(lv_volume_curve())
    with pytest.raises(ValueError, match="No outlet resistance"):
        AortaWorkflowSimulateFlowFSI(physiology).process(
            geometry=geometry,
            inflow_m3_s=1e-3 * inflow,
            inflow_interval_s=interval_s,
            dt_s=DT_S,
        )


def test_process_rejects_a_mesh_without_a_wall() -> None:
    """A rigid lumen mesh lacks the wall keys the FSI model needs."""
    ports = {
        "root": {"label_id": 10, "role": "inlet"},
        "distal": {"label_id": 11, "role": "outlet"},
    }
    mesh = {
        "points_mm": np.zeros((4, 3)),
        "tetrahedra": np.array([[0, 1, 2, 3]]),
        "boundary_faces": np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]),
        "boundary_regions": np.array([0, 10, 11, 0]),
    }
    with pytest.raises(ValueError, match="wall_points_mm"):
        AortaWorkflowSimulateFlowFSI().process(
            geometry={"ports": ports, "wall_thickness_mm": 3.0},
            mesh=mesh,
            inflow_m3_s=np.array([0.0, 1e-4, 1e-4, 0.0]),
            inflow_interval_s=0.1,
        )


def test_process_requires_one_geometry_source() -> None:
    """Both or neither geometry source, one beat or a long frame interval fails."""
    workflow = AortaWorkflowSimulateFlowFSI()
    inflow = np.ones(4)
    with pytest.raises(ValueError, match="exactly one"):
        workflow.process(inflow_m3_s=inflow, inflow_interval_s=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        workflow.process(
            arch_labelmap(),
            geometry={},
            inflow_m3_s=inflow,
            inflow_interval_s=0.1,
        )
    with pytest.raises(ValueError, match="beats"):
        workflow.process(
            arch_labelmap(), inflow_m3_s=inflow, inflow_interval_s=0.1, beats=1
        )
    with pytest.raises(ValueError, match="frame_interval_s"):
        workflow.process(
            arch_labelmap(),
            inflow_m3_s=inflow,
            inflow_interval_s=0.1,
            frame_interval_s=1.5,
        )
