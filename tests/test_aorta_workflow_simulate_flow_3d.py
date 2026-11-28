"""Tests for AortaWorkflowSimulateFlow3D on a synthetic aortic arch.

Requires the ``[aorta]`` extra (``monai-physio[aorta]``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("skimage")

from monai_physio.aorta import AortaFlowPhysiology, AortaWorkflowSimulateFlow3D
from monai_physio.monai_physio_base import MONAIPhysioBase
from monai_physio.visualize_flow_fields import VisualizeFlowFields

from .test_aorta_flow_physiology import STROKE_ML, lv_volume_curve
from .test_aorta_process_geometry import AORTA, BRANCH, NETGEN, arch_labelmap
from .test_warp_navier_stokes import require_cudss

DT_S = 0.01
FRAME_INTERVAL_S = 0.05


@pytest.fixture(scope="module")
def arch_result() -> dict[str, Any]:
    """Two beats through the synthetic arch with one branch.

    The arch is the 56 x 24 x 64 labelmap of 1 mm voxels of
    ``test_aorta_process_geometry``; the inflow is the synthetic 36 mL LV
    curve.
    """
    require_cudss()
    physiology = AortaFlowPhysiology()
    inflow, interval_s = physiology.lv_inflow(lv_volume_curve())
    return AortaWorkflowSimulateFlow3D(physiology).process(
        arch_labelmap(),
        aorta_label_ids=(AORTA,),
        branch_label_ids={"BCT": BRANCH},
        inflow_m3_s=inflow,
        inflow_interval_s=interval_s,
        beats=2,
        dt_s=DT_S,
        frame_interval_s=FRAME_INTERVAL_S,
    )


@NETGEN
@pytest.mark.requires_gpu
def test_flow_is_conserved(arch_result: dict[str, Any]) -> None:
    """The root carries the imposed inflow and the outlets carry it out."""
    inflow = arch_result["series"]["inlet_flow_m3_s"]
    peak = inflow.max()
    np.testing.assert_allclose(
        arch_result["series"]["root_flow_m3_s"], inflow, rtol=0, atol=1e-3 * peak
    )
    outflow = sum(arch_result["series"]["outlet_flow_m3_s"].values())
    np.testing.assert_allclose(outflow, inflow, rtol=0, atol=1e-3 * peak)
    assert arch_result["summary"]["max_net_port_flow_relative"] < 1e-3
    assert set(arch_result["series"]["outlet_flow_m3_s"]) == {"distal", "bct"}


@NETGEN
@pytest.mark.requires_gpu
def test_outlet_pressures_replay_the_windkessels(arch_result: dict[str, Any]) -> None:
    """Fresh Windkessels fed the recorded outlet flows give the recorded pressures."""
    windkessels = arch_result["physiology"].windkessels(["distal", "bct"])
    for name, windkessel in windkessels.items():
        replayed = [
            windkessel.advance(float(q), DT_S)
            for q in arch_result["series"]["outlet_flow_m3_s"][name]
        ]
        np.testing.assert_allclose(
            replayed, arch_result["series"]["outlet_pressure_pa"][name], rtol=1e-12
        )


@NETGEN
@pytest.mark.requires_gpu
def test_summary_reports_the_last_beat(arch_result: dict[str, Any]) -> None:
    """The summary has the 1D workflow's keys for shared metrics plus 3D ones."""
    summary = arch_result["summary"]
    shared = {
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
    }
    assert set(summary) == shared | {
        "peak_speed_m_s",
        "peak_speed_away_from_outlets_m_s",
        "speed_99_9_percentile_m_s",
        "root_peak_reynolds",
        "max_net_port_flow_relative",
        "max_linear_residual",
        "mean_gmres_iterations",
        "unconverged_steps",
    }
    assert summary["stroke_volume_ml"] == pytest.approx(STROKE_ML, rel=1e-3)
    assert sum(summary["outlet_flow_shares"].values()) == pytest.approx(1, abs=1e-3)
    assert sum(summary["outlet_stroke_volume_ml"].values()) == pytest.approx(
        summary["stroke_volume_ml"], rel=1e-3
    )
    assert (
        summary["root_diastolic_mmhg"]
        < summary["root_mean_mmhg"]
        < summary["root_systolic_mmhg"]
    )
    assert 0 < summary["peak_speed_away_from_outlets_m_s"] <= summary["peak_speed_m_s"]
    assert summary["unconverged_steps"] == 0
    # Frames cover the last beat only.
    frame_times = arch_result["frames"]["times_s"]
    assert len(frame_times) == round(1.0 / FRAME_INTERVAL_S)
    assert frame_times[0] > arch_result["series"]["times_s"][-1] - 1.0


@NETGEN
@pytest.mark.requires_gpu
def test_save_results_open_in_the_viewer(
    arch_result: dict[str, Any], tmp_path: Path
) -> None:
    """The saved folder opens in VisualizeFlowFields and holds the summary."""
    assert set(arch_result) == {
        "geometry",
        "physiology",
        "mesh",
        "series",
        "frames",
        "run",
        "summary",
    }
    paths = AortaWorkflowSimulateFlow3D.save_results(arch_result, str(tmp_path))
    assert set(paths) == {
        "mesh",
        "time",
        "velocity_m_s",
        "pressure_pa",
        "steps",
        "run",
        "summary",
    }
    with open(paths["summary"], encoding="utf-8") as file:
        assert json.load(file) == arch_result["summary"]
    with open(paths["run"], encoding="utf-8") as file:
        assert json.load(file) == arch_result["run"]
    with np.load(paths["steps"]) as steps:
        np.testing.assert_array_equal(
            steps["outlet_flow_m3_s_bct"],
            arch_result["series"]["outlet_flow_m3_s"]["bct"],
        )
        np.testing.assert_array_equal(
            steps["gmres_iterations"], arch_result["series"]["gmres_iterations"]
        )
    viewer = VisualizeFlowFields(tmp_path)
    try:
        assert len(viewer.times) == len(arch_result["frames"]["times_s"])
        assert viewer.grid.n_points == arch_result["frames"]["velocity_m_s"].shape[1]
        assert viewer.index == int(np.argmax(arch_result["frames"]["inlet_flow_m3_s"]))
        viewer.set_field("velocity")
        pixels = viewer.screenshot(tmp_path / "view.png")
        assert pixels.size > 0
    finally:
        viewer.plotter.close()
    plot = Path(AortaWorkflowSimulateFlow3D.save_plots(arch_result, str(tmp_path)))
    assert plot.is_file() and plot.stat().st_size > 0


@NETGEN
@pytest.mark.requires_gpu
def test_process_takes_a_given_mesh(arch_result: dict[str, Any]) -> None:
    """A given geometry and mesh replace preparing and meshing.

    The arch's own mesh gives the fixture's flow; progress rises to 1.  The
    float32 GPU linear solves are not bit-reproducible: two runs of this arch
    have differed by about 0.7% of the root pressure at single samples in
    systole, so the last-beat root systolic, diastolic and mean pressures are
    compared within 2%.
    """
    reports: list[float] = []
    MONAIPhysioBase.set_progress_callback(
        lambda fraction, message: reports.append(fraction)
    )
    physiology = AortaFlowPhysiology()
    inflow, interval_s = physiology.lv_inflow(lv_volume_curve())
    try:
        result = AortaWorkflowSimulateFlow3D(physiology).process(
            geometry=arch_result["geometry"],
            mesh=arch_result["mesh"],
            inflow_m3_s=inflow,
            inflow_interval_s=interval_s,
            beats=2,
            dt_s=DT_S,
            frame_interval_s=FRAME_INTERVAL_S,
        )
    finally:
        MONAIPhysioBase.set_progress_callback(None)
    assert result["mesh"] is arch_result["mesh"]
    for key in ("root_systolic_mmhg", "root_diastolic_mmhg", "root_mean_mmhg"):
        assert result["summary"][key] == pytest.approx(
            arch_result["summary"][key], rel=2e-2
        ), key
    assert reports[0] == 0.0 and reports[-1] == 1.0
    assert np.all(np.diff(reports) >= 0)


def test_process_rejects_an_unusable_mesh() -> None:
    """A mesh without a required key or without a port's triangles fails."""
    ports = {
        "root": {"label_id": 10, "role": "inlet"},
        "distal": {"label_id": 11, "role": "outlet"},
    }
    mesh = {
        "points_mm": np.zeros((4, 3)),
        "tetrahedra": np.array([[0, 1, 2, 3]]),
        "boundary_faces": np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]),
        "boundary_regions": np.array([0, 10, 0, 0]),
    }
    workflow = AortaWorkflowSimulateFlow3D()
    inflow = np.array([0.0, 1e-4, 1e-4, 0.0])
    with pytest.raises(ValueError, match="lacks"):
        workflow.process(
            geometry={"ports": ports},
            mesh={key: mesh[key] for key in ("points_mm", "tetrahedra")},
            inflow_m3_s=inflow,
            inflow_interval_s=0.1,
        )
    with pytest.raises(ValueError, match=r"ports \['distal'\]"):
        workflow.process(
            geometry={"ports": ports},
            mesh=mesh,
            inflow_m3_s=inflow,
            inflow_interval_s=0.1,
        )


def test_process_requires_one_geometry_source() -> None:
    """Both or neither geometry source, one beat or a long frame interval fails."""
    workflow = AortaWorkflowSimulateFlow3D()
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
