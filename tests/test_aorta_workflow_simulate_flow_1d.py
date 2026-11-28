"""Tests for AortaWorkflowSimulateFlow1D on synthetic tubes and a synthetic arch.

Requires the ``[aorta]`` extra (``monai-physio[aorta]``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
import pyvista as pv

pytest.importorskip("skimage")

from monai_physio.aorta import AortaFlowPhysiology, AortaWorkflowSimulateFlow1D
from monai_physio.monai_physio_base import MONAIPhysioBase
from monai_physio.warp_navier_stokes import Windkessel

from .test_aorta_flow_physiology import BEATS, STROKE_ML, lv_volume_curve
from .test_aorta_process_geometry import AORTA, BRANCH, arch_labelmap

LENGTH_MM = 100.0
RADIUS_MM = 10.0
# Halfway between two 1 mm sections, where the trapezoidal rule integrates
# the step in the flow fraction exactly.
BRANCH_ARC_MM = 40.5


def tube_geometry(branch: bool) -> dict[str, Any]:
    """Return a straight 100 mm tube of 10 mm radius as a prepared geometry.

    The centerline has 1 mm sections; with *branch*, a ``bct`` outlet leaves
    at :data:`BRANCH_ARC_MM`.
    """
    arc = np.arange(0.0, LENGTH_MM + 0.5)
    centerline = pv.PolyData(
        np.column_stack([np.zeros_like(arc), np.zeros_like(arc), -arc]),
        lines=np.r_[len(arc), np.arange(len(arc))],
    )
    centerline.point_data["arc_length_mm"] = arc
    centerline.point_data["area_mm2"] = np.full(len(arc), math.pi * RADIUS_MM**2)
    centerline.point_data["radius_mm"] = np.full(len(arc), RADIUS_MM)
    surface = pv.Cylinder(
        center=(0.0, 0.0, -LENGTH_MM / 2),
        direction=(0.0, 0.0, 1.0),
        radius=RADIUS_MM,
        height=LENGTH_MM,
    ).triangulate()
    surface.cell_data["Region"] = np.zeros(surface.n_cells, dtype=np.int64)
    ports: dict[str, dict[str, Any]] = {
        "root": {"label_id": 10, "role": "inlet", "arc_length_mm": 0.0},
        "distal": {"label_id": 11, "role": "outlet", "arc_length_mm": LENGTH_MM},
    }
    if branch:
        ports["bct"] = {
            "label_id": 12,
            "role": "outlet",
            "arc_length_mm": BRANCH_ARC_MM,
        }
    return {"centerline": centerline, "surface": surface, "ports": ports}


def simulate(
    geometry: dict[str, Any], physiology: Optional[AortaFlowPhysiology] = None
) -> dict[str, Any]:
    """Run three beats of the synthetic 36 mL LV curve through *geometry*."""
    physiology = physiology or AortaFlowPhysiology()
    inflow, interval_s = physiology.lv_inflow(lv_volume_curve())
    return AortaWorkflowSimulateFlow1D(physiology).process(
        geometry=geometry,
        inflow_m3_s=inflow,
        inflow_interval_s=interval_s,
        beats=BEATS,
    )


@pytest.fixture(scope="module")
def branched_result() -> dict[str, Any]:
    """The tube with a branch, simulated with the default physiology."""
    return simulate(tube_geometry(branch=True))


def expected_drop(result: dict[str, Any], fraction_integral_m: np.ndarray) -> Any:
    """Return the Poiseuille and inertance drop of a uniform tube.

    *fraction_integral_m* is the integral of the flow fraction from each
    section to the distal end.
    """
    physiology = AortaFlowPhysiology()
    area = math.pi * (RADIUS_MM * 1e-3) ** 2
    inertance = physiology.density_kg_m3 / area * fraction_integral_m
    resistance = 8 * math.pi * physiology.viscosity_pa_s / area**2 * fraction_integral_m
    return (
        np.asarray(result["series"]["inlet_flow_derivative_m3_s2"])[:, None] * inertance
        + np.asarray(result["series"]["inlet_flow_m3_s"])[:, None] * resistance
    )


@pytest.mark.requires_gpu
def test_straight_tube_matches_poiseuille_and_inertance() -> None:
    """A uniform tube's pressure drop is (L - s)(rho/A dQ/dt + 8 pi mu/A^2 Q)."""
    result = simulate(tube_geometry(branch=False))
    remaining_m = (LENGTH_MM - np.arange(0.0, LENGTH_MM + 0.5)) * 1e-3
    drop = (
        result["series"]["pressure_pa"]
        - result["series"]["outlet_pressure_pa"]["distal"][:, None]
    )
    expected = expected_drop(result, remaining_m)
    np.testing.assert_allclose(drop, expected, rtol=1e-9, atol=1e-9)
    # The distal section carries the distal outlet's pressure exactly.
    series = result["series"]
    np.testing.assert_array_equal(
        series["pressure_pa"][:, -1], series["outlet_pressure_pa"]["distal"]
    )
    np.testing.assert_array_equal(
        series["root_pressure_pa"], series["pressure_pa"][:, 0]
    )
    area = math.pi * (RADIUS_MM * 1e-3) ** 2
    velocity = result["series"]["inlet_flow_m3_s"][:, None] / area
    np.testing.assert_allclose(
        result["series"]["velocity_m_s"],
        np.broadcast_to(velocity, drop.shape),
        rtol=1e-12,
    )


@pytest.mark.requires_gpu
def test_branch_takes_its_share(branched_result: dict[str, Any]) -> None:
    """Flow drops by the branch share at its port; every outlet shares P_out."""
    share = 0.17 / (0.70 + 0.17)
    arc = np.arange(0.0, LENGTH_MM + 0.5)
    expected_fraction = np.where(arc > BRANCH_ARC_MM, 1.0 - share, 1.0)
    np.testing.assert_allclose(
        branched_result["sections"]["flow_fraction"], expected_fraction
    )
    inflow = branched_result["series"]["inlet_flow_m3_s"]
    np.testing.assert_allclose(
        branched_result["series"]["flow_m3_s"],
        inflow[:, None] * expected_fraction[None, :],
    )
    outlet_flow = branched_result["series"]["outlet_flow_m3_s"]
    np.testing.assert_allclose(outlet_flow["bct"], share * inflow)
    np.testing.assert_allclose(outlet_flow["bct"] + outlet_flow["distal"], inflow)

    # Windkessels split in proportion to flow give the whole-body Windkessel.
    physiology = branched_result["physiology"]
    whole_body = Windkessel.from_total(
        1.0,
        physiology.total_resistance_pa_s_m3,
        physiology.total_compliance_m3_pa,
        physiology.proximal_fraction,
        physiology.initial_pressure_pa,
    )
    dt_s = float(np.diff(branched_result["series"]["times_s"])[0])
    expected_pressure = np.asarray([whole_body.advance(q, dt_s) for q in inflow])
    for pressure in branched_result["series"]["outlet_pressure_pa"].values():
        np.testing.assert_allclose(pressure, expected_pressure, rtol=1e-12)

    integral_mm = np.where(
        arc < BRANCH_ARC_MM,
        (BRANCH_ARC_MM - arc) + (LENGTH_MM - BRANCH_ARC_MM) * (1.0 - share),
        (LENGTH_MM - arc) * (1.0 - share),
    )
    drop = branched_result["series"]["pressure_pa"] - expected_pressure[:, None]
    np.testing.assert_allclose(
        drop, expected_drop(branched_result, integral_mm * 1e-3), rtol=1e-9, atol=1e-9
    )


@pytest.mark.requires_gpu
def test_summary_reports_the_last_beat(branched_result: dict[str, Any]) -> None:
    """The last beat carries the whole 36 mL ejection at 60 bpm."""
    summary = branched_result["summary"]
    assert set(summary) == {
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
        "peak_velocity_m_s",
        "peak_reynolds",
    }
    assert summary["stroke_volume_ml"] == pytest.approx(STROKE_ML, rel=1e-4)
    assert summary["cardiac_output_l_min"] == pytest.approx(2.16, rel=1e-4)
    simulated = branched_result["physiology"]
    assert summary["input_systolic_mmhg"] == 112.0
    assert summary["input_diastolic_mmhg"] == 78.0
    assert summary["input_mean_mmhg"] == pytest.approx(78.0 + 34.0 / 3.0)
    assert (
        summary["outlet_total_resistance_pa_s_m3"] == simulated.total_resistance_pa_s_m3
    )
    assert summary["outlet_total_compliance_m3_pa"] == simulated.total_compliance_m3_pa
    assert sum(summary["outlet_stroke_volume_ml"].values()) == pytest.approx(
        summary["stroke_volume_ml"]
    )
    assert summary["root_diastolic_mmhg"] < summary["root_mean_mmhg"]
    assert summary["root_mean_mmhg"] < summary["root_systolic_mmhg"]
    assert summary["root_pulse_pressure_mmhg"] == pytest.approx(
        summary["root_systolic_mmhg"] - summary["root_diastolic_mmhg"]
    )
    area = math.pi * (RADIUS_MM * 1e-3) ** 2
    peak_velocity = summary["peak_inflow_ml_s"] * 1e-6 / area
    assert summary["peak_velocity_m_s"] == pytest.approx(peak_velocity, rel=1e-6)
    physiology = AortaFlowPhysiology()
    assert summary["peak_reynolds"] == pytest.approx(
        physiology.density_kg_m3
        * peak_velocity
        * 2e-3
        * RADIUS_MM
        / physiology.viscosity_pa_s,
        rel=1e-6,
    )


@pytest.mark.requires_gpu
def test_outlets_are_calibrated_to_the_input_pressures() -> None:
    """The outlet pressure has the input extremes from the first beat."""
    physiology = AortaFlowPhysiology(
        systolic_pressure_mmhg=124.0, diastolic_pressure_mmhg=82.0
    )
    inflow, interval_s = physiology.lv_inflow(lv_volume_curve())
    workflow = AortaWorkflowSimulateFlow1D(physiology)
    result = workflow.process(
        geometry=tube_geometry(branch=False),
        inflow_m3_s=inflow,
        inflow_interval_s=interval_s,
    )
    assert workflow.physiology is physiology
    assert physiology.total_resistance_pa_s_m3 == 3.33e8
    times = result["series"]["times_s"]
    outlet = (
        result["series"]["outlet_pressure_pa"]["distal"] / AortaFlowPhysiology.MMHG_PA
    )
    for beat in range(BEATS):
        window = (times >= beat) & (times < beat + 1)
        assert outlet[window].max() == pytest.approx(124.0, abs=0.5)
        assert outlet[window].min() == pytest.approx(82.0, abs=0.5)
    assert abs(result["summary"]["mean_pressure_change_mmhg"]) < 0.05

    given = AortaWorkflowSimulateFlow1D(physiology).process(
        geometry=tube_geometry(branch=False),
        inflow_m3_s=inflow,
        inflow_interval_s=interval_s,
        calibrate_outlets=False,
    )
    assert given["physiology"] is physiology
    assert given["summary"]["outlet_total_resistance_pa_s_m3"] == 3.33e8


@pytest.mark.requires_gpu
def test_save_results_and_plots(
    branched_result: dict[str, Any], tmp_path: Path
) -> None:
    """Every file is written and reads back with the last-beat arrays."""
    assert set(branched_result) == {
        "geometry",
        "physiology",
        "sections",
        "series",
        "summary",
    }
    paths = AortaWorkflowSimulateFlow1D.save_results(branched_result, str(tmp_path))
    assert set(paths) == {"centerline", "surface", "fields", "summary"}
    centerline = pv.read(paths["centerline"])
    for name in (
        "systolic_pressure_mmhg",
        "diastolic_pressure_mmhg",
        "mean_pressure_mmhg",
        "peak_velocity_m_s",
        "flow_fraction",
    ):
        assert name in centerline.point_data
    systolic = np.asarray(centerline.point_data["systolic_pressure_mmhg"])
    assert systolic[0] == pytest.approx(
        branched_result["summary"]["root_systolic_mmhg"]
    )
    assert (
        pv.read(paths["surface"]).n_cells
        == branched_result["geometry"]["surface"].n_cells
    )
    with np.load(paths["fields"]) as fields:
        np.testing.assert_array_equal(
            fields["pressure_pa"], branched_result["series"]["pressure_pa"]
        )
        assert "outlet_pressure_pa_bct" in fields
        np.testing.assert_array_equal(
            fields["flow_fraction"], branched_result["sections"]["flow_fraction"]
        )
    with open(paths["summary"], encoding="utf-8") as file:
        assert json.load(file) == branched_result["summary"]
    plot = Path(AortaWorkflowSimulateFlow1D.save_plots(branched_result, str(tmp_path)))
    assert plot.is_file() and plot.stat().st_size > 0


@pytest.mark.requires_gpu
def test_process_prepares_a_labelmap() -> None:
    """A labelmap is prepared into root, distal and branch ports."""
    physiology = AortaFlowPhysiology()
    inflow, interval_s = physiology.lv_inflow(lv_volume_curve())
    result = AortaWorkflowSimulateFlow1D(physiology).process(
        arch_labelmap(),
        aorta_label_ids=(AORTA,),
        branch_label_ids={"BCT": BRANCH},
        inflow_m3_s=inflow,
        inflow_interval_s=interval_s,
    )
    assert set(result["series"]["outlet_pressure_pa"]) == {"distal", "bct"}
    fraction = result["sections"]["flow_fraction"]
    assert fraction[0] == 1.0
    assert fraction[-1] == pytest.approx(0.70 / 0.87)
    assert result["summary"]["stroke_volume_ml"] == pytest.approx(STROKE_ML, rel=1e-4)


def test_process_gives_an_absent_branch_share_to_the_other_outlets() -> None:
    """A branch label with no voxels gets no port, and its share is spread.

    The remaining outlets' shares are renormalized to sum to one, so every
    outlet together still carries the whole inflow.
    """
    physiology = AortaFlowPhysiology()
    inflow, interval_s = physiology.lv_inflow(lv_volume_curve())
    result = AortaWorkflowSimulateFlow1D(physiology).process(
        arch_labelmap(),
        aorta_label_ids=(AORTA,),
        branch_label_ids={"BCT": BRANCH, "LCCA": BRANCH + 1},
        inflow_m3_s=inflow,
        inflow_interval_s=interval_s,
    )
    assert set(result["series"]["outlet_pressure_pa"]) == {"distal", "bct"}
    shares = result["summary"]["outlet_flow_shares"]
    assert shares == pytest.approx({"distal": 0.70 / 0.87, "bct": 0.17 / 0.87})
    outlet_ml = sum(result["summary"]["outlet_stroke_volume_ml"].values())
    assert outlet_ml == pytest.approx(result["summary"]["stroke_volume_ml"])


@pytest.mark.requires_gpu
def test_process_reports_progress() -> None:
    """Progress rises from 0 to 1 and never falls."""
    reports: list[tuple[float, str]] = []
    MONAIPhysioBase.set_progress_callback(
        lambda fraction, message: reports.append((fraction, message))
    )
    try:
        simulate(tube_geometry(branch=False))
    finally:
        MONAIPhysioBase.set_progress_callback(None)
    fractions = [fraction for fraction, _ in reports]
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert np.all(np.diff(fractions) >= 0)


def test_process_requires_one_geometry_source() -> None:
    """Neither or both of a labelmap and a geometry is an error."""
    workflow = AortaWorkflowSimulateFlow1D()
    inflow = np.ones(4)
    with pytest.raises(ValueError, match="exactly one"):
        workflow.process(inflow_m3_s=inflow, inflow_interval_s=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        workflow.process(
            arch_labelmap(),
            geometry=tube_geometry(branch=False),
            inflow_m3_s=inflow,
            inflow_interval_s=0.1,
        )
    with pytest.raises(ValueError, match="beats"):
        workflow.process(
            geometry=tube_geometry(branch=False),
            inflow_m3_s=inflow,
            inflow_interval_s=0.1,
            beats=1,
        )
