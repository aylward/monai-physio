"""Tests for the aortic flow physiology inputs.

Requires the ``[aorta]`` extra (``monai-physio[aorta]``).
"""

from __future__ import annotations

import itk
import numpy as np
import pytest
from scipy.integrate import simpson

pytest.importorskip("skimage")

from monai_physio.aorta import AortaFlowPhysiology
from monai_physio.warp_navier_stokes import Windkessel

BEATS = 3
FRAMES_PER_BEAT = 33
STROKE_ML = 36.0


def lv_volume_curve(noise_ml: float = 0.0) -> np.ndarray:
    """Return three synthetic beats of LV volume in mL, 33 frames each.

    Each beat ejects 36 mL over frames 0-10, refills over frames 10-20 and
    stays at 120 mL over frames 20-32, where ``noise_ml`` uniform noise is
    added to frames 21-31.
    """
    rng = np.random.default_rng(7)
    beat = np.concatenate(
        [
            np.linspace(120.0, 120.0 - STROKE_ML, 11),
            np.linspace(120.0 - STROKE_ML, 120.0, 11)[1:],
            np.full(12, 120.0),
        ]
    )
    beats = []
    for _ in range(BEATS):
        noisy = beat.copy()
        noisy[21:32] += rng.uniform(-noise_ml, noise_ml, 11)
        beats.append(noisy)
    return np.concatenate(beats)


def test_lv_ejection_inflow_clean_curve() -> None:
    """Every ejection carries the stroke volume and filling carries nothing."""
    interval_s = 0.03
    flux = AortaFlowPhysiology.lv_ejection_inflow(lv_volume_curve(), interval_s)
    assert flux.shape == (BEATS * FRAMES_PER_BEAT,)
    assert np.all(flux >= 0)
    assert AortaFlowPhysiology._ejection_count(flux) == BEATS
    per_beat_ml = flux.reshape(BEATS, -1).sum(axis=1) * interval_s * 1e6
    np.testing.assert_allclose(per_beat_ml, STROKE_ML, rtol=1e-12)
    assert np.all(flux.reshape(BEATS, -1)[:, 10:] == 0)


def test_lv_ejection_inflow_discards_noise_ejections() -> None:
    """Sub-0.5 mL noise losses during diastasis are not counted as ejections."""
    volume = lv_volume_curve(noise_ml=0.25)
    unfiltered = AortaFlowPhysiology.lv_ejection_inflow(
        volume, 0.03, minimum_ejection_fraction=0.0
    )
    assert AortaFlowPhysiology._ejection_count(unfiltered) > BEATS
    flux = AortaFlowPhysiology.lv_ejection_inflow(volume, 0.03)
    assert AortaFlowPhysiology._ejection_count(flux) == BEATS
    # The clean frames bracketing diastasis isolate every noise loss.
    np.testing.assert_array_equal(
        flux, AortaFlowPhysiology.lv_ejection_inflow(lv_volume_curve(), 0.03)
    )


def test_inlet_flow_preserves_stroke_volume_and_periodicity() -> None:
    """The smoothed inflow integrates to the stroke volume of every beat."""
    physiology = AortaFlowPhysiology()
    flux, interval_s = physiology.lv_inflow(lv_volume_curve())
    beat_s = FRAMES_PER_BEAT * interval_s
    # Beat windows start in diastasis, many smoothing widths from ejection.
    start_s = 26 * interval_s
    for beat in range(BEATS):
        times = np.linspace(0.0, beat_s, 20001) + start_s + beat * beat_s
        flow, _ = physiology.inlet_flow(times, flux, interval_s)
        assert flow.min() >= 0.0
        volume_ml = simpson(flow, x=times) * 1e6
        assert volume_ml == pytest.approx(STROKE_ML, rel=1e-6)
    times = np.linspace(0.0, BEATS * beat_s, 1001)
    flow, derivative = physiology.inlet_flow(times, flux, interval_s)
    shifted, shifted_derivative = physiology.inlet_flow(
        times + BEATS * beat_s, flux, interval_s
    )
    np.testing.assert_allclose(shifted, flow, atol=1e-12 * flow.max())
    np.testing.assert_allclose(
        shifted_derivative, derivative, atol=1e-9 * np.abs(derivative).max()
    )


def test_retimed_interval_matches_experiments() -> None:
    """Three beats in 100 frames give the experiments' 0.03 s at 60 bpm."""
    volumes = AortaFlowPhysiology.reference_lv_volumes_ml()
    assert AortaFlowPhysiology().lv_inflow(volumes)[1] == pytest.approx(0.03)
    fast = AortaFlowPhysiology(heart_rate_bpm=75.0)
    assert fast.lv_inflow(volumes)[1] == pytest.approx(0.024)


def test_lv_volumes_ml_uses_spacing_with_rotated_direction() -> None:
    """LV voxels of a 12-voxel synthetic labelmap times the voxel volume."""
    spacing = (0.5, 0.8, 1.25)
    angle = np.deg2rad(30.0)
    direction = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    labelmaps = []
    counts = []
    for frame in range(3):
        array = np.zeros((12, 12, 12), dtype=np.uint8)
        array[2 : 6 + frame, 3:8, 4:9] = 1
        array[8:10, 3:5, 4:6] = 2
        array[0, 0, :] = 3
        counts.append((4 + frame) * 5 * 5 + 2 * 2 * 2)
        labelmap = itk.image_from_array(array)
        labelmap.SetSpacing(spacing)
        labelmap.SetDirection(itk.matrix_from_array(direction))
        labelmap.SetOrigin((10.0, -4.0, 2.5))
        labelmaps.append(labelmap)
    volumes = AortaFlowPhysiology.lv_volumes_ml(labelmaps, [1, 2])
    voxel_ml = 0.5 * 0.8 * 1.25 / 1000.0
    np.testing.assert_allclose(volumes, np.asarray(counts) * voxel_ml, rtol=1e-12)


def parallel_totals(
    physiology: AortaFlowPhysiology, names: list[str]
) -> tuple[float, float]:
    """Return the parallel resistance and summed compliance of the outlets."""
    windkessels = physiology.windkessels(names)
    assert list(windkessels) == names
    conductance = sum(
        1.0 / (w.proximal_resistance + w.distal_resistance)
        for w in windkessels.values()
    )
    for w in windkessels.values():
        resistance = w.proximal_resistance + w.distal_resistance
        assert w.proximal_resistance == pytest.approx(
            physiology.proximal_fraction * resistance
        )
        assert w.pressure == pytest.approx(80.0 * AortaFlowPhysiology.MMHG_PA)
    return 1.0 / conductance, sum(w.compliance for w in windkessels.values())


@pytest.mark.parametrize(
    "names",
    [
        ["distal", "bct", "lsa", "lcca"],
        ["distal", "bct", "lsa"],
        ["lcca", "distal"],
        ["bct"],
    ],
)
@pytest.mark.parametrize(
    "physiology",
    [
        AortaFlowPhysiology(),
        AortaFlowPhysiology(
            total_resistance_pa_s_m3=3.55e8, total_compliance_m3_pa=2.2e-9
        ),
    ],
)
def test_windkessels_reproduce_totals(
    physiology: AortaFlowPhysiology, names: list[str]
) -> None:
    """Renormalized fractions keep the total resistance and compliance."""
    resistance, compliance = parallel_totals(physiology, names)
    assert resistance == pytest.approx(physiology.total_resistance_pa_s_m3)
    assert compliance == pytest.approx(physiology.total_compliance_m3_pa)


def test_windkessels_split_by_flow_fraction() -> None:
    """An outlet's resistance scales with the inverse of its renormalized share."""
    physiology = AortaFlowPhysiology()
    windkessels = physiology.windkessels(["distal", "bct"])
    distal = windkessels["distal"]
    resistance = distal.proximal_resistance + distal.distal_resistance
    assert resistance == pytest.approx(3.33e8 * 0.87 / 0.70)
    assert distal.compliance == pytest.approx(6.8e-9 * 0.70 / 0.87)


def test_windkessels_reject_unknown_outlets() -> None:
    """Unknown outlet names are reported by name."""
    with pytest.raises(ValueError, match="'renal'"):
        AortaFlowPhysiology().windkessels(["distal", "renal"])


def test_lv_inflow_retimes_to_the_heart_rate() -> None:
    """Three 33-frame beats at 60 bpm give 1/33 s intervals and 36 mL beats."""
    physiology = AortaFlowPhysiology()
    flux, interval_s = physiology.lv_inflow(lv_volume_curve())
    assert interval_s == pytest.approx(1.0 / FRAMES_PER_BEAT)
    np.testing.assert_array_equal(
        flux, AortaFlowPhysiology.lv_ejection_inflow(lv_volume_curve(), interval_s)
    )
    assert flux.sum() * interval_s * 1e6 == pytest.approx(BEATS * STROKE_ML)
    with pytest.raises(ValueError, match="no ejection"):
        physiology.lv_inflow(np.full(10, 120.0))


def test_reference_lv_volumes_mirror_tutorial_13() -> None:
    """Three 31.5 mL beats from 72 mL, 0.03 s per frame at 60 bpm."""
    volumes = AortaFlowPhysiology.reference_lv_volumes_ml()
    assert volumes.shape == (100,)
    assert volumes.max() == pytest.approx(72.0)
    assert volumes.min() == pytest.approx(40.5)
    physiology = AortaFlowPhysiology(heart_rate_bpm=60.0)
    flux, interval_s = physiology.lv_inflow(volumes)
    assert interval_s == pytest.approx(0.03)
    assert AortaFlowPhysiology._ejection_count(flux) == 3
    assert flux.sum() * interval_s * 1e6 == pytest.approx(3 * 31.5)
    assert flux.max() * 1e6 == pytest.approx(137.0, rel=0.01)


def test_mean_pressure_uses_the_one_third_rule() -> None:
    """The input mean pressure is diastolic plus a third of the pulse."""
    physiology = AortaFlowPhysiology(
        systolic_pressure_mmhg=120.0, diastolic_pressure_mmhg=81.0
    )
    assert physiology.mean_pressure_mmhg == pytest.approx(94.0)
    assert AortaFlowPhysiology().mean_pressure_mmhg == pytest.approx(78.0 + 34.0 / 3.0)


def whole_body_pressure(
    physiology: AortaFlowPhysiology, flux: np.ndarray, interval_s: float, dt_s: float
) -> tuple[np.ndarray, float]:
    """Step the whole-body Windkessel over one inflow period from beat start.

    Returns the outlet pressure in mmHg and the final capacitor pressure in Pa.
    """
    steps = round(len(flux) * interval_s / dt_s)
    times = (
        AortaFlowPhysiology.beat_start_s(flux, interval_s)
        + np.arange(1, steps + 1) * dt_s
    )
    flow = physiology.inlet_flow(times, flux, interval_s)[0]
    windkessel = Windkessel.from_total(
        1.0,
        physiology.total_resistance_pa_s_m3,
        physiology.total_compliance_m3_pa,
        physiology.proximal_fraction,
        physiology.initial_pressure_pa,
    )
    pressure = [windkessel.advance(float(q), dt_s) for q in flow]
    return np.asarray(pressure) / AortaFlowPhysiology.MMHG_PA, windkessel.pressure


@pytest.mark.parametrize("dt_s", [0.001, 0.005])
def test_calibrated_reproduces_input_pressures(dt_s: float) -> None:
    """The calibrated 0D model has the input extremes and is periodic."""
    physiology = AortaFlowPhysiology(
        systolic_pressure_mmhg=124.0, diastolic_pressure_mmhg=82.0
    )
    flux, interval_s = physiology.lv_inflow(lv_volume_curve())
    calibrated = physiology.calibrated(flux, interval_s, dt_s)
    pressure, end_pa = whole_body_pressure(calibrated, flux, interval_s, dt_s)
    assert pressure.max() == pytest.approx(124.0, abs=0.1)
    assert pressure.min() == pytest.approx(82.0, abs=0.1)
    # The mean is R times the mean inflow, not the one-third rule's estimate.
    assert pressure.mean() == pytest.approx(
        calibrated.total_resistance_pa_s_m3 * flux.mean() / AortaFlowPhysiology.MMHG_PA,
        abs=0.1,
    )
    assert end_pa / AortaFlowPhysiology.MMHG_PA == pytest.approx(
        calibrated.initial_pressure_pa / AortaFlowPhysiology.MMHG_PA, abs=1e-6
    )
    # Only the outlet totals and the initial pressure change, on a copy.
    assert physiology.total_resistance_pa_s_m3 == 3.33e8
    assert calibrated.systolic_pressure_mmhg == 124.0
    assert calibrated.proximal_fraction == physiology.proximal_fraction
    assert calibrated.flow_fractions == physiology.flow_fractions


def test_calibrated_scales_with_the_inflow() -> None:
    """Half the inflow needs twice the resistance and half the compliance."""
    physiology = AortaFlowPhysiology()
    flux, interval_s = physiology.lv_inflow(lv_volume_curve())
    full = physiology.calibrated(flux, interval_s)
    half = physiology.calibrated(0.5 * flux, interval_s)
    assert half.total_resistance_pa_s_m3 == pytest.approx(
        2.0 * full.total_resistance_pa_s_m3, rel=1e-6
    )
    assert half.total_compliance_m3_pa == pytest.approx(
        0.5 * full.total_compliance_m3_pa, rel=1e-6
    )
    assert half.initial_pressure_pa == pytest.approx(full.initial_pressure_pa)


def test_calibrated_rejects_unreachable_inputs() -> None:
    """Unordered pressures and unreachable pressures are errors."""
    flux, interval_s = AortaFlowPhysiology().lv_inflow(lv_volume_curve())
    with pytest.raises(ValueError, match="diastolic < systolic"):
        AortaFlowPhysiology(
            systolic_pressure_mmhg=80.0, diastolic_pressure_mmhg=120.0
        ).calibrated(flux, interval_s)
    # Half the resistance proximal gives far more than a 2 mmHg pulse.
    with pytest.raises(ValueError, match="No outlet resistance"):
        AortaFlowPhysiology(
            systolic_pressure_mmhg=92.0,
            diastolic_pressure_mmhg=90.0,
            proximal_fraction=0.5,
        ).calibrated(flux, interval_s)


def test_calibrated_with_wall_compliance() -> None:
    """With a wall in front, the replayed 0D model has the input pressures.

    The replay solves each backward-Euler step of ``C_w dP/dt = Q_in - Q``,
    ``P = R_p Q + P_d`` and ``C dP_d/dt = Q - P_d / R_d`` as one linear
    system, over 30 periods from the calibrated capacitor pressure; the
    last period's extremes are the input pressures within 0.1 mmHg.
    """
    physiology = AortaFlowPhysiology(
        systolic_pressure_mmhg=124.0, diastolic_pressure_mmhg=82.0
    )
    flux, interval_s = physiology.lv_inflow(lv_volume_curve())
    dt_s = 0.0025
    rigid = physiology.calibrated(flux, interval_s, dt_s)
    wall_c = 0.3 * rigid.total_compliance_m3_pa
    elastic = physiology.calibrated(
        flux, interval_s, dt_s, wall_compliance_m3_pa=wall_c
    )
    # The wall in front stores more of the stroke than its share of the
    # rigid total, so the outlets need less than the rest.
    assert elastic.total_compliance_m3_pa < rigid.total_compliance_m3_pa - wall_c
    assert physiology.calibrated(
        flux, interval_s, dt_s, wall_compliance_m3_pa=1e-18
    ).total_compliance_m3_pa == pytest.approx(rigid.total_compliance_m3_pa, rel=1e-6)

    period_s = len(flux) * interval_s
    steps = round(period_s / dt_s)
    dt_s = period_s / steps
    times = (
        AortaFlowPhysiology.beat_start_s(flux, interval_s)
        + np.arange(1, 30 * steps + 1) * dt_s
    )
    inflow = physiology.inlet_flow(times, flux, interval_s)[0]
    proximal = elastic.proximal_fraction * elastic.total_resistance_pa_s_m3
    distal = elastic.total_resistance_pa_s_m3 - proximal
    compliance = elastic.total_compliance_m3_pa
    root = distal_pa = elastic.initial_pressure_pa
    pressure = []
    for q_in in inflow:
        # Unknowns (P, P_d, Q).
        matrix = np.array(
            [
                [wall_c / dt_s, 0.0, 1.0],
                [1.0, -1.0, -proximal],
                [0.0, compliance / dt_s + 1.0 / distal, -1.0],
            ]
        )
        rhs = np.array(
            [wall_c / dt_s * root + q_in, 0.0, compliance / dt_s * distal_pa]
        )
        root, distal_pa, _ = np.linalg.solve(matrix, rhs)
        pressure.append(root)
    last = np.asarray(pressure[-steps:]) / AortaFlowPhysiology.MMHG_PA
    assert last.max() == pytest.approx(124.0, abs=0.1)
    assert last.min() == pytest.approx(82.0, abs=0.1)
