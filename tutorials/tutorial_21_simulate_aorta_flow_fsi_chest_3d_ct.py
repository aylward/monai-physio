"""
Tutorial 21: 3D Blood Flow With a Moving Aortic Wall in the Tutorial 13 Aorta

Purpose
-------
Repeat Tutorial 20 with an elastic wall: the blood pressure pushes the aortic
wall outward and the moving wall changes the blood flow (two-way
fluid-structure interaction, FSI).  The aorta, inflow and patient inputs are
those of Tutorials 19 and 20.

1. **Geometry.** :class:`monai_physio.aorta.AortaProcessGeometry` (called by the
   workflow) cuts the aorta and opens its three arch branches as in
   Tutorials 19 and 20, on a 1.25 mm grid, and adds a wall shell around the
   lumen.  Netgen fills the lumen and the wall together so that they share
   every interface triangle.
2. **Patient inputs.** As in Tutorials 19 and 20, set in one block below:
   the patient's heart rate and systolic and diastolic blood pressure (the
   example 112/78 mmHg are inputs, not results) and the CT-derived inflow,
   the left-ventricle (LV) volume loss of Tutorial 13's hundred warped
   labelmaps retimed to the heart rate.
3. **Wall.** A Neo-Hookean wall, by default 1.5 mm thick with Young's
   modulus 1 MPa (:class:`AortaFlowPhysiology`), prestressed so that the
   imaged aorta is its shape at the calibrated pressure where the
   simulation starts, clamped around every outlet and held by weak springs
   standing in for the surrounding tissue.  Netgen needs a shell at
   least three grid spacings thick, so on the 1.25 mm grid the wall is
   meshed 3.75 mm thick at 0.4 MPa, the same modulus times thickness.
4. **Outlets.** As in Tutorials 19 and 20 the outlets end in Windkessels
   calibrated to the input pressures, except that the wall now stores part
   of each stroke in front of them: the workflow measures the wall's
   compliance with a static wall solve (prestressed at the diastolic and
   loaded with the systolic input) and calibrates the outlets with it (much
   less outlet compliance than in Tutorials 19 and 20).
5. **Flow and wall.** :class:`monai_physio.aorta.AortaWorkflowSimulateFlowFSI`
   couples the 3D blood flow of Tutorial 20 to the wall on the GPU (NVIDIA
   Warp FEM, cuDSS), 2.5 ms steps for three beats, iterating blood and wall to
   agreement every step, and keeps the fields and wall displacement of the
   last beat.

Tutorial 13's heart ejects about 32 mL per beat over about 0.5 s (ejection
fraction about 44%, 1.9 L/min, under two fifths of a resting adult's
output), so the peak inflow, about 140 mL/s, and the speeds are below adult
values, and the outlet resistance fitted to adult pressures is more than
twice an adult's; see Tutorial 19.

The model is educational: a massless wall of uniform thickness and
stiffness, a Newtonian fluid and outlet parameters assumed rather than
measured; time and mesh convergence are not established.  The 3.75 mm,
0.4 MPa surrogate wall is somewhat more compliant than the 1.5 mm, 1 MPa
wall it stands for (keeping modulus times thickness is a thin-wall
approximation), and the wall motion, strain, distensibility and wave speed
reported are the surrogate's.  Expect the root diastolic pressure near the
input and the systolic a few mmHg below it (about 3 mmHg in a reference
run).  Most of that is the lumen being more compliant in the flow than in
the static solve its compliance is measured with (the summary's
``lumen_compliance_m3_pa``, 2.47e-9 m^3/Pa in the reference run, against
the static 2.17e-9 m^3/Pa); the run also starts at the outlet capacitor
pressure (89.5 mmHg) rather than the periodic root pressure of the lumped
model at that phase (92.1 mmHg).  The mean pressure rises by about
0.5 mmHg per beat: the lumped model's own transient plus the three slightly
unequal beats of the inflow (31.67, 31.96 and 31.89 mL), not numerical
leakage, since each beat's volume balance closes within about 0.01 mL.
With the wall in front of the outlets the pressure waveform is rounder, so
the same systolic and diastolic pressures come with a higher mean than in
Tutorials 19 and 20 (about 94 against 92 mmHg).  Expect a wall that bulges about half a millimetre, a lumen
volume change of several percent over the beat and a pulse-wave speed of a
few metres per second.  The run takes about 45-60 minutes.

The blood and wall agree in every step within the coupling tolerances of
:attr:`AortaWorkflowSimulateFlowFSI.COUPLING`.  Its absolute tolerance,
10 nm, is looser than the FSI experiment's 1 nm: the interface moves only
micrometres per step, so the absolute term decides convergence in nearly
every step, and on a 1 mm netgen mesh of this aorta the float32 blood solve
held the residual near 5-10 nm, so 1 nm capped many steps.  With 10 nm the
inflow minus the outflow and the lumen's volume change stays within about
0.4% of the peak inflow; the summary reports the largest RMS coupling
residual (``max_coupling_residual_m``).

Prerequisites
-------------
Tutorial 13, whose output this tutorial reads.  A CUDA GPU and NVIDIA cuDSS
(the ``nvidia-cudss-cu13`` or ``-cu12`` wheel, or ``CUDSS_LIBRARY`` set to
the library file).  Tutorials 19 and 20 are optional; when their
``summary.json`` files are present the models are compared.

Data Required
-------------
Requires the ``[aorta]`` extra (``monai-physio[aorta]``).

Under ``output/tutorial_13_combine_heart_and_lung_motion_chest_3d_ct/``:

  * ``chest_ct_merged_labelmap.mha`` - the aorta and its branches
  * ``combined_labelmap_000.mha`` ... ``combined_labelmap_099.mha`` - the
    warped labelmaps whose LV volume gives the inflow

When run as a test, the reference LV volume curve
:meth:`monai_physio.aorta.AortaFlowPhysiology.reference_lv_volumes_ml`
replaces theirs: the test-mode Tutorial 13 heart, from a 3 mm copy of the
scan and a briefly trained network, barely moves.  The aorta is still the
test-mode Tutorial 13 one.

Outputs
-------
Under ``output/tutorial_21_simulate_aorta_flow_fsi_chest_3d_ct/``, a
:class:`monai_physio.VisualizeFlowFields` results folder:

  * ``mesh.npz``, ``time.npz``, ``velocity_m_s.npy``, ``pressure_pa.npy``,
    ``displacement_m.npy`` - the lumen mesh and the last beat's fields
  * ``steps.npz``          - root, outlet, wall and solver series of every
    step
  * ``run.json``           - solver settings, compliances and timings
  * ``summary.json``       - last-beat pressures, flows, speeds and wall
    motion
  * ``inflow.json``        - LV volume per frame and the derived inflow
  * ``comparison.json``    - differences from Tutorials 19 and 20, when
    present
  * ``pressure_flow.png``  - root and distal pressure, the flows and the wall
    motion
  * ``peak_systole_pressure.png`` / ``peak_systole_speed.png`` /
    ``peak_systole_displacement.png`` - the fields at peak inflow
  * ``displacement_last_beat.mp4`` - the last beat's wall displacement
    (``save_movie``)

To explore the fields interactively::

  from pathlib import Path
  from monai_physio.visualize_flow_fields import VisualizeFlowFields, create_app
  viewer = VisualizeFlowFields(Path(OUTPUT_DIR), off_screen=False)
  create_app(viewer, "Tutorial 21", []).start(port=8703)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import itk
import numpy as np
from parameters_base import ParametersBase

from monai_physio import MONAIPhysioBase, ProcessTests
from monai_physio.aorta import AortaFlowPhysiology, AortaWorkflowSimulateFlowFSI
from monai_physio.visualize_flow_fields import VisualizeFlowFields

# Labels of Tutorial 13's merged labelmap: the NV-Segment CT/MRI ids its
# thorax segmentation uses, with the Simpleware left ventricle written under
# NV-Segment's ventricle_left id.
AORTA_LABEL_ID = 6
BRANCH_LABEL_IDS = {
    "BCT": 109,  # brachiocephalic_trunk
    "LCCA": 112,  # common_carotid_artery_left
    "LSA": 123,  # subclavian_artery_left
}
LV_LABEL_ID = 151  # ventricle_left

# Summary values compared with Tutorials 19 and 20.
COMPARED_KEYS = (
    "root_systolic_mmhg",
    "root_diastolic_mmhg",
    "root_mean_mmhg",
    "root_pulse_pressure_mmhg",
    "root_to_distal_max_mmhg",
    "root_to_distal_min_mmhg",
    "stroke_volume_ml",
)

if __name__ == "__main__":
    log_level = logging.INFO
    class_name = "tutorial_21_simulate_aorta_flow_fsi_chest_3d_ct"
    logger = MONAIPhysioBase(class_name=class_name, log_level=log_level)

    # ---- Inputs ------------------------------------------------------------
    test_mode = ProcessTests.running_as_test()
    paths = ParametersBase()
    tutorial_13_dir = (
        paths.output_directory(test_mode)
        / "tutorial_13_combine_heart_and_lung_motion_chest_3d_ct"
    )
    labelmap_file = tutorial_13_dir / "chest_ct_merged_labelmap.mha"
    frame_files = sorted(tutorial_13_dir.glob("combined_labelmap_*.mha"))
    earlier_summaries = {
        "1d": paths.output_directory(test_mode)
        / "tutorial_19_simulate_aorta_flow_1d_chest_3d_ct"
        / "summary.json",
        "3d": paths.output_directory(test_mode)
        / "tutorial_20_simulate_aorta_flow_3d_chest_3d_ct"
        / "summary.json",
    }
    output_dir = paths.output_directory(test_mode) / class_name

    # ---- Parameters --------------------------------------------------------
    # Heart beats simulated; the summary reports the last one, the same
    # LV beat as Tutorials 19 and 20 report.
    beats = 3
    # Solver step, grid spacing (which sets the mesh size), largest interior
    # element and the interval of the saved fields.
    dt_s = 0.0025
    spacing_mm = 1.25
    element_size_mm = 2.0
    frame_interval_s = 0.03
    # Record the last beat's wall displacement as an MP4 (needs
    # imageio-ffmpeg).
    save_movie = True

    if not labelmap_file.exists() or not frame_files:
        raise FileNotFoundError(
            f"Tutorial 13 output not found in {tutorial_13_dir}\n"
            "Run tutorial_13_combine_heart_and_lung_motion_chest_3d_ct.py first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # ---- Per-patient inputs ------------------------------------------------
    # The patient's heart rate and blood pressure (e.g. a cuff measurement
    # with the CT) are user inputs, not simulation results; these are example
    # values (typical adult rest: 60-100 bpm, about 120/80 mmHg).  The outlet
    # Windkessels are calibrated to them.  The wall keeps the defaults of
    # AortaFlowPhysiology (young_modulus_pa, poisson_ratio, wall_thickness_mm,
    # tether_stiffness_pa_m).
    physiology = AortaFlowPhysiology(
        heart_rate_bpm=60.0,
        systolic_pressure_mmhg=112.0,
        diastolic_pressure_mmhg=78.0,
    )
    # The aortic inflow from the CT: the LV volume loss of Tutorial 13's
    # labelmaps, read one frame at a time by the generator (a hundred
    # full-resolution labelmaps do not fit in memory together).
    # A test run's Tutorial 13 heart, from a 3 mm copy of the scan and a
    # briefly trained network, barely moves (about 2 mL), so a reference
    # curve of this heart's LV volume stands in for it.
    if test_mode:
        logger.log_info(
            "Test mode: the reference LV volume curve replaces Tutorial 13's, "
            "whose test-mode heart barely moves"
        )
        lv_volume_ml = AortaFlowPhysiology.reference_lv_volumes_ml()
    else:
        lv_volume_ml = AortaFlowPhysiology.lv_volumes_ml(
            (itk.imread(str(frame)) for frame in frame_files), [LV_LABEL_ID]
        )
    inflow_m3_s, interval_s = physiology.lv_inflow(lv_volume_ml)
    cardiac_output_l_min = float(inflow_m3_s.mean()) * 6.0e4
    logger.log_info(
        "Patient inputs: blood pressure %.0f/%.0f mmHg, heart rate %.0f bpm; "
        "LV volume %.1f-%.1f mL over %d frames, %.3f s per frame, "
        "cardiac output %.2f L/min",
        physiology.systolic_pressure_mmhg,
        physiology.diastolic_pressure_mmhg,
        physiology.heart_rate_bpm,
        lv_volume_ml.min(),
        lv_volume_ml.max(),
        len(lv_volume_ml),
        interval_s,
        cardiac_output_l_min,
    )

    # ---- Simulate ----------------------------------------------------------
    workflow = AortaWorkflowSimulateFlowFSI(physiology, log_level=log_level)
    result = workflow.process(
        itk.imread(str(labelmap_file)),
        aorta_label_ids=(AORTA_LABEL_ID,),
        branch_label_ids=BRANCH_LABEL_IDS,
        inflow_m3_s=inflow_m3_s,
        inflow_interval_s=interval_s,
        beats=beats,
        dt_s=dt_s,
        spacing_mm=spacing_mm,
        element_size_mm=element_size_mm,
        frame_interval_s=frame_interval_s,
    )
    saved = AortaWorkflowSimulateFlowFSI.save_results(result, str(output_dir))
    plot_file = AortaWorkflowSimulateFlowFSI.save_plots(result, str(output_dir))
    with open(output_dir / "inflow.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "lv_volume_ml": lv_volume_ml.tolist(),
                "interval_s": interval_s,
                "inflow_m3_s": inflow_m3_s.tolist(),
                "cardiac_output_l_min": cardiac_output_l_min,
            },
            file,
            indent=2,
        )

    summary = result["summary"]
    logger.log_info(
        "Input pressure %.0f/%.0f mmHg (mean %.1f); simulated root pressure "
        "%.1f/%.1f mmHg (mean %.1f, %+.2f mmHg from the beat before); "
        "compliance total %.3g = wall %.3g + outlets %.3g m^3/Pa",
        summary["input_systolic_mmhg"],
        summary["input_diastolic_mmhg"],
        summary["input_mean_mmhg"],
        summary["root_systolic_mmhg"],
        summary["root_diastolic_mmhg"],
        summary["root_mean_mmhg"],
        summary["mean_pressure_change_mmhg"],
        summary["total_compliance_m3_pa"],
        summary["wall_compliance_m3_pa"],
        summary["outlet_total_compliance_m3_pa"],
    )
    logger.log_info(
        "Wall: peak displacement %.2f mm (outward %.2f mm), lumen volume "
        "strain %.1f%%, diameter strain %.1f%%, distensibility %.2e /mmHg, "
        "Bramwell-Hill wave speed %.1f m/s; coupling %.1f iterations per "
        "step, %d unconverged steps",
        summary["peak_wall_displacement_mm"],
        summary["peak_outward_wall_displacement_mm"],
        summary["volume_strain_percent"],
        summary["diameter_strain_percent"],
        summary["distensibility_per_mmhg"],
        summary["bramwell_hill_wave_speed_m_s"],
        summary["mean_coupling_iterations"],
        summary["unconverged_coupling_steps"],
    )

    # ---- Compare with Tutorials 19 and 20 ----------------------------------
    comparison: dict[str, dict[str, dict[str, float]]] = {}
    for model, summary_file in earlier_summaries.items():
        if not summary_file.exists():
            logger.log_info("No %s summary at %s to compare", model, summary_file)
            continue
        with open(summary_file, encoding="utf-8") as file:
            earlier = json.load(file)
        comparison[model] = {
            key: {model: earlier[key], "fsi": summary[key]} for key in COMPARED_KEYS
        }
        comparison[model].update(
            {
                f"flow share {name}": {
                    model: earlier["outlet_flow_shares"][name],
                    "fsi": share,
                }
                for name, share in summary["outlet_flow_shares"].items()
            }
        )
        for key, values in comparison[model].items():
            values[f"fsi_minus_{model}"] = values["fsi"] - values[model]
            logger.log_info(
                "%-26s %s %8.3f  FSI %8.3f  FSI-%s %+7.3f",
                key,
                model.upper(),
                values[model],
                values["fsi"],
                model.upper(),
                values[f"fsi_minus_{model}"],
            )
    if comparison:
        with open(output_dir / "comparison.json", "w", encoding="utf-8") as file:
            json.dump(comparison, file, indent=2)

    # ---- Peak-systole views ------------------------------------------------
    # As in Tutorial 20: the viewer opens at the saved frame of peak inflow,
    # the slice is the coronal plane through the root center and the camera
    # starts anterior, turned 30 degrees about the vertical.  The mesh is
    # shown moved by the wall displacement at true scale.
    viewer = VisualizeFlowFields(
        output_dir,
        caption=(
            "Neo-Hookean wall; P2/P1 Warp FEM; RCR outlets; true-scale motion\n"
            "Demonstration: time and mesh convergence not established"
        ),
    )
    try:
        root_y_m = result["geometry"]["ports"]["root"]["center_mm"][1] * 1e-3
        low_m, high_m = viewer.grid.bounds[2:4]
        viewer.set_slice("y", (root_y_m - low_m) / (high_m - low_m))
        viewer.set_vectors(True)
        center = np.asarray(viewer.grid.center)
        viewer.plotter.camera_position = [
            tuple(center - [0.0, 1.0, 0.0]),
            tuple(center),
            (0.0, 0.0, 1.0),
        ]
        viewer.plotter.reset_camera()
        viewer.rotate_camera(30, 15)
        view_files = []
        for field, filename in (
            ("pressure", "peak_systole_pressure.png"),
            ("velocity", "peak_systole_speed.png"),
            ("displacement", "peak_systole_displacement.png"),
        ):
            viewer.set_field(field)
            viewer.screenshot(output_dir / filename)
            view_files.append(output_dir / filename)
        logger.log_info("Peak-systole view: %s", viewer.snapshot())
        if save_movie:
            viewer.set_field("displacement")
            movie = viewer.record_movie(output_dir / "displacement_last_beat.mp4")
            logger.log_info("Movie: %s", movie)
    finally:
        viewer.plotter.close()
    logger.log_info("Tutorial 21 took %.1f s", time.perf_counter() - started)

    # Testing: the pressure, flow and wall curves and the peak-systole views.
    screenshots = [Path(plot_file), *view_files]

    tutorial_results = {
        "summary": summary,
        "files": {**saved, "plot": plot_file},
        "screenshots": screenshots,
    }
