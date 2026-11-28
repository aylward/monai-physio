"""
Tutorial 20: 3D Rigid-Wall Blood Flow in the Tutorial 13 Aorta

Purpose
-------
Repeat Tutorial 19 in three dimensions: the same aorta, the same inflow and
the same outlets, but the full velocity and pressure fields of the lumen
instead of section means along its centerline.  The wall is still rigid;
Tutorial 21 lets it move.

1. **Geometry.** :class:`monai_physio.aorta.AortaProcessGeometry` (called by the
   workflow) cuts the aorta of the merged chest labelmap normal to its
   centerline 10 mm inside each end and opens the brachiocephalic trunk,
   left common carotid and left subclavian arteries as outlets, exactly as
   in Tutorial 19.  Netgen then fills the lumen alone with tetrahedra about
   2 mm across, refined to eight elements across each narrow outlet.
2. **Patient inputs.** As in Tutorial 19, set in one block below: the
   patient's heart rate and systolic and diastolic blood pressure (for
   example a cuff measurement; the example 112/78 mmHg are inputs, not
   results) and the CT-derived inflow, the left-ventricle (LV) volume loss
   of Tutorial 13's hundred warped labelmaps retimed to the heart rate.  The
   inflow enters through the root with a blunt profile.
3. **Outlets.** As in Tutorial 19, each outlet ends in a three-element
   Windkessel from :class:`AortaFlowPhysiology`, whose totals the workflow
   calibrates to the input pressures and this inflow
   (:meth:`AortaFlowPhysiology.calibrated`).  In 3D the flow split is
   computed: each outlet takes what its Windkessel and the lumen let
   through.
4. **Flow.** :class:`monai_physio.aorta.AortaWorkflowSimulateFlow3D` solves the
   incompressible Navier-Stokes equations with quadratic velocity and linear
   pressure on the GPU (NVIDIA Warp FEM, cuDSS preconditioner), 5 ms steps
   for three beats, and keeps the fields of the last beat.

Tutorial 13's heart ejects about 32 mL per beat over about 0.5 s (ejection
fraction about 44%, 1.9 L/min, under two fifths of a resting adult's
output), so the peak inflow, about 140 mL/s, and the speeds are below adult
values, and the outlet resistance fitted to adult pressures is more than
twice an adult's; see Tutorial 19.

The model is educational: a rigid wall, a Newtonian fluid and outlet
parameters assumed rather than measured; time and mesh convergence are not
established.  Expect simulated root pressures within a few mmHg of the
input pressures, root pressures and flow shares within a few mmHg and a
fraction of a percent of Tutorial 19's, and peak speeds in the arch of
about 0.5-1 m/s; the small branch caps are faster.

Prerequisites
-------------
Tutorial 13, whose output this tutorial reads.  A CUDA GPU and NVIDIA cuDSS
(the ``nvidia-cudss-cu13`` or ``-cu12`` wheel, or ``CUDSS_LIBRARY`` set to
the library file).  Tutorial 19 is optional; when its ``summary.json`` is
present the two models are compared.

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
Under ``output/tutorial_20_simulate_aorta_flow_3d_chest_3d_ct/``, a
:class:`monai_physio.VisualizeFlowFields` results folder:

  * ``mesh.npz``, ``time.npz``, ``velocity_m_s.npy``, ``pressure_pa.npy``
    - the lumen mesh and the last beat's fields
  * ``steps.npz``          - root, outlet and solver series of every step
  * ``run.json``           - solver settings and timings
  * ``summary.json``       - last-beat pressures, flows and speeds
  * ``inflow.json``        - LV volume per frame and the derived inflow
  * ``comparison_1d.json`` - differences from Tutorial 19, when present
  * ``pressure_flow.png``  - root and distal pressure and the outlet flows
  * ``peak_systole_pressure.png`` / ``peak_systole_speed.png`` - the fields
    at peak inflow, with a slice and velocity arrows
  * ``pressure_last_beat.mp4`` - the last beat's pressure (``save_movie``)

To explore the fields interactively::

  from pathlib import Path
  from monai_physio.visualize_flow_fields import VisualizeFlowFields, create_app
  viewer = VisualizeFlowFields(Path(OUTPUT_DIR), off_screen=False)
  create_app(viewer, "Tutorial 20", []).start(port=8702)
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
from monai_physio.aorta import AortaFlowPhysiology, AortaWorkflowSimulateFlow3D
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

# Summary values compared with Tutorial 19's 1D model.
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
    class_name = "tutorial_20_simulate_aorta_flow_3d_chest_3d_ct"
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
    tutorial_19_summary = (
        paths.output_directory(test_mode)
        / "tutorial_19_simulate_aorta_flow_1d_chest_3d_ct"
        / "summary.json"
    )
    output_dir = paths.output_directory(test_mode) / class_name

    # ---- Parameters --------------------------------------------------------
    # Heart beats simulated; the summary reports the last one.  The outlets
    # start at their calibrated periodic pressure, so the beats barely drift.
    beats = 3
    # Solver step, lumen element size and the interval of the saved fields.
    dt_s = 0.005
    element_size_mm = 2.0
    frame_interval_s = 0.03
    # Record the last beat's pressure as an MP4 (needs imageio-ffmpeg).
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
    # Windkessels are calibrated to them.
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
    workflow = AortaWorkflowSimulateFlow3D(physiology, log_level=log_level)
    result = workflow.process(
        itk.imread(str(labelmap_file)),
        aorta_label_ids=(AORTA_LABEL_ID,),
        branch_label_ids=BRANCH_LABEL_IDS,
        inflow_m3_s=inflow_m3_s,
        inflow_interval_s=interval_s,
        beats=beats,
        dt_s=dt_s,
        element_size_mm=element_size_mm,
        frame_interval_s=frame_interval_s,
    )
    saved = AortaWorkflowSimulateFlow3D.save_results(result, str(output_dir))
    plot_file = AortaWorkflowSimulateFlow3D.save_plots(result, str(output_dir))
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
        "stroke volume %.1f mL; peak speed %.2f m/s more than %.0f mm from "
        "the outlets",
        summary["input_systolic_mmhg"],
        summary["input_diastolic_mmhg"],
        summary["input_mean_mmhg"],
        summary["root_systolic_mmhg"],
        summary["root_diastolic_mmhg"],
        summary["root_mean_mmhg"],
        summary["mean_pressure_change_mmhg"],
        summary["stroke_volume_ml"],
        summary["peak_speed_away_from_outlets_m_s"],
        AortaWorkflowSimulateFlow3D.OUTLET_EXCLUSION_MM,
    )

    # ---- Compare with Tutorial 19 ------------------------------------------
    if tutorial_19_summary.exists():
        with open(tutorial_19_summary, encoding="utf-8") as file:
            summary_1d = json.load(file)
        comparison = {
            key: {
                "1d": summary_1d[key],
                "3d": summary[key],
                "3d_minus_1d": summary[key] - summary_1d[key],
            }
            for key in COMPARED_KEYS
        }
        comparison["outlet_flow_shares"] = {
            name: {
                "1d": summary_1d["outlet_flow_shares"][name],
                "3d": share,
                "3d_minus_1d": share - summary_1d["outlet_flow_shares"][name],
            }
            for name, share in summary["outlet_flow_shares"].items()
        }
        with open(output_dir / "comparison_1d.json", "w", encoding="utf-8") as file:
            json.dump(comparison, file, indent=2)
        for key in COMPARED_KEYS:
            logger.log_info(
                "%-26s 1D %8.2f  3D %8.2f  3D-1D %+7.2f",
                key,
                comparison[key]["1d"],
                comparison[key]["3d"],
                comparison[key]["3d_minus_1d"],
            )
        for name, values in comparison["outlet_flow_shares"].items():
            logger.log_info(
                "%-26s 1D %7.2f%%  3D %7.2f%%  3D-1D %+6.2f points",
                f"flow share {name}",
                100 * values["1d"],
                100 * values["3d"],
                100 * values["3d_minus_1d"],
            )
    else:
        logger.log_info("No Tutorial 19 summary at %s to compare", tutorial_19_summary)

    # ---- Peak-systole views ------------------------------------------------
    # The viewer opens at the saved frame of peak inflow.  The slice is the
    # coronal plane through the root center, along the ascending aorta.  The
    # camera starts anterior, with the patient's left on the right of the
    # image, and turns 30 degrees about the vertical to open the arch.
    viewer = VisualizeFlowFields(
        output_dir,
        caption=(
            "Rigid wall; P2/P1 Warp FEM; RCR outlets\n"
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
        ):
            viewer.set_field(field)
            viewer.screenshot(output_dir / filename)
            view_files.append(output_dir / filename)
        logger.log_info("Peak-systole view: %s", viewer.snapshot())
        if save_movie:
            viewer.set_field("pressure")
            movie = viewer.record_movie(output_dir / "pressure_last_beat.mp4")
            logger.log_info("Movie: %s", movie)
    finally:
        viewer.plotter.close()
    logger.log_info("Tutorial 20 took %.1f s", time.perf_counter() - started)

    # Testing: the pressure and flow curves and the peak-systole views.
    screenshots = [Path(plot_file), *view_files]

    tutorial_results = {
        "summary": summary,
        "files": {**saved, "plot": plot_file},
        "screenshots": screenshots,
    }
