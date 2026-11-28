"""
Tutorial 19: 1D Rigid-Wall Blood Flow in the Tutorial 13 Aorta

Purpose
-------
Simulate the pressure and flow of the aorta Tutorial 13 animated, driven by
the heart Tutorial 13 animated, with the simplest model that still has a
pulse: a rigid tube along the aorta's centerline.  It runs in seconds once
the inputs are read, and Tutorials 20 and 21 repeat it in 3D with a rigid and
an elastic wall.

1. **Geometry.** :class:`monai_physio.aorta.AortaProcessGeometry` (called by the
   workflow) cuts the aorta of the merged chest labelmap normal to its
   centerline 10 mm inside each end, finds where the brachiocephalic trunk,
   left common carotid and left subclavian arteries leave it, and measures the
   lumen area of every section along the centerline.
2. **Patient inputs.** The simulation takes three per-patient inputs, set
   in one block below: the patient's heart rate and systolic and diastolic
   blood pressure, for example a cuff measurement taken with the CT, and
   the aortic inflow derived from the CT.  The left-ventricle (LV) volume of
   each of Tutorial 13's hundred warped labelmaps is counted; with an ideal
   aortic valve the aortic inflow is the LV's volume loss while it empties.
   The series holds three beats and is retimed to the heart rate
   (:meth:`AortaFlowPhysiology.lv_inflow`).  The example pressures,
   112/78 mmHg, are not simulation results: replace them with your
   patient's.
3. **Outlets.** Each outlet ends in a three-element Windkessel standing for
   the vessels beyond it.  The workflow calibrates their total resistance
   and compliance so that this inflow reproduces the input systolic and
   diastolic pressures in a whole-body Windkessel, and starts them at its
   periodic pressure
   (:meth:`AortaFlowPhysiology.calibrated`).
4. **Flow.** :class:`monai_physio.aorta.AortaWorkflowSimulateFlow1D` imposes the
   outlet flow split (70% distal, 17% BCT, 8% LSA, 5% LCCA) along the
   centerline and evaluates the inertial and Poiseuille pressure drop on the
   GPU with NVIDIA Warp.

This heart is not a typical resting adult's.  Tutorial 13's LV labelmaps
give an ejection fraction of about 44% (72 to 40 mL), a stroke volume of
about 32 mL and, at 60 bpm, a cardiac output of 1.9 L/min, under two fifths
of a resting adult's (the experiments' USD LV mesh of the same heart gives
36 mL).  Ejection is long, about 0.5 s, so the peak inflow, about 140 mL/s,
and the aortic speeds, about 0.55 m/s, are well below adult values (several
hundred mL/s and about 1 m/s).  Adult pressures with this low flow need a
total outlet resistance of about 3.8e8 Pa s/m^3, more than twice a resting
adult's (about 1.65e8), and its proximal 6% then carries most of the pulse:
the proximal resistance times the peak inflow is about 24 of the 33 mmHg,
so the root pressure follows the shape of the inflow.

The model is educational: no wall motion, no convective acceleration and a
flow split fixed by assumption rather than computed.  Expect a simulated root
pressure within a few mmHg of the input pressure (it adds the pressure drop
along the aorta), section-mean velocities around half a meter per second and
a root-to-distal difference of a few mmHg.

Prerequisites
-------------
Tutorial 13, whose output this tutorial reads.  A CUDA GPU.

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
Under ``output/tutorial_19_simulate_aorta_flow_1d_chest_3d_ct/``:

  * ``centerline.vtp``     - centerline with last-beat pressure and velocity
  * ``lumen_surface.vtp``  - the lumen surface tagged by port
  * ``fields.npz``         - every time x section field
  * ``summary.json``       - last-beat pressures, flows and velocities
  * ``inflow.json``        - LV volume per frame and the derived inflow
  * ``pressure_flow.png``  - root and distal pressure and the outlet flows
  * ``lumen_surface.png``  - the prepared lumen
"""

from __future__ import annotations

import json
import logging
import time

import itk
from parameters_base import ParametersBase

from monai_physio import MONAIPhysioBase, ProcessTests
from monai_physio.aorta import AortaFlowPhysiology, AortaWorkflowSimulateFlow1D

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

if __name__ == "__main__":
    log_level = logging.INFO
    class_name = "tutorial_19_simulate_aorta_flow_1d_chest_3d_ct"
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
    output_dir = paths.output_directory(test_mode) / class_name

    # ---- Parameters --------------------------------------------------------
    # Heart beats simulated; the summary reports the last one.  The outlets
    # start at their calibrated periodic pressure, so the beats barely drift.
    beats = 3

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
    workflow = AortaWorkflowSimulateFlow1D(physiology, log_level=log_level)
    result = workflow.process(
        itk.imread(str(labelmap_file)),
        aorta_label_ids=(AORTA_LABEL_ID,),
        branch_label_ids=BRANCH_LABEL_IDS,
        inflow_m3_s=inflow_m3_s,
        inflow_interval_s=interval_s,
        beats=beats,
    )
    saved = AortaWorkflowSimulateFlow1D.save_results(result, str(output_dir))
    plot_file = AortaWorkflowSimulateFlow1D.save_plots(result, str(output_dir))
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
    logger.log_info("Tutorial 19 took %.1f s", time.perf_counter() - started)

    summary = result["summary"]
    logger.log_info(
        "Input pressure %.0f/%.0f mmHg (mean %.1f); simulated root pressure "
        "%.1f/%.1f mmHg (mean %.1f, %+.2f mmHg from the beat before); "
        "stroke volume %.1f mL; peak velocity %.2f m/s",
        summary["input_systolic_mmhg"],
        summary["input_diastolic_mmhg"],
        summary["input_mean_mmhg"],
        summary["root_systolic_mmhg"],
        summary["root_diastolic_mmhg"],
        summary["root_mean_mmhg"],
        summary["mean_pressure_change_mmhg"],
        summary["stroke_volume_ml"],
        summary["peak_velocity_m_s"],
    )

    # Testing: the prepared lumen and the pressure and flow curves.
    tt = ProcessTests(
        class_name=class_name,
        results_dir=output_dir,
        log_level=log_level,
    )
    screenshots = [
        tt.save_screenshot_mesh(
            result["geometry"]["surface"],
            "lumen_surface.png",
            camera_position="xz",
            color="lightcoral",
        ),
        output_dir / "pressure_flow.png",
    ]

    tutorial_results = {
        "summary": summary,
        "files": {**saved, "plot": plot_file},
        "screenshots": screenshots,
    }
