"""Physiological inputs for aortic blood-flow simulation.

:class:`AortaFlowPhysiology` holds the per-patient inputs of a simulation -
the patient's systolic and diastolic blood pressure (for example a cuff
measurement) and heart rate - together with the blood properties, the outlet
three-element Windkessel totals and the aortic wall properties of the
BloodFlow experiments (``experiments/BloodFlow-*-Warp``). It builds the
smoothed periodic inlet flow and the per-outlet :class:`Windkessel` objects,
and :meth:`AortaFlowPhysiology.calibrated` fits the outlet totals so that
the patient's inflow reproduces the patient's input pressures.
:meth:`~AortaFlowPhysiology.lv_volumes_ml` and
:meth:`~AortaFlowPhysiology.lv_ejection_inflow` derive the inlet flow from the
left-ventricle (LV) volume of a cardiac labelmap series, assuming an ideal
non-regurgitant aortic valve and a closed mitral valve during ejection;
:meth:`~AortaFlowPhysiology.reference_lv_volumes_ml` is a reference series of
the Tutorial 13 heart.

The flow workflows calibrate the outlets to the input pressures by default;
only the fluid-structure (FSI) workflow uses the wall fields.

A long series of labelmaps need not fit in memory: pass
:meth:`~AortaFlowPhysiology.lv_volumes_ml` a generator, which reads one frame
at a time::

    lv_volume_ml = AortaFlowPhysiology.lv_volumes_ml(
        (itk.imread(f) for f in frame_files), [151]
    )
    inflow, interval_s = AortaFlowPhysiology().lv_inflow(lv_volume_ml)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar

import itk
import numpy as np
from scipy.optimize import brentq, fsolve

from ..visualize_flow_fields import VisualizeFlowFields
from ..warp_navier_stokes import Windkessel, inlet_flow


@dataclass
class AortaFlowPhysiology:
    """Per-patient inputs and blood, outlet and wall parameters of aortic flow.

    **Per-patient inputs.** ``systolic_pressure_mmhg``,
    ``diastolic_pressure_mmhg`` and ``heart_rate_bpm`` describe the patient
    being simulated - for example a cuff measurement taken with the CT - and
    are meant to be set by the user. Their defaults are example patient
    inputs (112/78 mmHg, the root pressures of
    ``experiments/BloodFlow-1D-Rigid-Warp``), not simulation outputs. The
    simulation does not predict these pressures: :meth:`calibrated` sets the
    outlet resistance and compliance so that the patient's inflow reproduces
    them as the systolic and diastolic pressure of a whole-body model.

    **Outlet totals.** ``total_resistance_pa_s_m3``,
    ``total_compliance_m3_pa`` and ``initial_pressure_pa`` may instead be set
    directly; :meth:`calibrated` overwrites them. Their defaults are the
    values the rigid BloodFlow experiments tuned for their inflow.
    """

    #: Pa per mmHg, the same value as
    #: :attr:`~monai_physio.VisualizeFlowFields.MMHG_PA`.
    MMHG_PA: ClassVar[float] = VisualizeFlowFields.MMHG_PA

    # Per-patient inputs; these defaults are an example patient.
    #: Patient heart rate; the inflow series is retimed to it.  Adult rest
    #: 60-100 bpm.
    heart_rate_bpm: float = 60.0
    #: Patient systolic arterial pressure, a user input the outlets are
    #: calibrated to.  Typical adult ~120 mmHg.
    systolic_pressure_mmhg: float = 112.0
    #: Patient diastolic arterial pressure, a user input the outlets are
    #: calibrated to.  Typical adult ~80 mmHg.
    diastolic_pressure_mmhg: float = 78.0
    #: Standard deviation of the periodic Gaussian that smooths the interval
    #: inflow staircase.
    inflow_smoothing_s: float = 0.03
    #: Total outlet resistance of all Windkessels in parallel.  Adult ~1.65e8
    #: (93 mmHg mean at 4.5 L/min); a lower cardiac output needs more for the
    #: same pressures.
    total_resistance_pa_s_m3: float = 3.33e8
    #: Summed outlet compliance.  Adult total arterial compliance ~1-2e-8
    #: m^3/Pa (1-2 mL/mmHg).
    total_compliance_m3_pa: float = 6.8e-9
    #: Fraction of each outlet resistance that is proximal.  Adult
    #: characteristic impedance 5-10% of R.
    proximal_fraction: float = 0.06
    #: Fraction of the inflow leaving each named outlet; renormalized over the
    #: outlets present in a model.  Assumed splits; adult arch branches take
    #: roughly 30% of cardiac output in total.
    flow_fractions: dict[str, float] = field(
        default_factory=lambda: {"distal": 0.70, "bct": 0.17, "lsa": 0.08, "lcca": 0.05}
    )
    #: Initial Windkessel capacitor pressure.  Adult diastolic ~80 mmHg.
    initial_pressure_pa: float = 80.0 * MMHG_PA
    #: Blood density.  Adult blood 1050-1060 kg/m^3.
    density_kg_m3: float = 1060.0
    #: Blood dynamic viscosity (Newtonian).  Adult blood 3-4 mPa s at high
    #: shear.
    viscosity_pa_s: float = 0.004
    #: Wall Young's modulus (FSI only).  Adult aorta ~0.3-1 MPa, stiffer with
    #: age.
    young_modulus_pa: float = 1.0e6
    #: Wall Poisson ratio (FSI only).  The arterial wall is nearly
    #: incompressible (~0.5).
    poisson_ratio: float = 0.45
    #: Wall thickness (FSI only).  Adult thoracic aorta ~2 mm.
    wall_thickness_mm: float = 1.5
    #: Stiffness of the elastic support of the outer wall surface (FSI only).
    #: Perivascular support, not measured.
    tether_stiffness_pa_m: float = 5.0e5

    @property
    def mean_pressure_mmhg(self) -> float:
        """Mean arterial pressure of the input pressures.

        The clinical estimate ``diastolic + (systolic - diastolic) / 3``,
        reported for reference; :meth:`calibrated` does not use it.
        """
        pulse = self.systolic_pressure_mmhg - self.diastolic_pressure_mmhg
        return self.diastolic_pressure_mmhg + pulse / 3.0

    def calibrated(
        self,
        interval_fluxes_m3_s: np.ndarray,
        interval_s: float,
        dt_s: float = 0.001,
        wall_compliance_m3_pa: float = 0.0,
    ) -> AortaFlowPhysiology:
        """Return a copy whose outlets reproduce the input pressures.

        The outlets are calibrated as the whole-body three-element Windkessel
        they split from (:meth:`Windkessel.from_total` with fraction one and
        ``proximal_fraction``), fed by the smoothed periodic inflow
        (:meth:`inlet_flow`) and stepped like the workflows' outlets.  An
        elastic aorta adds its own compliance ``C_w`` in front of them: the
        root pressure ``P`` then obeys ``C_w dP/dt = Q_in - Q_out`` with
        ``P`` the Windkessel pressure of the outflow ``Q_out`` (stepped
        implicitly in ``Q_out``, as the flow solver does), as in
        ``experiments/BloodFlow-3D-FSI-Simpleware-Warp``.

        - The total resistance and the total outlet compliance are fitted
          together, on their logarithms, so that the periodic root pressure
          has the input systolic pressure as its maximum and the input
          diastolic pressure as its minimum (the venous pressure is taken as
          zero).  The start is the resistance that gives the clinical mean
          :attr:`mean_pressure_mmhg` (in a periodic state the mean pressure
          is the total resistance times the mean flow) with the compliance,
          found by Brent root finding between 1e-12 and 1e-6 m^3/Pa, that
          gives the input pulse pressure; a 2D root solve then moves both.
        - The initial pressure is the periodic capacitor pressure at
          :meth:`beat_start_s`, where the flow workflows start, so their
          first beat starts near the periodic state, though only
          approximately when a wall or a 3D pressure drop is present.

        The proximal resistance carries the pulse that the flow drives
        through it, so with the large total resistance a low cardiac output
        needs, the root pressure follows the shape of the inflow.

        The model covers one period of the inflow series,
        ``len(interval_fluxes_m3_s) * interval_s``, which may hold several
        beats; the systolic and diastolic pressures are the extremes over
        all of them.  The root pressure of a flow model adds the pressure
        drop along its aorta.

        Args:
            interval_fluxes_m3_s: Mean inflow of each frame interval in
                m^3/s, for example from :meth:`lv_inflow`.
            interval_s: Duration of each frame interval in seconds.
            dt_s: Approximate backward-Euler time step of the model.
            wall_compliance_m3_pa: Compliance ``dV/dP`` of the aortic wall
                in front of the outlets; 0 for a rigid aorta.

        Returns:
            A copy with ``total_resistance_pa_s_m3``,
            ``total_compliance_m3_pa`` and ``initial_pressure_pa`` replaced.

        Raises:
            ValueError: If the input pressures do not satisfy
                ``0 < diastolic < systolic``, if the inflow has no ejection,
                or if no outlet resistance and compliance give the input
                pressures, as when the wall alone is too compliant.
        """
        systolic = self.systolic_pressure_mmhg
        diastolic = self.diastolic_pressure_mmhg
        if not 0 < diastolic < systolic:
            raise ValueError(
                f"Input pressures {systolic}/{diastolic} mmHg do not satisfy "
                "0 < diastolic < systolic"
            )
        fluxes = np.asarray(interval_fluxes_m3_s, dtype=np.float64)
        period_s = len(fluxes) * interval_s
        steps = max(1, round(period_s / dt_s))
        step_s = period_s / steps
        times = self.beat_start_s(fluxes, interval_s) + np.arange(1, steps + 1) * step_s
        flow = self.inlet_flow(times, fluxes, interval_s)[0]
        pulse_pa = (systolic - diastolic) * self.MMHG_PA
        unreachable = ValueError(
            f"No outlet resistance and compliance give {systolic:.1f}/"
            f"{diastolic:.1f} mmHg with this inflow, proximal_fraction "
            f"{self.proximal_fraction} and wall compliance "
            f"{wall_compliance_m3_pa:.3g} m^3/Pa"
        )

        def periodic(log_r: float, log_c: float) -> tuple[np.ndarray, float]:
            trial = replace(
                self,
                total_resistance_pa_s_m3=float(np.exp(log_r)),
                total_compliance_m3_pa=float(np.exp(log_c)),
            )
            return trial._periodic_pressure(flow, step_s, wall_compliance_m3_pa)

        def error_mmhg(log_rc: np.ndarray) -> np.ndarray:
            pressure = periodic(float(log_rc[0]), float(log_rc[1]))[0] / self.MMHG_PA
            return np.array([pressure.max() - systolic, pressure.min() - diastolic])

        log_r = float(np.log(self.mean_pressure_mmhg * self.MMHG_PA / fluxes.mean()))

        def pulse_error(log_c: float) -> float:
            return float(np.ptp(periodic(log_r, log_c)[0])) - pulse_pa

        low, high = float(np.log(1e-12)), float(np.log(1e-6))
        if pulse_error(low) < 0 or pulse_error(high) > 0:
            raise unreachable
        start = np.array([log_r, brentq(pulse_error, low, high, xtol=1e-8)])
        log_rc = np.asarray(fsolve(error_mmhg, start, xtol=1e-12), dtype=np.float64)
        # fsolve can stop away from a root; accept only a fit within 0.01 mmHg.
        if not np.all(np.abs(error_mmhg(log_rc)) < 1e-2):
            raise unreachable
        return replace(
            self,
            total_resistance_pa_s_m3=float(np.exp(log_rc[0])),
            total_compliance_m3_pa=float(np.exp(log_rc[1])),
            initial_pressure_pa=periodic(float(log_rc[0]), float(log_rc[1]))[1],
        )

    def _periodic_pressure(
        self, flow_m3_s: np.ndarray, dt_s: float, wall_compliance_m3_pa: float = 0.0
    ) -> tuple[np.ndarray, float]:
        """Return the periodic root pressure of the whole-body model.

        *flow_m3_s* holds one period of inflow at steps of *dt_s*; see
        :meth:`calibrated` for the model.  Its state, the root and capacitor
        pressures, at the end of the period is affine in the state at its
        start, so passes from three starts give the start that repeats, and a
        fourth pass from it gives the periodic pressure.

        Returns:
            Root pressure at every step in Pa and the periodic capacitor
            pressure at the start of the period.
        """

        def run(start_pa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            windkessel = Windkessel.from_total(
                1.0,
                self.total_resistance_pa_s_m3,
                self.total_compliance_m3_pa,
                self.proximal_fraction,
                float(start_pa[1]),
            )
            root = float(start_pa[0])
            pressure = []
            for inflow in flow_m3_s:
                outflow = float(inflow)
                if wall_compliance_m3_pa > 0:
                    resistance, offset = windkessel.implicit(dt_s)
                    rate = dt_s / wall_compliance_m3_pa
                    outflow = (root + rate * outflow - offset) / (resistance + rate)
                root = windkessel.advance(outflow, dt_s)
                pressure.append(root)
            return np.asarray(pressure), np.array([root, windkessel.pressure])

        end_from_zero = run(np.zeros(2))[1]
        response = np.column_stack([run(unit)[1] - end_from_zero for unit in np.eye(2)])
        start_pa = np.linalg.solve(np.eye(2) - response, end_from_zero)
        return run(start_pa)[0], float(start_pa[1])

    @staticmethod
    def lv_volumes_ml(
        labelmaps: Iterable[itk.Image], lv_label_ids: Sequence[int]
    ) -> np.ndarray:
        """Return the LV volume of each labelmap in mL.

        The labelmaps are visited once, in order, and none is kept, so a
        generator such as ``(itk.imread(f) for f in frame_files)`` reads one
        frame at a time.

        Args:
            labelmaps: One labelmap per time frame.
            lv_label_ids: Labels whose voxels form the LV cavity.

        Returns:
            Voxel count times voxel volume of each frame.
        """
        volumes = []
        for labelmap in labelmaps:
            voxel_ml = float(np.prod(np.asarray(labelmap.GetSpacing()))) / 1000.0
            voxels = np.isin(itk.array_view_from_image(labelmap), lv_label_ids)
            volumes.append(np.count_nonzero(voxels) * voxel_ml)
        return np.asarray(volumes)

    @staticmethod
    def reference_lv_volumes_ml() -> np.ndarray:
        """Return reference LV volumes for test-mode runs: three beats, 100 frames.

        This is reference data, not a measurement of the patient.  It follows
        the LV volume of the full-resolution Tutorial 13 labelmaps of the
        Chest-CT scan: end-diastolic 72 mL, end-systolic 40.5 mL (a 31.5 mL
        stroke), ejection over 36% of the beat, a plateau, rapid filling to
        64 mL, diastasis and the atrial kick back to 72 mL.  Volumes move
        between these knots along half-cosines, so the inflow rises and falls
        smoothly.  :meth:`lv_inflow` retimes it like a measured series, to
        0.03 s per frame at 60 bpm.  The tutorials use it in place of the
        measured series where the heart barely moves, as in the small
        test-mode run of Tutorial 13.

        Returns:
            LV volume of each of the 100 frames in mL.
        """
        phase_knots = np.array([0.0, 0.09, 0.45, 0.57, 0.81, 0.90, 1.0])
        volume_knots = np.array([72.0, 72.0, 40.5, 40.5, 64.0, 64.0, 72.0])
        phase = np.arange(100) * 3 / 100 % 1.0
        segment = np.searchsorted(phase_knots, phase, side="right") - 1
        start, end = phase_knots[segment], phase_knots[segment + 1]
        blend = (1.0 - np.cos(np.pi * (phase - start) / (end - start))) / 2.0
        low, high = volume_knots[segment], volume_knots[segment + 1]
        return np.asarray(low + (high - low) * blend)

    def lv_inflow(self, lv_volume_ml: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the interval inflow of a periodic LV volume series.

        The series is retimed so that its ejections (``_ejection_count``)
        beat at ``heart_rate_bpm``; the inflow is then
        :meth:`lv_ejection_inflow` at that frame interval.

        Args:
            lv_volume_ml: LV volume of each frame in mL, for example from
                :meth:`lv_volumes_ml`.

        Returns:
            Mean inflow of each frame interval in m^3/s and the interval in
            seconds.

        Raises:
            ValueError: If the series has no ejection.
        """
        beats = self._ejection_count(self.lv_ejection_inflow(lv_volume_ml, 1.0))
        if beats == 0:
            raise ValueError("The LV volume series has no ejection")
        interval_s = self._retimed_interval_s(len(lv_volume_ml), beats)
        return self.lv_ejection_inflow(lv_volume_ml, interval_s), interval_s

    @staticmethod
    def lv_ejection_inflow(
        lv_volume_ml: np.ndarray,
        interval_s: float,
        minimum_ejection_fraction: float = 0.1,
    ) -> np.ndarray:
        """Return the mean aortic inflow of each frame interval in m^3/s.

        The series is periodic: interval ``i`` runs from frame ``i`` to frame
        ``i + 1`` and the last one wraps to frame 0. With an ideal valve the
        inflow is the positive LV volume loss over the interval and zero while
        the LV fills. Segmentation noise also produces small losses, so an
        ejection, a run of consecutive losing intervals, is kept only if its
        volume reaches ``minimum_ejection_fraction`` of the LV volume range
        (maximum minus minimum, about the stroke volume).

        Args:
            lv_volume_ml: LV volume of each frame in mL.
            interval_s: Duration of each frame interval in seconds.
            minimum_ejection_fraction: Smallest kept ejection volume relative to
                the LV volume range.

        Returns:
            Inflow of each interval; ``len(lv_volume_ml)`` values.
        """
        volume = np.asarray(lv_volume_ml, dtype=np.float64)
        loss = np.maximum(volume - np.roll(volume, -1), 0.0)
        minimum = minimum_ejection_fraction * float(np.ptp(volume))
        ejecting = loss > 0
        # Start at a non-ejecting interval so that no run wraps around.
        order = np.roll(np.arange(len(loss)), -int(np.argmin(ejecting)))
        run: list[int] = []
        for i in [*order, order[0]]:
            if ejecting[i]:
                run.append(i)
                continue
            if run and loss[run].sum() < minimum:
                loss[run] = 0.0
            run = []
        return loss * 1e-6 / interval_s

    @staticmethod
    def _ejection_count(interval_fluxes: np.ndarray) -> int:
        """Return the number of ejections, periodic runs of positive flux."""
        ejecting = np.asarray(interval_fluxes) > 0
        return int(np.count_nonzero(ejecting & ~np.roll(ejecting, 1)))

    @staticmethod
    def beat_start_s(interval_fluxes: np.ndarray, interval_s: float) -> float:
        """Return the middle of the filling gap before the first ejection.

        The flow workflows start their simulated time here, so every beat window
        holds one whole ejection.

        Args:
            interval_fluxes: Mean flux of each interval of a periodic series.
            interval_s: Duration of each interval in seconds.

        Returns:
            The time in seconds, negative when the gap wraps from the end of
            the series.

        Raises:
            ValueError: If the series has no ejection.
        """
        ejecting = np.asarray(interval_fluxes) > 0
        onsets = np.flatnonzero(ejecting & ~np.roll(ejecting, 1))
        if len(onsets) == 0:
            raise ValueError("The inflow has no ejection")
        onset = int(onsets[0])
        # Intervals since the previous ejection, which may wrap from the end.
        gap = int(np.argmax(np.roll(ejecting, -onset)[::-1]))
        return (onset - 0.5 * gap) * interval_s

    def _retimed_interval_s(self, frame_count: int, beat_count: int) -> float:
        """Return the frame interval that makes a series beat at the heart rate.

        Args:
            frame_count: Number of frames in the periodic series.
            beat_count: Number of heart beats the series covers, for example
                ``_ejection_count`` of its inflow.
        """
        return beat_count * 60.0 / self.heart_rate_bpm / frame_count

    def inlet_flow(
        self,
        times_s: np.ndarray,
        interval_fluxes_m3_s: np.ndarray,
        interval_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the smoothed periodic inlet flow Q(t) and dQ/dt.

        Wraps :func:`monai_physio.warp_navier_stokes.inlet_flow` with
        ``inflow_smoothing_s``; the series repeats with period
        ``len(interval_fluxes_m3_s) * interval_s``.

        Args:
            times_s: Evaluation times in seconds.
            interval_fluxes_m3_s: Mean flux of each frame interval in m^3/s.
            interval_s: Duration of each frame interval in seconds.

        Returns:
            Flow in m^3/s and its derivative in m^3/s^2 at ``times_s``.
        """
        return inlet_flow(
            times_s, interval_fluxes_m3_s, interval_s, self.inflow_smoothing_s
        )

    def windkessels(self, outlet_names: Sequence[str]) -> dict[str, Windkessel]:
        """Return one Windkessel per outlet from the totals.

        The flow fractions of ``outlet_names`` are renormalized to sum to one,
        so the outlet resistances in parallel equal the total resistance and
        the outlet compliances sum to the total compliance.

        Raises:
            ValueError: If an outlet has no flow fraction.
        """
        unknown = [name for name in outlet_names if name not in self.flow_fractions]
        if unknown:
            raise ValueError(
                f"No flow fraction for outlets {unknown}; "
                f"known outlets are {sorted(self.flow_fractions)}"
            )
        total = sum(self.flow_fractions[name] for name in outlet_names)
        return {
            name: Windkessel.from_total(
                self.flow_fractions[name] / total,
                self.total_resistance_pa_s_m3,
                self.total_compliance_m3_pa,
                self.proximal_fraction,
                self.initial_pressure_pa,
            )
            for name in outlet_names
        }
