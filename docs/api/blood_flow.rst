====================================
Blood Flow (Warp)
====================================

.. currentmodule:: monai_physio

GPU blood-flow building blocks written with NVIDIA Warp FEM. They are
demonstrated by the ``experiments/BloodFlow-*`` folders: 1D and 3D flow in a
rigid aorta and two-way fluid-structure interaction with a compliant wall.
The solvers are demonstrations, not validated physiology.

* :class:`WarpNavierStokes` - P2/P1 incompressible Navier-Stokes on
  tetrahedra, with a cuDSS-preconditioned GMRES solve in float32 or float64.
* :class:`Windkessel` - three-element (RCR) outlet model.
* :func:`~monai_physio.warp_navier_stokes.inlet_flow` - smooth periodic inlet
  flow from per-interval fluxes, such as left-ventricle volume loss.
* :class:`WarpHyperelasticWall` - prestressed Neo-Hookean vessel wall.
* :class:`WarpFluidStructure` - Dirichlet-Neumann coupling with IQN-ILS
  acceleration.
* :class:`VisualizeFlowFields` - interactive viewer and movie export of the
  saved flow fields.

The aorta classes of the optional ``monai_physio.aorta`` subpackage apply
these models to an aorta labelmap. They need the ``[aorta]`` extra
(``pip install "monai-physio[aorta]"``, which includes ``[netgen]``) and are
imported from the subpackage, not from ``monai_physio``::

    from monai_physio.aorta import AortaFlowPhysiology, AortaWorkflowSimulateFlow1D

* :class:`~monai_physio.aorta.AortaProcessGeometry` - centerline, section
  areas, ports and netgen meshes of an aorta labelmap.
* :class:`~monai_physio.aorta.AortaFlowPhysiology` - the per-patient inputs
  (systolic and diastolic blood pressure, heart rate), blood, Windkessel and
  wall parameters, the inflow derived from a left-ventricle volume series, and
  the calibration of the outlet Windkessels to the input pressures. The blood
  pressures are user-specified patient values that the outlets are calibrated
  to, not simulation outputs. Its static methods
  :meth:`~monai_physio.aorta.AortaFlowPhysiology.lv_volumes_ml` and
  :meth:`~monai_physio.aorta.AortaFlowPhysiology.lv_ejection_inflow` turn a
  labelmap series into LV volumes and inflow,
  :meth:`~monai_physio.aorta.AortaFlowPhysiology.beat_start_s` gives the
  time at which the flow workflows start, and
  :meth:`~monai_physio.aorta.AortaFlowPhysiology.reference_lv_volumes_ml`
  returns the reference LV volume series the tutorials use in test mode.
  ``AortaFlowPhysiology.MMHG_PA`` is the Pa per mmHg conversion, the same
  value as ``VisualizeFlowFields.MMHG_PA``.
* :class:`~monai_physio.aorta.AortaWorkflowSimulateFlowBase` - what the three
  workflows share: the geometry from a labelmap or a prepared geometry, the
  outlet calibration, the last-beat root-pressure summary and the pressure
  and flow plot.
* :class:`~monai_physio.aorta.AortaWorkflowSimulateFlow1D` - rigid-wall 1D
  pressure and flow along the centerline (Tutorial 19).
* :class:`~monai_physio.aorta.AortaWorkflowSimulateFlow3D` - rigid-wall 3D
  flow with :class:`WarpNavierStokes` in a netgen lumen mesh, saved for
  :class:`VisualizeFlowFields` (Tutorial 20).
* :class:`~monai_physio.aorta.AortaWorkflowSimulateFlowFSI` - 3D flow coupled
  to a moving Neo-Hookean wall with :class:`WarpFluidStructure` in conforming
  netgen lumen and wall meshes, with the outlets calibrated to the input
  pressures behind the wall's own compliance (Tutorial 21).

Each workflow's ``process`` returns a dict with ``"geometry"``,
``"physiology"``, ``"series"`` (one value per time sample, including
``"times_s"``, ``"inlet_flow_m3_s"``, ``"root_pressure_pa"`` and the outlet
name -> array dicts ``"outlet_flow_m3_s"`` and ``"outlet_pressure_pa"``) and
``"summary"``. The 1D result adds ``"sections"``, the arrays along the
centerline; the 3D and FSI results add ``"mesh"`` (the
:meth:`~monai_physio.aorta.AortaProcessGeometry.rigid_mesh` or
:meth:`~monai_physio.aorta.AortaProcessGeometry.fsi_mesh` keys), ``"frames"`` (the last beat's
saved fields) and ``"run"`` (solver settings and timings). The 3D and FSI
workflows also accept a ``mesh`` with those keys in place of meshing the
geometry.

The pressure preconditioner loads the NVIDIA cuDSS library at run time; set
``CUDSS_LIBRARY`` to its path when it is not found automatically.

Module Reference
================

.. autoclass:: WarpNavierStokes
   :members:

.. autoclass:: Windkessel
   :members:
   :exclude-members: proximal_resistance, distal_resistance, compliance, pressure

.. autofunction:: monai_physio.warp_navier_stokes.inlet_flow

.. autofunction:: monai_physio.warp_navier_stokes.find_cudss_library

.. autoclass:: WarpHyperelasticWall
   :members:

.. autoclass:: WarpFluidStructure
   :members:

.. autofunction:: monai_physio.warp_fluid_structure.quasi_newton_update

.. autoclass:: VisualizeFlowFields
   :members:

Aorta Subpackage
----------------

Requires the ``[aorta]`` extra (``monai-physio[aorta]``).

.. automodule:: monai_physio.aorta
   :no-members:

.. currentmodule:: monai_physio.aorta

.. autoclass:: AortaProcessGeometry
   :members:

.. autoclass:: AortaFlowPhysiology
   :members:

.. autoclass:: AortaWorkflowSimulateFlowBase
   :members:

.. autoclass:: AortaWorkflowSimulateFlow1D
   :members:

.. autoclass:: AortaWorkflowSimulateFlow3D
   :members:

.. autoclass:: AortaWorkflowSimulateFlowFSI
   :members:
