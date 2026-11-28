"""Aortic blood-flow simulation from an aorta labelmap.

This optional subpackage needs the ``[aorta]`` extra
(``pip install "monai-physio[aorta]"``), which installs scikit-image for the
centerline and netgen (the ``[netgen]`` extra) for the lumen and wall meshes.
``import monai_physio`` does not import it; import its classes from
``monai_physio.aorta``.

Every class here starts with ``Aorta`` and lives in the module named after the
rest of the class name, so :class:`AortaProcessGeometry` is in
``monai_physio.aorta.process_geometry`` and
:class:`AortaWorkflowSimulateFlow1D` in
``monai_physio.aorta.workflow_simulate_flow_1d``.

- :class:`AortaProcessGeometry` - centerline, sections, ports and meshes.
- :class:`AortaFlowPhysiology` - per-patient inputs, blood, outlet and wall
  parameters, and the inflow from a left-ventricle volume series.
- :class:`AortaWorkflowSimulateFlowBase` - what the flow workflows share.
- :class:`AortaWorkflowSimulateFlow1D` - rigid-wall 1D flow (Tutorial 19).
- :class:`AortaWorkflowSimulateFlow3D` - rigid-wall 3D flow (Tutorial 20).
- :class:`AortaWorkflowSimulateFlowFSI` - 3D flow with a moving wall
  (Tutorial 21).
"""

from importlib.util import find_spec

if find_spec("skimage") is None:
    raise ImportError(
        "monai_physio.aorta needs scikit-image; install the aorta extra: "
        'pip install "monai-physio[aorta]"'
    )

from .flow_physiology import AortaFlowPhysiology
from .process_geometry import AortaProcessGeometry
from .workflow_simulate_flow_1d import AortaWorkflowSimulateFlow1D
from .workflow_simulate_flow_3d import AortaWorkflowSimulateFlow3D
from .workflow_simulate_flow_base import AortaWorkflowSimulateFlowBase
from .workflow_simulate_flow_fsi import AortaWorkflowSimulateFlowFSI

__all__ = [
    "AortaFlowPhysiology",
    "AortaProcessGeometry",
    "AortaWorkflowSimulateFlow1D",
    "AortaWorkflowSimulateFlow3D",
    "AortaWorkflowSimulateFlowBase",
    "AortaWorkflowSimulateFlowFSI",
]
