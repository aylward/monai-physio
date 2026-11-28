"""Turn an aorta labelmap into the inputs of the aortic blood-flow solvers.

:class:`AortaProcessGeometry` prepares one aorta for three models:

- a 1D rigid model, which needs a centerline with section areas and the arc
  position of every branch outlet;
- a 3D rigid model, which needs a lumen tetrahedral mesh whose boundary
  triangles are tagged as wall or as a port (:meth:`rigid_mesh`);
- a 3D fluid-structure (FSI) model, which also needs a wall tetrahedral mesh
  that shares every wall triangle of the lumen exactly (:meth:`fsi_mesh`).

:meth:`prepare` builds a *flow labelmap* on an isotropic grid: the aortic
lumen (label 1), a wall shell of fixed thickness around it (label 2) and one
plug label per port (10, 11, ...).  The root and distal plugs are slabs cut
normal to the centerline a margin inside each end of the vessel; each branch
plug is the wall window where a branch label meets the aorta.  Every port is
then the patch of lumen boundary that touches its plug.  Meshing uses netgen
only (:meth:`ProcessVolumetricMesh.extract_label_tetrahedra_netgen`).

Geometry is in LPS millimetres; the solvers take metres, so convert at the
solver boundary.

Typical usage::

    geometry_tools = AortaProcessGeometry()
    geometry = geometry_tools.prepare(
        labelmap, aorta_label_ids=(6,),
        branch_label_ids={"BCT": 109, "LCCA": 112, "LSA": 123},
    )
    rigid = geometry_tools.rigid_mesh(geometry)
    fsi = geometry_tools.fsi_mesh(geometry)
"""

from __future__ import annotations

import importlib.util
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from ..monai_physio_base import MONAIPhysioBase
from ..process_images import ProcessImages
from ..process_mesh_kernels import ProcessMeshKernels
from ..process_surface_mesh import ProcessSurfaceMesh
from ..process_volumetric_mesh import ProcessVolumetricMesh


class AortaProcessGeometry(MONAIPhysioBase):
    """Prepare an aorta labelmap for 1D, 3D rigid and 3D FSI flow models.

    Ports are named ``"root"`` (the inlet), ``"distal"`` and the lowercased
    keys of ``branch_label_ids``, so they match the keys of
    :attr:`AortaFlowPhysiology.flow_fractions`.  Every port's region id is
    its plug label in the flow labelmap, and region 0 is the wall, in the
    surface and in both meshes.
    """

    LUMEN_LABEL = 1
    WALL_LABEL = 2
    FIRST_PORT_LABEL = 10

    #: Thickness of the root and distal plugs, beyond which the vessel is
    #: removed.
    END_PLUG_THICKNESS_MM = 4.0

    #: Taubin smoothing of the contoured surfaces.  More pinches the thin
    #: wall shell between the lumen and the background.
    SMOOTHING_ITERATIONS = 10

    #: Arc-length step of the raw centerline before smoothing, and the
    #: Gaussian smoothing of that centerline.
    CENTERLINE_STEP_MM = 0.5
    CENTERLINE_SIGMA_MM = 3.0

    #: Ports narrower than ``port_cells_across`` elements are refined within
    #: this many port radii of their center by :meth:`rigid_mesh`.
    PORT_ZONE_RADII = 1.5

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize AortaProcessGeometry.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self._image_tools = ProcessImages(log_level=log_level)
        self._kernels = ProcessMeshKernels(log_level=log_level)
        self._surface_tools = ProcessSurfaceMesh(log_level=log_level)
        self._volume_tools = ProcessVolumetricMesh(log_level=log_level)

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------
    def prepare(
        self,
        labelmap_image: itk.Image,
        aorta_label_ids: Sequence[int] = (1,),
        branch_label_ids: Optional[Mapping[str, int]] = None,
        *,
        root_point_mm: Optional[Sequence[float]] = None,
        distal_point_mm: Optional[Sequence[float]] = None,
        spacing_mm: float = 0.7,
        wall_thickness_mm: float = 2.0,
        end_margin_mm: float = 10.0,
        branch_gap_mm: float = 3.0,
        centerline_spacing_mm: float = 1.5,
    ) -> dict[str, Any]:
        """Build the flow labelmap, tagged lumen surface, ports and centerline.

        1. The aorta labels (and branch labels) are cropped and resampled to
           an isotropic *spacing_mm* grid with a label vote; the lumen is the
           largest connected component of the aorta labels.
        2. The distal end is the lumen voxel at *distal_point_mm*, else the
           most central voxel of the most inferior lumen slice; the root end
           is the voxel at *root_point_mm*, else the lumen voxel geodesically
           farthest from the distal end.  The centerline is the
           radius-weighted interior geodesic between them, smoothed.
        3. The wall is every voxel within *wall_thickness_mm* outside the
           lumen.  Each branch opens an outlet window: the voxels outside the
           lumen within ``max(branch_gap_mm, wall_thickness_mm)`` of it and
           within *branch_gap_mm* of that branch's label.
        4. The vessel is cut normal to the centerline *end_margin_mm* inside
           each end; a :attr:`END_PLUG_THICKNESS_MM` slab beyond each cut
           becomes that end's plug and the rest beyond it is removed.
        5. The labels' surfaces are contoured together and the lumen
           surface's triangles are tagged: a triangle shared with the wall
           surface is wall (0), any other belongs to the port whose plug it
           touches.

        Args:
            labelmap_image: Labelmap holding the aorta and, optionally, its
                branches.
            aorta_label_ids: Labels whose union is the aortic lumen.
            branch_label_ids: Branch name -> label.  Each branch present
                becomes an outlet port named by its lowercased key.  ``None``
                (or empty) gives a tube with only the root and distal ports.
            root_point_mm: Optional LPS point at the root end of the vessel;
                the centerline starts at the lumen voxel nearest it.
            distal_point_mm: Optional LPS point at the distal end of the
                vessel; the centerline ends at the lumen voxel nearest it.
            spacing_mm: Isotropic grid spacing of the flow labelmap and the
                contoured surfaces.  The FSI mesh size grows about as its
                inverse cube.
            wall_thickness_mm: Thickness of the wall shell.
            end_margin_mm: Distance along the centerline from each end of the
                vessel to its cut.  0 cuts through the end points.
            branch_gap_mm: How far a branch label may lie from the lumen and
                still open a window, and how far the window spreads around
                it.
            centerline_spacing_mm: Arc-length spacing of the returned
                centerline points.

        Returns:
            ``"labelmap"``: the flow labelmap (uint16; 1 lumen, 2 wall,
            10+ port plugs) on the isotropic grid, with the input direction.
            ``"ports"``: port name -> ``label_id``, ``role`` (``"inlet"`` or
            ``"outlet"``), ``center_mm`` (area-weighted centroid of the port
            triangles), ``normal`` (unit, out of the lumen), ``area_mm2``
            (magnitude of the summed triangle vector areas, the area that
            flux crosses) and ``arc_length_mm`` (arc length of the
            centerline point nearest the center).
            ``"surface"``: the closed, outward-oriented lumen surface with
            cell data ``"Region"`` (0 wall, else the port's ``label_id``).
            ``"centerline"``: polyline from the root cut to the distal cut
            with point data ``arc_length_mm`` (0 at the root),
            ``area_mm2`` (lumen section normal to the centerline),
            ``radius_mm`` (the equivalent radius ``sqrt(area / pi)``) and
            ``tangent``.
            ``"wall_thickness_mm"``: *wall_thickness_mm*, the thickness of
            the wall shell that :meth:`fsi_mesh` meshes.

        Raises:
            ValueError: If the labelmap holds none of *aorta_label_ids*, if
                the centerline is too short for the two end margins, or if
                the root or distal port receives no surface triangle.
        """
        started = time.perf_counter()
        branches = {
            name.lower(): int(label) for name, label in (branch_label_ids or {}).items()
        }
        aorta_ids = [int(label) for label in aorta_label_ids]

        grid, labels = self._isotropic_labels(
            labelmap_image,
            aorta_ids,
            list(branches.values()),
            spacing_mm,
            wall_thickness_mm + branch_gap_mm + self.END_PLUG_THICKNESS_MM,
        )
        lumen = np.isin(labels, aorta_ids)
        lumen = self._largest_component(lumen)
        if not lumen.any():
            raise ValueError(f"The labelmap holds none of labels {aorta_ids}")
        self.log_info(
            "Lumen: %.1f mL on a %s grid of %.3g mm",
            lumen.sum() * spacing_mm**3 / 1000,
            "x".join(str(n) for n in labels.shape[::-1]),
            spacing_mm,
        )

        inside_mm = self._distance_mm(lumen, spacing_mm)
        path_mm = self._medial_path(
            grid, lumen, inside_mm, spacing_mm, root_point_mm, distal_point_mm
        )
        curve, arc, tangent = self._smooth_curve(path_mm)
        length = float(arc[-1])
        if length <= 2 * end_margin_mm + 2 * spacing_mm:
            raise ValueError(
                f"The {length:.1f} mm centerline is too short for "
                f"{end_margin_mm} mm end margins"
            )
        self.log_info("Centerline: %.1f mm between the vessel ends", length)

        flow, port_labels = self._flow_labels(
            grid,
            labels,
            lumen,
            inside_mm,
            branches,
            (curve, arc, tangent),
            spacing_mm,
            wall_thickness_mm,
            end_margin_mm,
            branch_gap_mm,
        )
        flow_image = itk.GetImageFromArray(flow)
        flow_image.CopyInformation(grid)

        surfaces = self._surface_tools.extract_label_surfaces(
            flow_image,
            isotropic_spacing_mm=spacing_mm,
            smoothing_iterations=self.SMOOTHING_ITERATIONS,
        )
        lumen_points = np.asarray(surfaces[self.LUMEN_LABEL].points, dtype=np.float64)
        lumen_faces = np.asarray(
            surfaces[self.LUMEN_LABEL].regular_faces, dtype=np.int64
        )
        wall = surfaces[self.WALL_LABEL]
        regions = self._tag_lumen_faces(
            lumen_points,
            lumen_faces,
            (np.asarray(wall.points), np.asarray(wall.regular_faces)),
            {
                label: np.asarray(surfaces[label].points)
                for label in port_labels.values()
                if label in surfaces
            },
        )
        surface = pv.PolyData(
            lumen_points,
            faces=np.column_stack([np.full(len(lumen_faces), 3), lumen_faces]).ravel(),
        )
        surface.cell_data["Region"] = regions

        start = int(np.searchsorted(arc, end_margin_mm))
        stop = int(np.searchsorted(arc, length - end_margin_mm))
        centerline = self._centerline(
            grid,
            lumen,
            curve[start : stop + 1],
            centerline_spacing_mm,
            2.5 * float(inside_mm.max()) + spacing_mm,
        )
        ports = self._ports(surface, port_labels, centerline)
        for name, port in ports.items():
            self.log_info(
                "Port %-8s label %d: %.1f mm2 at arc %.1f mm, center %s mm",
                name,
                port["label_id"],
                port["area_mm2"],
                port["arc_length_mm"],
                np.round(port["center_mm"], 2).tolist(),
            )
        self.log_info("Prepared the geometry in %.1f s", time.perf_counter() - started)
        return {
            "labelmap": flow_image,
            "ports": ports,
            "surface": surface,
            "centerline": centerline,
            "wall_thickness_mm": float(wall_thickness_mm),
        }

    def _isotropic_labels(
        self,
        labelmap_image: itk.Image,
        aorta_ids: list[int],
        branch_ids: list[int],
        spacing_mm: float,
        margin_mm: float,
    ) -> tuple[itk.Image, np.ndarray]:
        """Crop to the aorta and branch labels, pad and vote-resample."""
        values = itk.GetArrayViewFromImage(labelmap_image)
        kept = np.where(np.isin(values, aorta_ids + branch_ids), values, 0)
        selected = itk.GetImageFromArray(kept.astype(np.uint16))
        selected.CopyInformation(labelmap_image)
        pad = int(np.ceil(margin_mm / min(labelmap_image.GetSpacing()))) + 2
        cropped = self._image_tools.crop_to_nonzero(selected, pad)
        grid = self._kernels.vote_resample_labelmap(cropped, spacing_mm)
        return grid, np.asarray(itk.GetArrayFromImage(grid))

    @staticmethod
    def _distance_mm(mask: np.ndarray, spacing_mm: float) -> np.ndarray:
        """Return the distance in mm from each voxel of *mask* to its outside."""
        return np.asarray(
            ndimage.distance_transform_edt(mask, sampling=[spacing_mm] * 3)
        )

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        """Return the largest 6-connected component of *mask*."""
        components, count = cast(tuple[np.ndarray, int], ndimage.label(mask))
        if count <= 1:
            return np.asarray(mask, dtype=bool)
        sizes = np.bincount(components.ravel())[1:]
        return np.asarray(components == 1 + int(np.argmax(sizes)))

    def _medial_path(
        self,
        grid: itk.Image,
        lumen: np.ndarray,
        inside_mm: np.ndarray,
        spacing_mm: float,
        root_point_mm: Optional[Sequence[float]],
        distal_point_mm: Optional[Sequence[float]],
    ) -> np.ndarray:
        """Return the radius-weighted interior geodesic from root to distal end."""
        lumen_kji = np.argwhere(lumen)
        lumen_mm = self._image_tools.index_to_world(grid, lumen_kji)
        tree = cKDTree(lumen_mm)
        sampling = [spacing_mm] * 3

        if distal_point_mm is None:
            # The most inferior lumen slice (LPS +z is superior), at its center.
            low = lumen_mm[:, 2] <= lumen_mm[:, 2].min() + 0.5 * spacing_mm
            candidates = lumen_kji[low]
            distal = candidates[np.argmax(inside_mm[tuple(candidates.T)])]
        else:
            distal = lumen_kji[tree.query(np.asarray(distal_point_mm))[1]]
        distal_seed = tuple(int(v) for v in distal)

        if root_point_mm is None:
            search = MCP_Geometric(
                np.where(lumen, 1.0, np.inf), fully_connected=True, sampling=sampling
            )
            geodesic, _ = search.find_costs([distal_seed])
            geodesic = np.where(lumen & np.isfinite(geodesic), geodesic, -1.0)
            root_seed = tuple(
                int(v) for v in np.unravel_index(np.argmax(geodesic), lumen.shape)
            )
        else:
            root = lumen_kji[tree.query(np.asarray(root_point_mm))[1]]
            root_seed = tuple(int(v) for v in root)

        # A cost of 1 / r^2 keeps the path near the middle of the lumen.
        cost = np.where(
            lumen, 1.0 / np.maximum(inside_mm, 0.5 * spacing_mm) ** 2, np.inf
        )
        search = MCP_Geometric(cost, fully_connected=True, sampling=sampling)
        search.find_costs([root_seed], [distal_seed])
        path_kji = np.asarray(search.traceback(distal_seed))
        return self._image_tools.index_to_world(grid, path_kji)

    def _smooth_curve(
        self, path_mm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resample a voxel path by arc length and smooth it.

        Returns:
            The points, their arc lengths and unit tangents.
        """
        curve, _ = self._resample(path_mm, self.CENTERLINE_STEP_MM)
        curve = ndimage.gaussian_filter1d(
            curve,
            self.CENTERLINE_SIGMA_MM / self.CENTERLINE_STEP_MM,
            axis=0,
            mode="nearest",
        )
        return self._arc_and_tangent(curve)

    @staticmethod
    def _resample(curve: np.ndarray, step_mm: float) -> tuple[np.ndarray, np.ndarray]:
        """Resample a polyline uniformly by arc length."""
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        keep = np.r_[True, np.diff(arc) > 1e-9]
        curve, arc = curve[keep], arc[keep]
        samples = np.linspace(0.0, arc[-1], int(np.ceil(arc[-1] / step_mm)) + 1)
        resampled = np.column_stack(
            [np.interp(samples, arc, curve[:, axis]) for axis in range(3)]
        )
        return resampled, samples

    @staticmethod
    def _arc_and_tangent(
        curve: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return *curve*, its arc lengths from the first point and unit tangents."""
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        tangent = np.gradient(curve, arc, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        return curve, arc, tangent

    def _flow_labels(
        self,
        grid: itk.Image,
        labels: np.ndarray,
        lumen: np.ndarray,
        inside_mm: np.ndarray,
        branches: dict[str, int],
        centerline: tuple[np.ndarray, np.ndarray, np.ndarray],
        spacing_mm: float,
        wall_thickness_mm: float,
        end_margin_mm: float,
        branch_gap_mm: float,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Return the flow label array and port name -> plug label.

        *centerline* holds the smoothed curve's points, arc lengths and
        tangents; *inside_mm* is the distance from each lumen voxel to the
        outside.
        """
        curve, arc, tangent = centerline
        outside_mm = self._distance_mm(~lumen, spacing_mm)
        flow = np.zeros(labels.shape, dtype=np.uint16)
        flow[(outside_mm > 0) & (outside_mm <= wall_thickness_mm)] = self.WALL_LABEL
        flow[lumen] = self.LUMEN_LABEL

        port_labels = {
            "root": self.FIRST_PORT_LABEL,
            "distal": self.FIRST_PORT_LABEL + 1,
        }
        present = {n: b for n, b in branches.items() if np.any(labels == b)}
        for name in branches.keys() - present.keys():
            self.log_warning("Branch %s has no voxels; it has no port", name)
        if present:
            branch_mm = np.stack(
                [self._distance_mm(labels != b, spacing_mm) for b in present.values()]
            )
            # Where two windows overlap, the voxel goes to the nearer branch.
            nearest = np.argmin(branch_mm, axis=0)
            near_lumen = ~lumen & (outside_mm <= max(branch_gap_mm, wall_thickness_mm))
            for index, name in enumerate(present):
                window = near_lumen & (branch_mm[index] <= branch_gap_mm)
                window &= nearest == index
                if not window.any():
                    self.log_warning(
                        "Branch %s is over %.3g mm from the lumen; it has no port",
                        name,
                        branch_gap_mm,
                    )
                    continue
                port_labels[name] = self.FIRST_PORT_LABEL + len(port_labels)
                flow[window] = port_labels[name]

        # Cut each end normal to the centerline.  Only voxels whose nearest
        # centerline point lies beyond the cut, or less than about two radii
        # inside it, can be cut, so the plane cannot clip another part of a
        # curved vessel.
        vessel_kji = np.argwhere(flow > 0)
        vessel_mm = self._image_tools.index_to_world(grid, vessel_kji)
        vessel_arc = arc[cKDTree(curve).query(vessel_mm)[1]]
        length = float(arc[-1])
        for name, cut_arc, sign in (
            ("root", end_margin_mm, -1.0),
            ("distal", length - end_margin_mm, 1.0),
        ):
            index = int(np.clip(np.searchsorted(arc, cut_arc), 0, len(arc) - 1))
            center, normal = curve[index], tangent[index]
            radius = float(
                ndimage.map_coordinates(
                    inside_mm,
                    self._image_tools.world_to_index(grid, center[None]).T,
                    order=1,
                )[0]
            )
            reach = 2 * radius + wall_thickness_mm
            # sign * offset > 0 is beyond the cut, away from the domain.
            beyond = sign * ((vessel_mm - center) @ normal)
            candidates = (sign * (vessel_arc - cut_arc) >= -reach) & (beyond > 0)
            plug = candidates & (beyond <= self.END_PLUG_THICKNESS_MM)
            removed = candidates & (beyond > self.END_PLUG_THICKNESS_MM)
            flow[tuple(vessel_kji[plug].T)] = port_labels[name]
            flow[tuple(vessel_kji[removed].T)] = 0

        # Cuts and windows can isolate fragments of the lumen or the wall.
        # A lumen fragment is dropped.  A wall fragment joins the nearest
        # plug rather than leaving the lumen open to the background or a
        # loose piece of wall in the FSI model.
        lumen_now = flow == self.LUMEN_LABEL
        flow[lumen_now & ~self._largest_component(lumen_now)] = 0
        wall = flow == self.WALL_LABEL
        fragments = wall & ~self._largest_component(wall)
        if fragments.any():
            _, nearest = cast(
                tuple[np.ndarray, np.ndarray],
                ndimage.distance_transform_edt(
                    flow < self.FIRST_PORT_LABEL, return_indices=True
                ),
            )
            flow[fragments] = flow[tuple(nearest[:, fragments])]
            self.log_info("%d wall fragment voxels join a plug", int(fragments.sum()))
        exposed = 0
        lumen_now = flow == self.LUMEN_LABEL
        for axis in range(3):
            for shift in (1, -1):
                exposed += int(
                    (lumen_now & (np.roll(flow, shift, axis=axis) == 0)).sum()
                )
        if exposed:
            self.log_warning("%d lumen voxel faces touch the background", exposed)
        counts = {label: int((flow == label).sum()) for label in port_labels.values()}
        self.log_info("Port plug voxels: %s", counts)
        return flow, port_labels

    def _centerline(
        self,
        grid: itk.Image,
        lumen: np.ndarray,
        curve: np.ndarray,
        step_mm: float,
        half_width_mm: float,
    ) -> pv.PolyData:
        """Return the polyline with arc length, section area and radius.

        Sections are sampled over a square of *half_width_mm* around each
        point.
        """
        points, _ = self._resample(curve, step_mm)
        points, arc, tangent = self._arc_and_tangent(points)
        areas = np.array(
            [
                self._section_area(grid, lumen, c, t, half_width_mm)
                for c, t in zip(points, tangent)
            ]
        )
        missing = ~np.isfinite(areas)
        if missing.all():
            raise ValueError("The centerline lies outside the lumen")
        if missing.any():
            self.log_warning(
                "%d of %d sections miss the lumen; their areas are interpolated",
                int(missing.sum()),
                len(areas),
            )
            areas[missing] = np.interp(arc[missing], arc[~missing], areas[~missing])
        polyline = pv.PolyData(points, lines=np.r_[len(points), np.arange(len(points))])
        polyline.point_data["arc_length_mm"] = arc
        polyline.point_data["area_mm2"] = areas
        polyline.point_data["radius_mm"] = np.sqrt(areas / np.pi)
        polyline.point_data["tangent"] = tangent
        return polyline

    def _section_area(
        self,
        grid: itk.Image,
        lumen: np.ndarray,
        center: np.ndarray,
        normal: np.ndarray,
        half_width_mm: float,
    ) -> float:
        """Area of the lumen section through *center* normal to *normal*.

        The plane is sampled every quarter voxel over a square of
        *half_width_mm* and the connected part holding *center* is counted.
        The lumen given is the uncut one, so sections next to the end cuts
        are whole.
        """
        u = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        step = float(grid.GetSpacing()[0]) / 4
        offsets = np.arange(-half_width_mm, half_width_mm + step / 2, step)
        plane = (
            center + offsets[:, None, None] * u + offsets[None, :, None] * v
        ).reshape(-1, 3)
        inside = ndimage.map_coordinates(
            lumen.astype(np.uint8),
            self._image_tools.world_to_index(grid, plane).T,
            order=0,
        ).reshape(len(offsets), len(offsets))
        parts, _ = cast(tuple[np.ndarray, int], ndimage.label(inside))
        middle = len(offsets) // 2
        part = parts[middle, middle]
        if part == 0:
            return float("nan")
        return float((parts == part).sum() * step * step)

    def _ports(
        self,
        surface: pv.PolyData,
        port_labels: dict[str, int],
        centerline: pv.PolyData,
    ) -> dict[str, dict[str, Any]]:
        """Return each port's label, role, center, normal, area and arc length."""
        corners = np.asarray(surface.points)[surface.regular_faces]
        vector_areas = 0.5 * np.cross(
            corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        )
        areas = np.linalg.norm(vector_areas, axis=1)
        centroids = corners.mean(axis=1)
        regions = np.asarray(surface.cell_data["Region"])
        tree = cKDTree(np.asarray(centerline.points))
        arc = np.asarray(centerline.point_data["arc_length_mm"])
        ports: dict[str, dict[str, Any]] = {}
        for name, label in port_labels.items():
            selected = regions == label
            if not selected.any():
                if name in ("root", "distal"):
                    raise ValueError(
                        f"The {name} port received no surface triangle; check "
                        "the end points and end_margin_mm"
                    )
                self.log_warning("Port %s has no surface triangles; dropped", name)
                continue
            center = (centroids[selected] * areas[selected, None]).sum(axis=0)
            center /= areas[selected].sum()
            total = vector_areas[selected].sum(axis=0)
            ports[name] = {
                "label_id": label,
                "role": "inlet" if name == "root" else "outlet",
                "center_mm": center.tolist(),
                "normal": (total / np.linalg.norm(total)).tolist(),
                "area_mm2": float(np.linalg.norm(total)),
                "arc_length_mm": float(arc[tree.query(center)[1]]),
            }
        return ports

    # ------------------------------------------------------------------
    # Boundary triangle regions
    # ------------------------------------------------------------------
    def _tag_lumen_faces(
        self,
        lumen_points: np.ndarray,
        lumen_faces: np.ndarray,
        wall: tuple[np.ndarray, np.ndarray],
        plug_points: Mapping[int, np.ndarray],
    ) -> np.ndarray:
        """Tag lumen boundary triangles as wall (0) or by plug label.

        A lumen triangle is wall only when the wall holds exactly the same
        triangle, so every wall triangle of the lumen is a wall triangle too.
        Any other triangle goes to the plug holding most of its vertices; one
        touching no plug (where the contours of three labels meet) goes to
        the plug nearest its centroid.

        Args:
            lumen_points: Lumen points.
            lumen_faces: Lumen boundary triangles.
            wall: (points, boundary triangles) of the wall.
            plug_points: Plug label -> that plug's points.

        Returns:
            The region of every lumen triangle.

        Raises:
            ValueError: If triangles are off the wall but there is no plug.
        """
        regions = np.zeros(len(lumen_faces), dtype=np.int64)
        off_wall = ~self._volume_tools.shared_faces(lumen_points, lumen_faces, [wall])
        if not off_wall.any():
            return regions
        if not plug_points:
            raise ValueError("Lumen triangles are off the wall but there is no port")
        labels = np.asarray(list(plug_points))
        faces = lumen_faces[off_wall]
        votes = np.zeros((len(faces), len(labels)), dtype=np.int64)
        for column, points in enumerate(plug_points.values()):
            lumen_ids, plug_ids = self._volume_tools.common_point_ids(
                lumen_points, points
            )
            votes[:, column] = np.isin(lumen_ids, plug_ids)[faces].sum(axis=1)
        choice = labels[np.argmax(votes, axis=1)]
        no_vote = votes.max(axis=1) == 0
        if no_vote.any():
            centroids = lumen_points[faces[no_vote]].mean(axis=1)
            distance = np.stack(
                [cKDTree(p).query(centroids)[0] for p in plug_points.values()]
            )
            choice[no_vote] = labels[np.argmin(distance, axis=0)]
        regions[off_wall] = choice
        return regions

    # ------------------------------------------------------------------
    # Meshes
    # ------------------------------------------------------------------
    def rigid_mesh(
        self,
        geometry: Mapping[str, Any],
        element_size_mm: float = 2.0,
        port_cells_across: int = 8,
    ) -> dict[str, Any]:
        """Mesh the lumen alone with netgen for a rigid-wall 3D flow model.

        The lumen label of the flow labelmap is contoured on its own and
        netgen re-tiles it to triangles about *element_size_mm* across, so
        the mesh is much coarser than the conforming :meth:`fsi_mesh`.
        Ports narrower than *port_cells_across* elements get elements of
        their diameter / *port_cells_across* within
        :attr:`PORT_ZONE_RADII` port radii of their center; otherwise a
        narrow opening is tiled by a few triangles whose no-slip rim blocks
        it.  Each boundary triangle takes the region of the nearest triangle
        of ``geometry["surface"]``.

        Args:
            geometry: The result of :meth:`prepare`.
            element_size_mm: Target edge length away from small ports.
            port_cells_across: Elements across the narrowest port diameter.

        Returns:
            ``"points_mm"`` (n, 3), ``"tetrahedra"`` (m, 4) positively
            oriented, ``"boundary_faces"`` (f, 3) wound outward,
            ``"boundary_regions"`` (f,) 0 for wall else the port's
            ``label_id``, and ``"port_element_size_mm"`` port name -> the
            element size used at that port.

        Raises:
            ImportError: If netgen, part of the ``[aorta]`` extra, is missing.
            ValueError: If a port receives no boundary triangle.
        """
        self._require_netgen()
        started = time.perf_counter()
        flow = geometry["labelmap"]
        ports = geometry["ports"]
        spacing_mm = float(flow.GetSpacing()[0])
        size_image, port_sizes = self._port_size_image(
            flow, ports, element_size_mm, port_cells_across
        )
        mesh = self._volume_tools.extract_label_tetrahedra_netgen(
            flow,
            element_size_mm=element_size_mm,
            label_ids=[self.LUMEN_LABEL],
            smoothing_iterations=self.SMOOTHING_ITERATIONS,
            shared_interfaces=False,
            contour_spacing_mm=spacing_mm,
            element_size_image=size_image,
        )[self.LUMEN_LABEL]
        points, tetrahedra = self._volume_tools.tetrahedra_arrays(mesh)
        faces = self._volume_tools.boundary_faces(tetrahedra)

        surface = geometry["surface"]
        centroids = points[faces].mean(axis=1)
        nearest = surface.find_closest_cell(centroids)
        regions = np.asarray(surface.cell_data["Region"])[nearest].astype(np.int64)
        for name, port in ports.items():
            count = int((regions == port["label_id"]).sum())
            if count == 0:
                raise ValueError(f"Port {name} received no boundary triangle")
            self.log_info("Rigid mesh port %-8s: %d triangles", name, count)
        self.log_info(
            "Rigid lumen mesh: %d points, %d tetrahedra, %d boundary triangles "
            "in %.1f s",
            len(points),
            len(tetrahedra),
            len(faces),
            time.perf_counter() - started,
        )
        return {
            "points_mm": points,
            "tetrahedra": tetrahedra,
            "boundary_faces": faces,
            "boundary_regions": regions,
            "port_element_size_mm": {
                name: port_sizes.get(name, element_size_mm) for name in ports
            },
        }

    @staticmethod
    def _require_netgen() -> None:
        """Raise a clear ImportError when netgen is not installed."""
        if importlib.util.find_spec("netgen") is None:
            raise ImportError(
                "AortaProcessGeometry meshes with netgen; install the aorta "
                'extra: pip install "monai-physio[aorta]"'
            )

    def _port_size_image(
        self,
        flow: itk.Image,
        ports: Mapping[str, Mapping[str, Any]],
        element_size_mm: float,
        port_cells_across: int,
    ) -> tuple[Optional[itk.Image], dict[str, float]]:
        """Return the element-size image refining the narrow ports, if any."""
        spacing_mm = float(flow.GetSpacing()[0])
        zones: dict[str, tuple[np.ndarray, float, float]] = {}
        for name, port in ports.items():
            radius = float(np.sqrt(port["area_mm2"] / np.pi))
            size = 2 * radius / port_cells_across
            if size < 0.9 * element_size_mm:
                zones[name] = (
                    np.asarray(port["center_mm"]),
                    self.PORT_ZONE_RADII * radius,
                    size,
                )
        if not zones:
            return None, {}
        shape = itk.GetArrayViewFromImage(flow).shape
        zone_labels = np.zeros(shape, dtype=np.uint8)
        # Coarse zones first, so a finer zone overwrites any overlap.
        ordered = sorted(zones.items(), key=lambda item: -item[1][2])
        sizes: dict[int, float] = {}
        for label, (_, (center, zone_radius, size)) in enumerate(ordered, 1):
            center_kji = self._image_tools.world_to_index(flow, center[None])[0]
            reach = zone_radius / spacing_mm
            low = np.maximum(np.floor(center_kji - reach), 0).astype(int)
            high = np.minimum(np.ceil(center_kji + reach) + 1, shape).astype(int)
            box = np.ogrid[tuple(slice(a, b) for a, b in zip(low, high))]
            # The grid is isotropic and its direction orthonormal, so index
            # distance times the spacing is the distance in mm.
            distance2 = sum((axis - c) ** 2 for axis, c in zip(box, center_kji))
            region = zone_labels[tuple(slice(a, b) for a, b in zip(low, high))]
            region[distance2 * spacing_mm**2 <= zone_radius**2] = label
            sizes[label] = size
        zone_image = itk.GetImageFromArray(zone_labels)
        zone_image.CopyInformation(flow)
        size_image = ProcessVolumetricMesh.create_element_size_image(zone_image, sizes)
        port_sizes = {name: zone[2] for name, zone in zones.items()}
        self.log_info("Refining narrow ports to %s mm", port_sizes)
        return size_image, port_sizes

    def fsi_mesh(
        self,
        geometry: Mapping[str, Any],
        element_size_mm: float = 2.0,
        collar_mm: float = 2.0,
    ) -> dict[str, Any]:
        """Mesh the lumen and wall together with netgen for an FSI model.

        All labels of the flow labelmap are contoured together and netgen
        fills each surface keeping its triangles, so the lumen and wall
        meshes have identical points on their interface.  The contours of
        three labels do not quite conform where they meet, so a lumen
        triangle is wall only when it is exactly a wall triangle; the rest
        are port triangles.  Every wall triangle of the lumen, and so every
        P2 edge node on it, is then a wall node, as
        :class:`WarpFluidStructure` requires.  The wall is clamped within
        *collar_mm* of every plug and every non-conforming triangle, all of
        which lie at the ports, and its outer surface carries the tether.

        The boundary triangles keep the contour's size, so
        ``geometry["labelmap"]``'s spacing, not *element_size_mm*, sets the
        mesh size; prepare with a coarser ``spacing_mm`` for a smaller mesh.

        Args:
            geometry: The result of :meth:`prepare`.
            element_size_mm: Largest interior edge length.
            collar_mm: Wall points this close to a plug or a non-conforming
                lumen triangle are clamped.

        Returns:
            The keys of :meth:`rigid_mesh` but ``"port_element_size_mm"`` for
            the lumen: ``"points_mm"``, ``"tetrahedra"`` (positive),
            ``"boundary_faces"`` (outward) and ``"boundary_regions"`` (0 for
            wall, else the port's ``label_id``); then
            ``"wall_points_mm"``, ``"wall_tetrahedra"`` (positive),
            ``"wall_clamped"`` (bool per wall point) and
            ``"wall_tether_faces"`` (outward wall triangles facing the
            background) of the solid; and ``"nonconforming_faces"``, the
            number of lumen triangles that match no wall or plug triangle.

        Raises:
            ImportError: If netgen, part of the ``[aorta]`` extra, is missing.
        """
        self._require_netgen()
        started = time.perf_counter()
        flow = geometry["labelmap"]
        spacing_mm = float(flow.GetSpacing()[0])
        meshes = self._volume_tools.extract_label_tetrahedra_netgen(
            flow,
            element_size_mm=element_size_mm,
            smoothing_iterations=self.SMOOTHING_ITERATIONS,
            shared_interfaces=True,
            contour_spacing_mm=spacing_mm,
        )
        arrays = {
            label: self._volume_tools.tetrahedra_arrays(mesh)
            for label, mesh in meshes.items()
        }
        lumen_points, lumen_tetrahedra = arrays[self.LUMEN_LABEL]
        wall_points, wall_tetrahedra = arrays[self.WALL_LABEL]
        faces = self._volume_tools.boundary_faces(lumen_tetrahedra)
        wall = (wall_points, self._volume_tools.boundary_faces(wall_tetrahedra))
        plugs = {
            port["label_id"]: (
                arrays[port["label_id"]][0],
                self._volume_tools.boundary_faces(arrays[port["label_id"]][1]),
            )
            for port in geometry["ports"].values()
            if port["label_id"] in arrays
        }
        regions = self._tag_lumen_faces(
            lumen_points, faces, wall, {k: p for k, (p, _) in plugs.items()}
        )
        conforming = self._volume_tools.shared_faces(
            lumen_points, faces, [wall, *plugs.values()]
        )
        # The tether is every wall triangle that neither the lumen nor a plug
        # shares, which faces the background.
        tether = wall[1][
            ~self._volume_tools.shared_faces(
                *wall, [(lumen_points, faces), *plugs.values()]
            )
        ]
        anchors = [lumen_points[faces[~conforming]].reshape(-1, 3)]
        anchors += [points for points, _ in plugs.values()]
        distance, _ = cKDTree(np.concatenate(anchors)).query(
            wall_points, distance_upper_bound=collar_mm
        )
        clamped = np.asarray(distance) <= collar_mm
        nonconforming = int((~conforming).sum())

        self.log_info(
            "FSI meshes: lumen %d points, %d tetrahedra; wall %d points, "
            "%d tetrahedra, %d clamped points, %d tether triangles; "
            "%d of %d lumen triangles do not conform; %.1f s",
            len(lumen_points),
            len(lumen_tetrahedra),
            len(wall_points),
            len(wall_tetrahedra),
            int(clamped.sum()),
            len(tether),
            nonconforming,
            len(faces),
            time.perf_counter() - started,
        )
        return {
            "points_mm": lumen_points,
            "tetrahedra": lumen_tetrahedra,
            "boundary_faces": faces,
            "boundary_regions": regions,
            "wall_points_mm": wall_points,
            "wall_tetrahedra": wall_tetrahedra,
            "wall_clamped": clamped,
            "wall_tether_faces": tether,
            "nonconforming_faces": nonconforming,
        }
