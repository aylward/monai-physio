"""Tests for AortaProcessGeometry on a synthetic aortic arch.

Set ``MONAI_PHYSIO_AORTA_LABELMAP`` to Tutorial 13's
``chest_ct_merged_labelmap.mha`` to also check the ports of the real aorta
(``test_prepare_tutorial_13_ports``); that test is skipped otherwise.

Requires the ``[aorta]`` extra (``monai-physio[aorta]``).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest
from scipy.spatial import cKDTree

pytest.importorskip("skimage")

from monai_physio.aorta import AortaProcessGeometry
from monai_physio.process_images import ProcessImages
from monai_physio.process_volumetric_mesh import ProcessVolumetricMesh

AORTA, BRANCH = 6, 9
RADIUS_MM = 5.0
# The arch's ends: the root leg ends higher than the distal leg, so the
# distal end is the most inferior lumen.
ROOT_END_MM = np.array([-16.0, 0.0, 10.0])
DISTAL_END_MM = np.array([16.0, 0.0, 3.0])
NETGEN = pytest.mark.skipif(
    importlib.util.find_spec("netgen") is None,
    reason="netgen extra (netgen-mesher) not installed",
)
# Vertex pairs of a tetrahedron's six edges and a triangle's three.
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_FACE_EDGES = ((0, 1), (1, 2), (0, 2))
# Optional real-data check: the Tutorial 13 merged chest labelmap.
TUTORIAL_13_LABELMAP = os.environ.get("MONAI_PHYSIO_AORTA_LABELMAP", "")


def arch_centerline() -> np.ndarray:
    """Return the arch centerline in LPS mm: two legs and a semicircle."""
    root_leg = np.linspace(ROOT_END_MM, [-16.0, 0.0, 32.0], 200)
    angle = np.linspace(np.pi, 0.0, 400)
    arch = np.column_stack(
        [16.0 * np.cos(angle), np.zeros_like(angle), 32.0 + 16.0 * np.sin(angle)]
    )
    distal_leg = np.linspace([16.0, 0.0, 32.0], DISTAL_END_MM, 300)
    return np.concatenate([root_leg, arch, distal_leg])


def arch_labelmap(with_branch: bool = True) -> itk.Image:
    """Return a 56 x 24 x 64 labelmap of 1 mm voxels holding an aortic arch.

    The direction flips the y axis.  The arch (label 6) is a 5 mm radius
    tube with flat ends; the branch (label 9) is a 3 mm radius vertical tube leaving the
    top of the arch.
    """
    shape_kji = (64, 24, 56)
    image = itk.GetImageFromArray(np.zeros(shape_kji, dtype=np.uint16))
    image.SetSpacing([1.0, 1.0, 1.0])
    image.SetOrigin([-28.0, 12.0, 0.0])
    direction = np.diag([1.0, -1.0, 1.0])
    image.SetDirection(itk.matrix_from_array(direction))
    kji = np.argwhere(np.ones(shape_kji, dtype=bool))
    points = ProcessImages.index_to_world(image, kji)
    labels = np.zeros(len(points), dtype=np.uint16)
    distance, _ = cKDTree(arch_centerline()).query(points)
    if with_branch:
        lateral = np.linalg.norm(points[:, :2], axis=1)
        branch = (lateral <= 3.0) & (points[:, 2] > 48.0) & (points[:, 2] < 61.0)
        labels[branch] = BRANCH
    # Flat ends: nothing below either leg's end.
    below_end = np.where(points[:, 0] < 0, ROOT_END_MM[2], DISTAL_END_MM[2])
    labels[(distance <= RADIUS_MM) & (points[:, 2] >= below_end)] = AORTA
    array = labels.reshape(shape_kji)
    labelmap = itk.GetImageFromArray(array)
    labelmap.CopyInformation(image)
    return labelmap


@pytest.fixture(scope="module")
def arch_geometry() -> dict[str, Any]:
    """The prepared arch with one branch, on a 1 mm grid."""
    return AortaProcessGeometry().prepare(
        arch_labelmap(),
        aorta_label_ids=(AORTA,),
        branch_label_ids={"Top": BRANCH},
        spacing_mm=1.0,
        branch_gap_mm=1.5,
        centerline_spacing_mm=1.0,
    )


def closed_and_outward(
    points: np.ndarray, tetrahedra: np.ndarray, faces: np.ndarray
) -> float:
    """Return the WarpNavierStokes closure ratio, 1 for a closed outward skin."""
    cells = points[tetrahedra]
    volume = np.einsum(
        "ij,ij->",
        cells[:, 1] - cells[:, 0],
        np.cross(cells[:, 2] - cells[:, 0], cells[:, 3] - cells[:, 0]),
    )
    corners = points[faces]
    area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]) / 2
    return float(np.einsum("ij,ij->", corners.mean(axis=1), area) * 2 / volume)


def six_volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Return six times each tetrahedron's signed volume."""
    cells = points[tetrahedra]
    return np.asarray(np.linalg.det(cells[:, 1:] - cells[:, :1]))


@pytest.mark.requires_gpu
def test_prepare_finds_ports_on_the_arch(arch_geometry: dict[str, Any]) -> None:
    """The root, distal and branch ports lie where the arch puts them."""
    ports = arch_geometry["ports"]
    assert list(ports) == ["root", "distal", "top"]
    assert [p["label_id"] for p in ports.values()] == [10, 11, 12]
    assert [p["role"] for p in ports.values()] == ["inlet", "outlet", "outlet"]
    # Each end is cut 10 mm along the centerline inside the vessel end.
    expected = {
        "root": ([-16.0, 0.0, 20.0], [0.0, 0.0, -1.0]),
        "distal": ([16.0, 0.0, 13.0], [0.0, 0.0, -1.0]),
        "top": ([0.0, 0.0, 53.0], [0.0, 0.0, 1.0]),
    }
    for name, (center, normal) in expected.items():
        assert np.linalg.norm(np.subtract(ports[name]["center_mm"], center)) < 2.5
        assert np.dot(ports[name]["normal"], normal) > 0.9
    end_area = np.pi * RADIUS_MM**2
    for name in ("root", "distal"):
        assert ports[name]["area_mm2"] == pytest.approx(end_area, rel=0.2)
    # Root cut (z = 20) to z = 32 on the leg, then a quarter of the arch.
    assert ports["root"]["arc_length_mm"] < 1.0
    assert ports["top"]["arc_length_mm"] == pytest.approx(12 + 8 * np.pi, abs=3.0)
    length = arch_geometry["centerline"].point_data["arc_length_mm"][-1]
    assert ports["distal"]["arc_length_mm"] == pytest.approx(length, abs=1.0)


@pytest.mark.requires_gpu
def test_prepare_flow_labelmap(arch_geometry: dict[str, Any]) -> None:
    """The flow labelmap holds lumen, wall and three plugs on the input frame."""
    flow = arch_geometry["labelmap"]
    assert set(np.unique(itk.GetArrayViewFromImage(flow)).tolist()) == {
        0,
        1,
        2,
        10,
        11,
        12,
    }
    np.testing.assert_allclose(
        itk.array_from_matrix(flow.GetDirection()), np.diag([1.0, -1.0, 1.0])
    )
    np.testing.assert_allclose(flow.GetSpacing(), [1.0, 1.0, 1.0])

    surface = arch_geometry["surface"]
    assert surface.n_open_edges == 0
    assert set(np.unique(surface.cell_data["Region"]).tolist()) == {0, 10, 11, 12}


@pytest.mark.requires_gpu
def test_prepare_centerline_inside_lumen(arch_geometry: dict[str, Any]) -> None:
    """The centerline stays in the lumen and its sections match the tube."""
    flow = arch_geometry["labelmap"]
    centerline = arch_geometry["centerline"]
    arc = centerline.point_data["arc_length_mm"]
    interior = (arc > 1.5) & (arc < arc[-1] - 1.5)
    kji = np.rint(
        ProcessImages.world_to_index(flow, np.asarray(centerline.points))
    ).astype(int)
    labels = itk.GetArrayViewFromImage(flow)[tuple(kji[interior].T)]
    assert np.all(labels == 1)
    distance, _ = cKDTree(arch_centerline()).query(np.asarray(centerline.points))
    assert np.max(distance) < 1.5
    area = centerline.point_data["area_mm2"]
    assert np.median(area) == pytest.approx(np.pi * RADIUS_MM**2, rel=0.15)
    np.testing.assert_allclose(
        centerline.point_data["radius_mm"], np.sqrt(area / np.pi)
    )
    assert np.diff(arc) == pytest.approx(1.0, abs=0.05)


@pytest.mark.requires_gpu
def test_prepare_without_branches_is_a_two_port_tube() -> None:
    """Without branch labels the arch has only its root and distal ports."""
    geometry = AortaProcessGeometry().prepare(
        arch_labelmap(with_branch=False),
        aorta_label_ids=(AORTA,),
        spacing_mm=1.0,
    )
    assert list(geometry["ports"]) == ["root", "distal"]
    labels = set(np.unique(itk.GetArrayViewFromImage(geometry["labelmap"])).tolist())
    assert labels == {0, 1, 2, 10, 11}
    assert set(np.unique(geometry["surface"].cell_data["Region"]).tolist()) == {
        0,
        10,
        11,
    }


@pytest.mark.requires_gpu
def test_prepare_follows_given_end_points() -> None:
    """Given end points swap which end is the root."""
    geometry = AortaProcessGeometry().prepare(
        arch_labelmap(with_branch=False),
        aorta_label_ids=(AORTA,),
        root_point_mm=DISTAL_END_MM.tolist(),
        distal_point_mm=ROOT_END_MM.tolist(),
        spacing_mm=1.0,
    )
    root = np.asarray(geometry["ports"]["root"]["center_mm"])
    assert np.linalg.norm(root - [16.0, 0.0, 13.0]) < 2.5


@NETGEN
@pytest.mark.requires_gpu
def test_rigid_mesh_tags_ports(arch_geometry: dict[str, Any]) -> None:
    """The coarse lumen mesh is closed and each port lands on its region."""
    mesh = AortaProcessGeometry().rigid_mesh(
        arch_geometry, element_size_mm=2.0, port_cells_across=8
    )
    points, tetrahedra = mesh["points_mm"], mesh["tetrahedra"]
    faces, regions = mesh["boundary_faces"], mesh["boundary_regions"]
    assert np.all(six_volumes(points, tetrahedra) > 0)
    assert closed_and_outward(points, tetrahedra, faces) == pytest.approx(1, abs=1e-8)
    assert set(np.unique(regions).tolist()) == {0, 10, 11, 12}
    for name, port in arch_geometry["ports"].items():
        corners = points[faces[regions == port["label_id"]]]
        area = np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        center = (corners.mean(axis=1) * area[:, None]).sum(axis=0) / area.sum()
        assert np.linalg.norm(center - port["center_mm"]) < 2.0, name
    # Every port is narrower than 8 elements of 2 mm, so each is refined.
    assert set(mesh["port_element_size_mm"]) == {"root", "distal", "top"}
    assert all(size < 2.0 for size in mesh["port_element_size_mm"].values())


@NETGEN
@pytest.mark.requires_gpu
def test_fsi_mesh_wall_faces_are_wall_faces(arch_geometry: dict[str, Any]) -> None:
    """Every fluid wall triangle, and its P2 edge nodes, is on the wall mesh."""
    mesh = AortaProcessGeometry().fsi_mesh(arch_geometry, element_size_mm=2.0)
    points, tetrahedra = mesh["points_mm"], mesh["tetrahedra"]
    faces, regions = mesh["boundary_faces"], mesh["boundary_regions"]
    wall_points, wall_tetrahedra = mesh["wall_points_mm"], mesh["wall_tetrahedra"]
    assert np.all(six_volumes(points, tetrahedra) > 0)
    assert np.all(six_volumes(wall_points, wall_tetrahedra) > 0)
    assert closed_and_outward(points, tetrahedra, faces) == pytest.approx(1, abs=1e-8)
    assert set(np.unique(regions).tolist()) == {0, 10, 11, 12}

    tools = ProcessVolumetricMesh
    wall_faces = tools.boundary_faces(wall_tetrahedra)
    fluid_wall = faces[regions == 0]
    assert tools.shared_faces(points, fluid_wall, [(wall_points, wall_faces)]).all()
    pairs = np.concatenate([wall_tetrahedra[:, [a, b]] for a, b in _EDGES])
    wall_midpoints = wall_points[np.unique(np.sort(pairs, axis=1), axis=0)].mean(1)
    face_pairs = np.concatenate([fluid_wall[:, [a, b]] for a, b in _FACE_EDGES])
    face_midpoints = points[np.sort(face_pairs, axis=1)].mean(axis=1)
    distance, _ = cKDTree(wall_midpoints).query(face_midpoints)
    assert np.max(distance) == 0.0

    clamped = mesh["wall_clamped"]
    assert 0 < clamped.sum() < len(clamped)
    tether = mesh["wall_tether_faces"]
    assert len(tether) > 0
    assert not tools.shared_faces(wall_points, tether, [(points, faces)]).any()
    # Every port rim is clamped: the wall points on each port's rim.
    for port in arch_geometry["ports"].values():
        rim = np.unique(faces[regions == port["label_id"]])
        distance, index = cKDTree(wall_points).query(points[rim])
        held = np.asarray(index)[np.asarray(distance) == 0.0]
        assert len(held) > 0 and clamped[held].all()


@pytest.mark.requires_gpu
@pytest.mark.skipif(
    not Path(TUTORIAL_13_LABELMAP).is_file(),
    reason="Set MONAI_PHYSIO_AORTA_LABELMAP to the Tutorial 13 merged labelmap",
)
def test_prepare_tutorial_13_ports() -> None:
    """The Tutorial 13 aorta's five ports lie where the reference run put them."""
    geometry = AortaProcessGeometry().prepare(
        itk.imread(TUTORIAL_13_LABELMAP),
        aorta_label_ids=(6,),
        branch_label_ids={"BCT": 109, "LCCA": 112, "LSA": 123},
    )
    expected = {
        "root": (-3.25, -4.88, -126.29),
        "distal": (15.33, -7.57, -301.27),
        "bct": (-5.94, 4.90, -53.86),
        "lcca": (1.79, 6.62, -52.87),
        "lsa": (8.75, 16.98, -48.00),
    }
    assert list(geometry["ports"]) == list(expected)
    for name, center in expected.items():
        offset = np.subtract(geometry["ports"][name]["center_mm"], center)
        assert np.linalg.norm(offset) < 5.0, name
