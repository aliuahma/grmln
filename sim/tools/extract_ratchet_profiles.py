"""Rerun with uv run --project sim --extra cad --with shapely==2.2.0 python
sim/tools/extract_ratchet_profiles.py SOURCE.stp.

Extract planar-face contact prisms from precise source BREP tessellation.
All output coordinates are source CAD world meters, before CAD_ROTATION.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon

from override_sim.robot import part_mass

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'sim/override_sim/assets/robot'
manifest = json.loads((ASSETS / 'manifest.json').read_text())


def load_brep(source):
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.TDF import TDF_Label, TDF_LabelSequence
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Trsf
    from extract_robot import geometric_properties

    reader = STEPCAFControl_Reader()
    reader.ReadFile(str(source))
    document = TDocStd_Document(TCollection_ExtendedString('ratchet'))
    if not reader.Transfer(document):
        raise RuntimeError('Cannot transfer source STEP assembly')
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    roots = TDF_LabelSequence()
    tool.GetFreeShapes(roots)
    meshes, world_shapes = {}, {}
    for node_id in ('1/468/56', '1/468/60', '1/468/110', '1/468/112'):
        indices = [int(part) for part in node_id.split('/')]
        label = roots.Value(indices[0])
        for index in indices[1:] + [None]:
            definition = TDF_Label()
            if tool.IsReference_s(label):
                tool.GetReferredShape_s(label, definition)
            else:
                definition = label
            if index is not None:
                children = TDF_LabelSequence()
                tool.GetComponents_s(definition, children, False)
                label = children.Value(index)
        shape = tool.GetShape_s(definition)
        transform = np.array(manifest['nodes'][node_id]['world_transform'])
        transform[:3, 3] *= 1000
        location = gp_Trsf()
        location.SetValues(*transform[:3].ravel().tolist())
        world_shapes[node_id] = BRepBuilderAPI_Transform(shape, location, True).Shape()
        BRepMesh_IncrementalMesh(shape, .005, False, .03, True).Perform()
        vertices, triangles = [], []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)
            if triangulation:
                start = len(vertices)
                for i in range(1, triangulation.NbNodes() + 1):
                    point = triangulation.Node(i).Transformed(location.Transformation())
                    vertices.append([point.X() * .001, point.Y() * .001, point.Z() * .001])
                for i in range(1, triangulation.NbTriangles() + 1):
                    triangles.append([start + j - 1 for j in triangulation.Triangle(i).Get()])
            explorer.Next()
        meshes[node_id] = (np.asarray(vertices), np.asarray(triangles))
    overlaps = {}
    for gear, pawl in [('1/468/56', '1/468/110'), ('1/468/60', '1/468/112')]:
        common = BRepAlgoAPI_Common(world_shapes[gear], world_shapes[pawl])
        common.Build()
        overlaps[f'{gear}:{pawl}'] = geometric_properties(common.Shape())['volume_m3']
    return meshes, overlaps


def convex_parts(profile):
    parts = list(shapely.constrained_delaunay_triangles(profile).geoms)
    allowed = profile.buffer(.00002)
    while True:
        index = shapely.STRtree(parts)
        merged = False
        for i, left in enumerate(parts):
            for j in index.query(left, predicate='intersects'):
                if j <= i:
                    continue
                right = parts[j]
                joined = left.union(right)
                if joined.geom_type != 'Polygon':
                    continue
                hull = joined.convex_hull
                if allowed.covers(hull):
                    parts[i] = hull
                    parts.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    return parts


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
args = parser.parse_args()
assert hashlib.sha256(args.source.read_bytes()).hexdigest() == manifest['source']['sha256']
meshes, exact_overlaps = load_brep(args.source)
profiles = []
for node_id in ('1/468/56', '1/468/60', '1/468/110', '1/468/112'):
    node = manifest['nodes'][node_id]
    definition = manifest['definitions'][node['definition']]
    points, faces = meshes[node_id]
    gear = node_id.endswith(('/56', '/60'))
    thickness_axis, plane_axes = (2, [0, 1]) if gear else (1, [0, 2])
    face_plane = 0.0 if gear else .00635
    thickness = [-.0047625, 0.0] if gear else [.003175, .00635]
    triangles = []
    for face in faces:
        vertices = points[face]
        if np.max(np.abs(vertices[:, thickness_axis] - face_plane)) < 1e-6:
            triangle = Polygon(np.round(vertices[:, plane_axes], 8))
            if triangle.area > 1e-14:
                triangles.append(triangle)
    profile = shapely.union_all(triangles).simplify(.000005, preserve_topology=True)
    if profile.geom_type == 'MultiPolygon':
        meaningful = [p for p in profile.geoms if p.area > 1e-10]
        profile = shapely.union_all(meaningful)
    profile = Polygon(profile.exterior)
    components = convex_parts(profile)
    transform = np.asarray(node['world_transform'])
    prisms = []
    for component in components:
        poly = np.asarray(component.exterior.coords[:-1])
        vertices = np.zeros((len(poly) * 2, 3))
        vertices[:len(poly), plane_axes] = poly
        vertices[len(poly):, plane_axes] = poly
        vertices[:len(poly), thickness_axis] = thickness[0]
        vertices[len(poly):, thickness_axis] = thickness[1]
        prisms.append(np.round(vertices @ transform[:3, :3].T + transform[:3, 3], 12).tolist())
    reconstructed = shapely.union_all(components)
    record = {
        'source_node': node_id,
        'name': node['name'],
        'source_mesh': definition['mesh'],
        'source_definition': node['definition'],
        'contact_source': 'Source BREP triangulated at 5 micrometers, angular tolerance .03 radians, no decimation.',
        'plane_axes': plane_axes,
        'thickness_axis': thickness_axis,
        'face_plane': face_plane,
        'thickness_bounds': thickness,
        'polygons': [np.asarray(p.exterior.coords[:-1]).tolist() for p in components],
        'prisms_world_m': prisms,
        'pivot_world_m': transform[:3, 3].tolist(),
        'axis_world': [0.0, .999995046737, -.003147459628],
        'source_volume_m3': definition['geometric_properties']['volume_m3'],
        'source_mass_kg': part_mass(node['name'], definition['geometric_properties']['volume_m3']),
        'verification': {
            'convex_component_count': len(components),
            'profile_area_m2': profile.area,
            'reconstruction_symmetric_difference_m2': profile.symmetric_difference(reconstructed).area,
            'outward_convexification_bound_m': .00002,
            'boundary_hausdorff_m': profile.boundary.hausdorff_distance(reconstructed.boundary),
            'simplification_tolerance_m': .000005,
            'planar_face_vertex_rounding_m': 1e-8,
        },
        'limitations': ['Extracted directly from BREP; linear tessellation tolerance 5 micrometers.',
                        'Gear contacts omit raised hubs; pawl contacts use only actual flat layer localY .003175 to .00635.',
                        'Pawl lower rounded/pocketed half is omitted, including .3175 mm of potential axial gear overlap.',
                        'Central shaft/pawl pivot holes are filled; collision pairs only target the opposing gear/pawl.',
                        'Convex parts may overlap and overfill the source outline by at most 20 micrometers.',
                        'Contact geometry has zero added mass; source part mass belongs to its moving body.'],
    }
    assert record['verification']['boundary_hausdorff_m'] <= .000025
    assert all(abs(p.area - p.convex_hull.area) < 1e-15 for p in components)
    profiles.append(record)
    print(node_id, len(triangles), profile.geom_type, record['verification'])
output = {
    'generator': 'sim/tools/extract_ratchet_profiles.py; shapely ' + shapely.__version__,
    'frame': 'Source CAD world meters, before robot.CAD_ROTATION; polygons remain definition-local.',
    'source': manifest['source'],
    'exact_brep_rest_intersection_volumes_m3': exact_overlaps,
    'profiles': profiles,
    'ownership': {
        '1/468/56': 'robot_claw_right_shaft', '1/468/60': 'robot_claw_left_shaft',
        '1/468/110': 'independent passive pawl on robot_claw_right_jaw',
        '1/468/112': 'independent passive pawl on robot_claw_left_jaw',
        '1/468/111': 'retained on robot_claw_right_jaw',
        '1/468/113': 'retained on robot_claw_left_jaw',
    },
}
(ASSETS / 'ratchet_profiles.json').write_text(json.dumps(output, indent=2) + '\n')
