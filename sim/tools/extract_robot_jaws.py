"""Extract claw arm contacts from the exact STEP plate face, preserving its holes.

Run: uv run --project sim --extra cad --with shapely python sim/tools/extract_robot_jaws.py robot.stp
Runtime needs only the resulting packaged profiles, not the original STEP.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.gp import gp_Pnt
from shapely import Point, Polygon, unary_union

from extract_aligner import convex_pieces
from extract_robot import (STEPCAFControl_Reader, TCollection_ExtendedString, TDF_Label,
                           TDF_LabelSequence, TDocStd_Document, TopAbs_FACE, TopExp_Explorer,
                           TopLoc_Location, TopoDS, XCAFDoc_DocumentTool, geometric_properties)


ASSETS = Path(__file__).resolve().parents[1] / 'override_sim/assets/robot'
VOID_POINT = [.00125638, .00477243, .03189813]


def arm_shape(source):
    reader = STEPCAFControl_Reader()
    reader.ReadFile(str(source))
    document = TDocStd_Document(TCollection_ExtendedString('robot'))
    reader.Transfer(document)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    roots = TDF_LabelSequence()
    tool.GetFreeShapes(roots)
    label = roots.Value(1)
    for index in [468, 53]:
        definition = TDF_Label()
        if tool.IsReference_s(label):
            tool.GetReferredShape_s(label, definition)
        else:
            definition = label
        children = TDF_LabelSequence()
        tool.GetComponents_s(definition, children, False)
        label = children.Value(index)
    definition = TDF_Label()
    if tool.IsReference_s(label):
        tool.GetReferredShape_s(label, definition)
    else:
        definition = label
    return tool.GetShape_s(definition)


def top_face_outline(shape):
    BRepMesh_IncrementalMesh(shape, .005, False, .03, True).Perform()
    polygons = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangles = BRep_Tool.Triangulation_s(face, location)
        if triangles:
            vertices = []
            for index in range(1, triangles.NbNodes() + 1):
                point = triangles.Node(index).Transformed(location.Transformation())
                vertices.append([point.X() * .001, point.Y() * .001, point.Z() * .001])
            vertices = np.asarray(vertices)
            for index in range(1, triangles.NbTriangles() + 1):
                triangle = vertices[np.asarray(triangles.Triangle(index).Get()) - 1]
                if np.max(abs(triangle[:, 1] - .00635)) < 1e-8:
                    polygon = Polygon(triangle[:, [0, 2]])
                    if polygon.area > 1e-14:
                        polygons.append(polygon)
        explorer.Next()
    return unary_union(polygons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    expected = json.loads((ASSETS / 'manifest.json').read_text())['source']
    with args.source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    source_file = {'filename': args.source.name, 'sha256': digest}
    if source_file != expected:
        raise ValueError(f'STEP source does not match robot manifest: expected {expected}, got {source_file}')
    shape = arm_shape(args.source)
    vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*(np.asarray(VOID_POINT) * 1000))).Vertex()
    distance = BRepExtrema_DistShapeShape(shape, vertex)
    distance.Perform()
    exact = top_face_outline(shape)
    outline = exact.simplify(.000025, preserve_topology=True)
    pieces = convex_pieces(outline)
    material = outline.representative_point()
    output = {
        'source_node': '1/468/53',
        'equivalent_nodes': [f'1/468/{index}' for index in [50, 51, 52, 53]],
        'plane_axes': [0, 2], 'thickness_axis': 1, 'thickness_bounds': [0, .00635],
        'polygons': [list(map(list, piece.exterior.coords[:-1])) for piece in pieces],
        'source': 'Original STEP BREP planar face; 5 micrometer tessellation.',
        'simplification_tolerance_m': .000025,
        'boundary_hausdorff_m': exact.boundary.hausdorff_distance(outline.boundary),
        'source_volume_m3': geometric_properties(shape)['volume_m3'],
        'profile_volume_m3': outline.area * .00635,
        'holes_preserved': len(outline.interiors),
        'regression_void_point_local_m': VOID_POINT,
        'regression_void_exact_distance_m': distance.Value() * .001,
        'material_point_local_m': [material.x, .003175, material.y],
        'source_file': source_file,
    }
    assert unary_union(pieces).symmetric_difference(outline).area < 1e-12
    assert not outline.contains(Point(VOID_POINT[0], VOID_POINT[2]))
    assert abs(output['profile_volume_m3'] / output['source_volume_m3'] - 1) < .001
    (ASSETS / 'jaw_profiles.json').write_text(json.dumps(output, indent=2) + '\n')
    print(f'{len(pieces)} convex prisms per arm; exact void clearance {distance.Value():.4f} mm')


if __name__ == '__main__':
    main()
