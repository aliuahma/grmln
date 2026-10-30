"""Regenerate functional contact hulls from the original field STEP assembly."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from override_sim.field import ASSETS
import extract_cad as cad


def clip_triangles(triangles, planes):
    for normal, origin in planes:
        distances = (triangles - origin) @ normal
        inside = distances >= -1e-12
        retained = triangles[inside.all(axis=1)]
        crossing = triangles[inside.any(axis=1) & ~inside.all(axis=1)]
        clipped = []
        for polygon in crossing:
            output = []
            for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
                first_distance = np.dot(first - origin, normal)
                second_distance = np.dot(second - origin, normal)
                if first_distance >= -1e-12:
                    output.append(first)
                if first_distance < 0 < second_distance or second_distance < 0 < first_distance:
                    output.append(first + (second - first) * first_distance / (first_distance - second_distance))
            clipped.extend([output[0], output[index], output[index + 1]]
                           for index in range(1, len(output) - 1))
        triangles = np.concatenate([retained, np.array(clipped).reshape(-1, 3, 3)])
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1)
        triangles = triangles[area > 1e-18]
    return triangles



def source_meshes(source):
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    expected = json.loads((ASSETS / 'manifest.json').read_text())['source']['sha256']
    if digest != expected:
        raise ValueError('STEP source does not match the field manifest SHA256')
    reader = cad.STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(source)) != cad.IFSelect_RetDone:
        raise RuntimeError(f'Cannot read {source}')
    document = cad.TDocStd_Document(cad.TCollection_ExtendedString('contacts'))
    if not reader.Transfer(document):
        raise RuntimeError('Cannot transfer STEP assembly')
    tool = cad.XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    meshes = {}
    for kind, definition in [('pin', '0:1:1:65'), ('cup', '0:1:1:62'), ('goal', '0:1:1:34')]:
        label = cad.TDF_Label()
        cad.TDF_Tool.Label_s(document.GetData(), definition, label, False)
        shape = tool.GetShape_s(label)
        cad.BRepMesh_IncrementalMesh(shape, .025, False, .08, True).Perform()
        vertices, faces = [], []
        explorer = cad.TopExp_Explorer(shape, cad.TopAbs_FACE)
        while explorer.More():
            face = cad.TopoDS.Face_s(explorer.Current())
            location = cad.TopLoc_Location()
            triangulation = cad.BRep_Tool.Triangulation_s(face, location)
            if triangulation is not None:
                offset = len(vertices)
                transform = location.Transformation()
                for index in range(1, triangulation.NbNodes() + 1):
                    point = triangulation.Node(index).Transformed(transform)
                    vertices.append((point.X() * .001, point.Y() * .001, point.Z() * .001))
                for index in range(1, triangulation.NbTriangles() + 1):
                    indices = list(triangulation.Triangle(index).Get())
                    if face.Orientation() == cad.TopAbs_REVERSED:
                        indices.reverse()
                    faces.append(tuple(offset + vertex - 1 for vertex in indices))
            explorer.Next()
        meshes[kind] = (definition, trimesh.Trimesh(vertices=vertices, faces=faces, process=True))
    return digest, meshes


def extract(source, destination):
    digest, meshes = source_meshes(source)
    parts = {'source': {'filename': source.name, 'sha256': digest,
                        'linear_deflection_m': .000025, 'angular_deflection_rad': .08,
                        'method': 'Unreduced STEP face tessellation, clipped into convex contact partitions.'}}
    for kind, sections, sectors in [
        ('pin', [.007, .008, .0085, .0095, .067], 8),
        ('cup', [.0135, .075, .081, .08275], 24),
        ('goal', [.009, .075, .0765, .077, .078], 48),
    ]:
        definition, mesh = meshes[kind]
        triangles = mesh.triangles
        bounds = [float(mesh.bounds[0, 1]), *sections, float(mesh.bounds[1, 1])]
        pieces = []
        for index in range(sectors):
            first = 2 * math.pi * index / sectors
            second = 2 * math.pi * (index + 1) / sectors
            angular = clip_triangles(triangles, [
                (np.array([-math.sin(first), 0, math.cos(first)]), np.zeros(3)),
                (np.array([math.sin(second), 0, -math.cos(second)]), np.zeros(3)),
            ])
            for low, high in zip(bounds, bounds[1:]):
                normal_y = np.cross(angular[:, 1] - angular[:, 0],
                                    angular[:, 2] - angular[:, 0])[:, 1]
                lower_face = np.all(np.abs(angular[:, :, 1] - low) < 1e-12, axis=1)
                upper_face = np.all(np.abs(angular[:, :, 1] - high) < 1e-12, axis=1)
                # A planar ledge belongs to the band containing the solid behind its outward normal.
                discard = (lower_face & (normal_y > 0)) | (upper_face & (normal_y < 0))
                owned = angular[~discard]
                points = clip_triangles(owned, [
                    (np.array([0, 1, 0]), np.array([0, low, 0])),
                    (np.array([0, -1, 0]), np.array([0, high, 0])),
                ]).reshape(-1, 3)
                pieces.append(points[ConvexHull(points).vertices].tolist())
        parts[kind] = {'source_definition': definition, 'axis': 'definition-local Y',
                       'axial_boundaries_m': bounds, 'angular_sectors': sectors, 'pieces': pieces}
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(parts, separators=(',', ':')) + '\n')
    temporary.replace(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=ASSETS / 'piece_contacts.json')
    arguments = parser.parse_args()
    extract(arguments.source, arguments.output)
