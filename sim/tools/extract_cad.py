#!/usr/bin/env python3
"""Export reusable visual meshes and the complete STEP assembly, without fastener meshes.

Run with uv run --project sim --extra cad python sim/tools/extract_cad.py /path/to/field.STEP.
Requires cadquery-ocp, numpy, trimesh, fast-simplification, and rtree.
STEP geometry is millimeters; every output value is meters.
Meshes are visual assets. Physics should use separately authored collision proxies.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import trimesh

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.gp import gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence, TDF_Tool
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool


def label_id(label):
    entry = TCollection_AsciiString()
    TDF_Tool.Entry_s(label, entry)
    return entry.ToCString()


def label_name(label):
    attribute = TDataStd_Name()
    return attribute.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(), attribute) else ''


def matrix(location):
    transform = location.Transformation()
    return [[round(transform.Value(i, j) * (0.001 if j == 4 else 1), 12) for j in range(1, 5)]
            for i in range(1, 4)] + [[0, 0, 0, 1]]


def bounds(shape):
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, False)
    return [round(value * 0.001, 12) for value in box.Get()] if not box.IsVoid() else None


def world_bounds(local_bounds, location):
    if local_bounds is None:
        return None
    transform = location.Transformation()
    points = [gp_Pnt(x * 1000, y * 1000, z * 1000).Transformed(transform) for x, y, z in
              itertools.product(*[(local_bounds[i], local_bounds[i + 3]) for i in range(3)])]
    return [round(function(getattr(point, axis)() * 0.001 for point in points), 12)
            for function in (min, max) for axis in ('X', 'Y', 'Z')]


def semantic(name):
    if name.startswith('Tape-'):
        return 'tape'
    if name == '276-6904-001':
        return 'foam_tile'
    part = re.match(r'276-9250-(\d{3})', name)
    if not part:
        return 'other'
    number = part.group(1)
    return {'001': 'pin_half', '002': 'cup_half', '003': 'goal_top', '004': 'short_goal_base',
            '005': 'tall_goal_base', '006': 'apriltag', '021': 'toggle_mount', '022': 'toggle_mount',
            '023': 'toggle_bar', '024': 'toggle_end', '027': 'toggle_end_cap',
            '031': 'loader_slide', '032': 'loader_guide', '033': 'loader_washer',
            '034': 'loader_crossbar', '035': 'loader_washer', '036': 'loader_spacer'}.get(number, 'field_part')


def mesh_selected(name):
    kind = semantic(name)
    return kind not in {'other', 'apriltag', 'loader_washer', 'loader_spacer'}


def write_obj(shape, destination, tolerance_mm, angular_tolerance):
    mesher = BRepMesh_IncrementalMesh(shape, tolerance_mm, False, angular_tolerance, True)
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError(f'Tessellation failed for {destination.name}')
    vertices, triangles = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            offset = len(vertices)
            transform = location.Transformation()
            for i in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(i).Transformed(transform)
                vertices.append((point.X() * 0.001, point.Y() * 0.001, point.Z() * 0.001))
            for i in range(1, triangulation.NbTriangles() + 1):
                indices = list(triangulation.Triangle(i).Get())
                if face.Orientation() == TopAbs_REVERSED:
                    indices.reverse()
                triangles.append(tuple(offset + index for index in indices))
        explorer.Next()
    if not triangles:
        raise RuntimeError(f'No triangles for {destination.name}')
    original = trimesh.Trimesh(vertices=vertices, faces=np.array(triangles) - 1, process=True)
    raw_triangles = len(original.faces)
    target = min(raw_triangles, 3000)
    mesh = original
    sampled_error = 0.0
    sample = np.vstack((original.vertices[np.linspace(0, len(original.vertices) - 1,
                                                    min(5000, len(original.vertices)), dtype=int)],
                        trimesh.sample.sample_surface(original, 2000, seed=0)[0]))
    while target < raw_triangles:
        candidate = original.simplify_quadric_decimation(face_count=target, aggression=7)
        sampled_error = float(trimesh.proximity.closest_point(candidate, sample)[1].max())
        reverse_sample = trimesh.sample.sample_surface(candidate, 2000, seed=0)[0]
        sampled_error = max(sampled_error, float(trimesh.proximity.closest_point(original, reverse_sample)[1].max()))
        bound_error = float(np.abs(candidate.bounds - original.bounds).max())
        if sampled_error <= 0.00075 and bound_error <= 0.0005:
            mesh = candidate
            break
        target = min(target * 2, raw_triangles)
    else:
        sampled_error = 0.0
    with destination.open('w') as output:
        output.write('# Definition-local visual geometry, meters. Not a collision mesh.\n')
        for vertex in mesh.vertices:
            output.write('v ' + ' '.join(f'{value:.9g}' for value in vertex) + '\n')
        for triangle in mesh.faces + 1:
            output.write('f ' + ' '.join(map(str, triangle)) + '\n')
    return {'vertices': len(mesh.vertices), 'triangles': len(mesh.faces), 'source_triangles': raw_triangles,
            'sampled_surface_error_m': sampled_error, 'mesh_bounds': mesh.bounds.reshape(-1).tolist(),
            'source_mesh_bounds': original.bounds.reshape(-1).tolist()}



def export_profiles(destination, manifest):
    profiles = {}
    for definition_id, definition in manifest['definitions'].items():
        if definition['semantic'] not in {'cup_half', 'pin_half', 'goal_top', 'short_goal_base', 'tall_goal_base'}:
            continue
        mesh = trimesh.load(destination / definition['mesh'], process=True, force='mesh')
        sections = []
        for y in np.linspace(mesh.bounds[0, 1] + 0.0001, mesh.bounds[1, 1] - 0.0001, 25):
            segments = trimesh.intersections.mesh_plane(mesh, [0, 1, 0], [0, y, 0])
            radii = np.linalg.norm(segments[:, :, [0, 2]].reshape(-1, 2), axis=1)
            sections.append({'local_y': float(y), 'radius_quantiles':
                             np.quantile(radii, [0, 0.1, 0.5, 0.9, 1]).tolist() if len(radii) else None})
        profiles[definition_id] = {'name': definition['name'], 'mesh_bounds': mesh.bounds.reshape(-1).tolist(),
                                   'radial_profile': sections}
    (destination / 'profiles.json').write_text(json.dumps({
        'units': 'm', 'axis': 'definition-local Y', 'radius_quantiles': [0, 0.1, 0.5, 0.9, 1],
        'method': 'Visual triangle-plane intersections; min radius can include internal ribs and fastenings; '
                  'not proof of an unobstructed circular cavity. See manifest for mesh approximation error.',
        'parts': profiles}, indent=2) + '\n')


def extract(source, destination, tolerance_mm=1.0, angular_tolerance=0.3):
    destination.mkdir(parents=True, exist_ok=True)
    mesh_directory = destination / 'meshes'
    mesh_directory.mkdir(exist_ok=True)
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise RuntimeError(f'Cannot read {source}')
    document = TDocStd_Document(TCollection_ExtendedString('field'))
    if not reader.Transfer(document):
        raise RuntimeError('Cannot transfer STEP assembly')
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    definitions, nodes = {}, {}

    def visit(label, node_id, parent_location):
        definition_label = TDF_Label()
        if tool.IsReference_s(label):
            tool.GetReferredShape_s(label, definition_label)
        else:
            definition_label = label
        definition_id = label_id(definition_label)
        local_location = tool.GetLocation_s(label)
        world_location = parent_location.Multiplied(local_location)
        children = TDF_LabelSequence()
        tool.GetComponents_s(definition_label, children, False)
        name = label_name(definition_label)
        if definition_id not in definitions:
            definition = {'name': name, 'semantic': semantic(name), 'is_assembly': children.Length() > 0,
                          'instances': [], 'local_bounds': None, 'mesh': None}
            if not children.Length():
                shape = tool.GetShape_s(definition_label)
                definition['local_bounds'] = bounds(shape)
                if mesh_selected(name):
                    filename = re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_') + '_' + definition_id.replace(':', '-') + '.obj'
                    print(f'Meshing {name}', flush=True)
                    definition['mesh'] = 'meshes/' + filename
                    definition['mesh_statistics'] = write_obj(shape, mesh_directory / filename,
                                                               tolerance_mm, angular_tolerance)
                    definition['mesh_bounds'] = definition['mesh_statistics'].pop('mesh_bounds')
                    definition['source_mesh_bounds'] = definition['mesh_statistics'].pop('source_mesh_bounds')
            definitions[definition_id] = definition
        definition = definitions[definition_id]
        definition['instances'].append(node_id)
        node = {'name': name, 'definition': definition_id, 'children': [],
                'local_transform': matrix(local_location), 'world_transform': matrix(world_location)}
        if not children.Length():
            node['world_bounds'] = world_bounds(definition['local_bounds'], world_location)
        nodes[node_id] = node
        # GetComponents appends to the sequence. Calling it a second time duplicates every child.
        for i in range(1, children.Length() + 1):
            child_id = f'{node_id}/{i}'
            node['children'].append(child_id)
            visit(children.Value(i), child_id, world_location)

    root_labels = TDF_LabelSequence()
    tool.GetFreeShapes(root_labels)
    roots = [str(i) for i in range(1, root_labels.Length() + 1)]
    for i, node_id in enumerate(roots, 1):
        visit(root_labels.Value(i), node_id, TopLoc_Location())
    tiles = [node['world_bounds'] for node in nodes.values()
             if definitions[node['definition']]['semantic'] == 'foam_tile']
    floor_heights = sorted({round(box[4], 9) for box in tiles})
    manifest = {
        'schema_version': 1,
        'source': {'filename': source.name, 'sha256': digest.hexdigest()},
        'units': 'm',
        'coordinate_system': {'horizontal_right': '+X', 'horizontal_up': '-Z', 'vertical_up': '+Y',
                              'floor_top_y': floor_heights[0] if len(floor_heights) == 1 else None,
                              'matrix': 'row-major 4x4; p_parent = local_transform @ p_local'},
        'tessellation': {'linear_tolerance_m': tolerance_mm * 0.001, 'angular_tolerance_rad': angular_tolerance,
                         'purpose': 'visual only; collision proxies are authored separately',
                         'decimation': 'quadric; target 3000 faces, doubling until sampled surface error <=0.75mm and bounds error <=0.5mm; fixed sampling seed 0'},
        'roots': roots, 'nodes': nodes, 'definitions': definitions,
        'statistics': {'nodes': len(nodes), 'definitions': len(definitions),
                       'root_children': [len(nodes[root]['children']) for root in roots],
                       'visual_meshes': sum(definition['mesh'] is not None for definition in definitions.values()),
                       'foam_tile_instances': len(tiles), 'foam_tile_top_y_values': floor_heights},
    }
    output = destination / 'manifest.json'
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    export_profiles(destination, manifest)
    print(json.dumps(manifest['statistics'], indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'override_sim' / 'assets' / 'cad')
    parser.add_argument('--tolerance-mm', type=float, default=1.0)
    parser.add_argument('--angular-tolerance', type=float, default=0.3)
    arguments = parser.parse_args()
    extract(arguments.source, arguments.output, arguments.tolerance_mm, arguments.angular_tolerance)
