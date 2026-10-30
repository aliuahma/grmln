#!/usr/bin/env python3
"""Export definition-local robot visuals and the complete STEP assembly in meters.

Run uv run --project sim --extra cad python sim/tools/extract_robot.py robot.stp.
Original STEP files are inputs only. Meshes are visual assets, not collision geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.gp import gp_Trsf

from extract_cad import bounds, label_id, label_name, matrix, world_bounds, write_obj


EXCLUDED = re.compile(r'screw|\bnut\b|spacer|washer|\binsert\b|collar|standoff|bearing', re.I)
GENERIC = {'SHELL', 'COMPOUND', 'SOLID'}


def analytic_cylinders(shape):
    cylinders = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        surface = BRepAdaptor_Surface(TopoDS.Face_s(explorer.Current()))
        if surface.GetType() == GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            point, axis = cylinder.Location(), cylinder.Axis().Direction()
            cylinders.append({'radius': cylinder.Radius() * 0.001,
                              'point': [point.X() * 0.001, point.Y() * 0.001, point.Z() * 0.001],
                              'axis': [axis.X(), axis.Y(), axis.Z()]})
        explorer.Next()
    return cylinders


def roller_profiles(shape):
    rollers = []
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_index = 0
    while solids.More():
        solid = solids.Current()
        surfaces = []
        faces = TopExp_Explorer(solid, TopAbs_FACE)
        while faces.More():
            surfaces.append(BRepAdaptor_Surface(TopoDS.Face_s(faces.Current())))
            faces.Next()
        bores = [surface.Cylinder() for surface in surfaces if surface.GetType() == GeomAbs_Cylinder
                 and abs(surface.Cylinder().Radius() - 3.06) < 1e-6]
        barrels = [surface for surface in surfaces if 'BSplineSurface' in str(surface.GetType())]
        if len(surfaces) == 16 and len(bores) == 1 and len(barrels) == 1:
            bore = bores[0]
            point, direction = bore.Location(), bore.Axis().Direction()
            origin = np.array([point.X(), point.Y(), point.Z()]) * 0.001
            axis = np.array([direction.X(), direction.Y(), direction.Z()])
            transverse_x = np.array([0., 0., 1.])
            transverse_y = np.cross(axis, transverse_x)
            transverse_y /= np.linalg.norm(transverse_y)
            rotation = np.array([transverse_x, transverse_y, axis])
            transform = gp_Trsf()
            transform.SetValues(*np.column_stack((rotation, -rotation @ (origin * 1000))).reshape(-1).tolist())
            aligned = BRepBuilderAPI_Transform(solid, transform, True).Shape()
            box = Bnd_Box()
            BRepBndLib.AddOptimal_s(aligned, box, False, False)
            limits = np.array(box.Get()) * 0.001
            center = origin + axis * ((limits[2] + limits[5]) / 2)
            samples = []
            barrel = barrels[0]
            for u in np.linspace(barrel.FirstUParameter(), barrel.LastUParameter(), 65):
                for v in np.linspace(barrel.FirstVParameter(), barrel.LastVParameter(), 129):
                    point = barrel.Value(float(u), float(v))
                    delta = np.array([point.X(), point.Y(), point.Z()]) * 0.001 - center
                    axial = float(delta @ axis)
                    samples.append([axial, float(np.linalg.norm(delta - axial * axis))])
            samples = np.array(samples)
            profile = []
            for axial in np.linspace(samples[:, 0].min(), samples[:, 0].max(), 21):
                nearby = samples[np.abs(samples[:, 0] - axial) <= 0.0001]
                profile.append({'axial_m': float(axial),
                                'outer_radius_m': float(nearby[:, 1].max()) if len(nearby) else None})
            rollers.append({'solid_index': solid_index, 'center': center.tolist(), 'axis': axis.tolist(),
                            'half_length_m': float((limits[5] - limits[2]) / 2),
                            'transverse_bounds_m': limits[[0, 1, 3, 4]].tolist(),
                            'sampled_outer_radius_max_m': float(samples[:, 1].max()),
                            'sampled_profile': profile})
        solid_index += 1
        solids.Next()
    if len(rollers) != 18:
        raise ValueError(f'Expected 18 omniwheel tire solids, found {len(rollers)}')
    return {'definition': '0:1:1:7', 'frame': 'definition local, meters',
            'method': 'Actual 16-face tire solids with a 3.06mm cylindrical bore. Axial bounds from '
                      'BRepBndLib.AddOptimal in roller-axis frame without triangulation. '
                      'Outer barrel BSpline surfaces sampled 65x129 per tire.', 'rollers': rollers}


def geometric_properties(shape):
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    center = properties.CentreOfMass()
    inertia = properties.MatrixOfInertia()
    return {
        'volume_m3': properties.Mass() * 1e-9,
        'volume_centroid_m': [center.X() * 0.001, center.Y() * 0.001, center.Z() * 0.001],
        'volume_inertia_m5': [[inertia.Value(i, j) * 1e-15 for j in range(1, 4)] for i in range(1, 4)],
        'meaning': 'Uniform volume moments about centroid. No material density or physical mass assigned.',
    }


def extract(source, destination, tolerance_mm=0.5, angular_tolerance=0.3, cylinders=None):
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
    document = TDocStd_Document(TCollection_ExtendedString('robot'))
    if not reader.Transfer(document):
        raise RuntimeError('Cannot transfer STEP assembly')
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    definitions, nodes, shapes = {}, {}, {}

    def visit(label, node_id, parent_location, ancestors):
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
        # Imported generic leaf labels inherit their actual part identity from the nearest named assembly.
        identity = next((ancestor for ancestor in reversed(ancestors) if ancestor not in GENERIC), name) if name in GENERIC else name
        hardware_identity = re.sub(r'no insert', '', identity, flags=re.I)
        hardware_identity = re.sub(r'bearing (?=holder|mount|bracket)', '', hardware_identity, flags=re.I)
        excluded = bool(EXCLUDED.search(hardware_identity))
        if definition_id not in definitions:
            definition = {'name': name, 'part_identity': identity, 'is_assembly': children.Length() > 0,
                          'instances': [], 'local_bounds': None, 'mesh': None,
                          'visual_exclusion': 'small hardware' if excluded else None}
            if not children.Length():
                shape = tool.GetShape_s(definition_label)
                definition['local_bounds'] = bounds(shape)
                if not excluded:
                    shapes[definition_id] = shape
                    definition['geometric_properties'] = geometric_properties(shape)
                    definition['analytic_cylinders'] = analytic_cylinders(shape)
            definitions[definition_id] = definition
        definitions[definition_id]['instances'].append(node_id)
        node = {'name': name, 'definition': definition_id, 'children': [],
                'local_transform': matrix(local_location), 'world_transform': matrix(world_location)}
        if not children.Length():
            node['world_bounds'] = world_bounds(definitions[definition_id]['local_bounds'], world_location)
        nodes[node_id] = node
        for i in range(1, children.Length() + 1):
            child_id = f'{node_id}/{i}'
            node['children'].append(child_id)
            visit(children.Value(i), child_id, world_location, ancestors + [name])

    root_labels = TDF_LabelSequence()
    tool.GetFreeShapes(root_labels)
    roots = [str(i) for i in range(1, root_labels.Length() + 1)]
    for i, node_id in enumerate(roots, 1):
        visit(root_labels.Value(i), node_id, TopLoc_Location(), [])
    manifest = {
        'schema_version': 1, 'source': {'filename': source.name, 'sha256': digest.hexdigest()},
        'units': 'm', 'coordinate_system': {
            'frame': 'Original robot STEP assembly frame; no guessed up or forward direction.',
            'matrix': 'row-major 4x4; p_parent = local_transform @ p_local',
            'mesh_frame': 'Definition local. Apply node world_transform once.',
        },
        'tessellation': {'linear_tolerance_m': tolerance_mm * 0.001, 'angular_tolerance_rad': angular_tolerance,
                         'purpose': 'visual only; collision proxies are authored separately',
                         'decimation': 'quadric; target 3000 faces, doubling until sampled bidirectional surface error <=0.75mm and bounds error <=0.5mm; sampling seed 0',
                         'error_scope': 'Sampled error relative to tessellation, not an analytic CAD Hausdorff bound.'},
        'analytic_geometry': {'frame': 'Definition local, meters.',
                              'cylinders': 'OCP analytic cylindrical surfaces. Duplicates are retained. Includes holes and rounded contours; these are candidate axes, not established joints.'},
        'roots': roots, 'nodes': nodes, 'definitions': definitions,
    }
    output = destination / 'manifest.json'
    if '0:1:1:7' in shapes:
        profiles = roller_profiles(shapes['0:1:1:7'])
        (destination / 'roller_profiles.json').write_text(json.dumps(profiles, indent=2) + '\n')
    for definition_id, shape in shapes.items():
        definition = definitions[definition_id]
        filename = re.sub(r'[^A-Za-z0-9_-]+', '_', definition['part_identity']).strip('_') + '_' + definition_id.replace(':', '-') + '.obj'
        print(f'Meshing {definition_id} {definition["part_identity"]}', flush=True)
        statistics = write_obj(shape, mesh_directory / filename, tolerance_mm, angular_tolerance)
        definition['mesh'] = 'meshes/' + filename
        definition['mesh_bounds'] = statistics.pop('mesh_bounds')
        definition['source_mesh_bounds'] = statistics.pop('source_mesh_bounds')
        definition['mesh_statistics'] = statistics
        output.write_text(json.dumps(manifest, indent=2) + '\n')
    if cylinders:
        evidence = json.loads(cylinders.read_text())
        for definition_id, definition in definitions.items():
            if definition_id in shapes:
                definition['analytic_cylinders'] = evidence.get(definition_id, [])
    manifest['statistics'] = {'nodes': len(nodes), 'definitions': len(definitions), 'visual_meshes': len(shapes),
                              'visual_instances': sum(len(definitions[key]['instances']) for key in shapes)}
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['statistics']), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'override_sim' / 'assets' / 'robot')
    parser.add_argument('--tolerance-mm', type=float, default=0.5)
    parser.add_argument('--angular-tolerance', type=float, default=0.3)
    parser.add_argument('--cylinders', type=Path, help='Optional analytic cylinder evidence JSON in definition-local meters.')
    arguments = parser.parse_args()
    extract(arguments.source, arguments.output, arguments.tolerance_mm, arguments.angular_tolerance, arguments.cylinders)
