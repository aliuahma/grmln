"""Rebuild display-only robot meshes. Run with uv run --project sim --extra cad."""
from pathlib import Path
import hashlib
import json

import numpy as np
import trimesh

ASSETS = Path(__file__).resolve().parents[1] / 'override_sim' / 'assets' / 'robot'
OUTPUT = ASSETS / 'meshes' / 'simple'
CHANNELS = {14: 13, 57: 56, 60: 59, 74: 73}
ANGLES = {25: 24, 104: 103, 109: 108, 112: 111}
MOTORS = {15, 65}
GEARS = {3, 35, 43, 86, 90, 141}
FLEXWHEELS = {99, 148}


def box(bounds):
    bounds = np.asarray(bounds)
    transform = np.eye(4)
    transform[:3, 3] = bounds.mean(axis=0)
    return trimesh.creation.box(extents=bounds[1] - bounds[0], transform=transform)


def extrusion(source, channel):
    low, high = source.bounds
    thickness = .0015875
    base_high = high.copy()
    base_high[1] = low[1] + thickness
    pieces = [box([low, base_high])]
    for side in [-1, 1] if channel else [-1]:
        side_low, side_high = low.copy(), high.copy()
        side_low[1] += thickness
        if side == -1:
            side_high[0] = low[0] + thickness
        else:
            side_low[0] = high[0] - thickness
        pieces.append(box([side_low, side_high]))
    return trimesh.util.concatenate(pieces)


def barrel(profile):
    axis = np.array(profile['axis'])
    transverse = np.cross(axis, [0., 0., 1.])
    transverse /= np.linalg.norm(transverse)
    other = np.cross(axis, transverse)
    samples = profile['sampled_profile']
    samples = [samples[index] for index in np.linspace(0, len(samples) - 1, 7).astype(int)]
    rings = [(sample['axial_m'], sample['outer_radius_m']) for sample in samples]
    rings = [(-profile['half_length_m'], rings[0][1]), *rings,
             (profile['half_length_m'], rings[-1][1])]
    vertices, faces = [], []
    segments = 12
    for axial, radius in rings:
        for angle in np.arange(segments) * 2 * np.pi / segments:
            vertices.append(np.array(profile['center']) + axis * axial
                            + radius * (transverse * np.cos(angle) + other * np.sin(angle)))
    for ring in range(len(rings) - 1):
        for index in range(segments):
            a = ring * segments + index
            b = ring * segments + (index + 1) % segments
            faces.extend([[a, b, a + segments], [b, b + segments, a + segments]])
    for offset, reverse in [(0, True), ((len(rings) - 1) * segments, False)]:
        for index in range(1, segments - 1):
            face = [offset, offset + index, offset + index + 1]
            faces.append(face[::-1] if reverse else face)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def wheel(source):
    components = trimesh.graph.connected_components(source.face_adjacency, min_len=3, engine='scipy')
    hub = source.submesh([max(components, key=len)], append=True, repair=False)
    hub.merge_vertices(digits_vertex=4)
    hub.update_faces(hub.unique_faces())
    hub.update_faces(hub.nondegenerate_faces())
    hub.remove_unreferenced_vertices()
    hub = hub.simplify_quadric_decimation(face_count=1800)
    profiles = json.loads((ASSETS / 'roller_profiles.json').read_text())
    result = trimesh.util.concatenate([hub, *(barrel(profile) for profile in profiles['rollers'])])
    result.vertices = np.clip(result.vertices, *source.bounds)
    return result


def surface_error(source, result):
    errors = []
    for left, right in [(source, result), (result, source)]:
        samples, _ = trimesh.sample.sample_surface(left, 512, seed=0)
        _, distances, _ = trimesh.proximity.closest_point(right, samples)
        errors.append(float(distances.max()))
    return {'sample_count_each_direction': 512, 'seed': 0,
            'source_to_simple_max_m': errors[0], 'simple_to_source_max_m': errors[1],
            'bounds_max_abs_m': float(np.max(np.abs(source.bounds - result.bounds)))}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    replacements = {}
    for key, definition in manifest['definitions'].items():
        index = int(key.split(':')[-1])
        if index not in {*CHANNELS, *ANGLES, *MOTORS, *GEARS, *FLEXWHEELS, 7}:
            continue
        source_path = ASSETS / definition['mesh']
        source = trimesh.load(source_path, force='mesh')
        if index in CHANNELS or index in ANGLES:
            result = extrusion(source, index in CHANNELS)
            method = 'Unperforated open U profile' if index in CHANNELS else 'Unperforated open L profile'
            companion = CHANNELS.get(index, ANGLES.get(index))
            companion_key = f'0:1:1:{companion}'
            companion_definition = manifest['definitions'][companion_key]
            replacements[companion_key] = {'mesh': None, 'source_mesh': companion_definition['mesh'],
                                           'reason': f'Hole walls omitted with unperforated {key} visual'}
        elif index in MOTORS:
            result = source.convex_hull.simplify_quadric_decimation(face_count=400)
            method = 'Rounded outer housing convex hull, small holes and recesses removed'
        elif index == 7:
            result = wheel(source)
            method = 'Decimated source hub plus 18 barrels sampled from CAD roller profiles'
        else:
            result = source.convex_hull
            method = 'Convex outer outline, bores and small surface detail removed'
        result.vertices = np.clip(result.vertices, *source.bounds)
        target = OUTPUT / (key.replace(':', '-') + '.obj')
        target.write_text(trimesh.exchange.obj.export_obj(result, include_normals=False, include_texture=False))
        replacements[key] = {'mesh': str(target.relative_to(ASSETS)), 'source_mesh': definition['mesh'],
                             'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                             'method': method, 'source_triangles': len(source.faces),
                             'simple_triangles': len(result.faces), 'source_bounds_m': source.bounds.tolist(),
                             'simple_bounds_m': result.bounds.tolist(), 'sampled_surface_error': surface_error(source, result)}
        print(key, len(source.faces), '->', len(result.faces), flush=True)
    source_count = simple_count = 0
    for node in manifest['nodes'].values():
        if node['children']:
            continue
        key = node['definition']
        definition = manifest['definitions'][key]
        if not definition.get('mesh'):
            continue
        source = trimesh.load(ASSETS / definition['mesh'], force='mesh')
        source_count += len(source.faces)
        replacement = replacements.get(key)
        simple_count += (replacement.get('simple_triangles', 0) if replacement else len(source.faces))
    metadata = {'schema_version': 1, 'purpose': 'Display meshes only; source mass and contact meshes remain authoritative',
                'generator': 'uv run --project sim --extra cad python sim/tools/generate_robot_visuals.py',
                'error_scope': '512 deterministic area-weighted samples in each direction; not a Hausdorff bound. '
                               'Unperforated profiles intentionally fill mounting holes. Motor hulls fill recesses.',
                'instanced_source_triangles': source_count, 'instanced_simple_triangles': simple_count,
                'definitions': replacements}
    (ASSETS / 'simple_visuals.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
