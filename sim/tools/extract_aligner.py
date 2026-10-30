"""Regenerate convex aligner contact profiles from packaged CAD meshes.

Run from the repository root with
uv run --project sim --with shapely python sim/tools/extract_aligner.py
"""
import json
from pathlib import Path

import trimesh
from shapely import Polygon, constrained_delaunay_triangles, unary_union


ASSETS = Path(__file__).resolve().parents[1] / 'override_sim' / 'assets' / 'robot'
TOLERANCE = .00025


def convex_pieces(outline):
    pieces = list(constrained_delaunay_triangles(outline).geoms)
    while True:
        candidates = []
        for i, first in enumerate(pieces):
            for j in range(i):
                combined = first.union(pieces[j])
                if combined.geom_type == 'Polygon' and abs(combined.convex_hull.area - combined.area) < 1e-12:
                    candidates.append((combined.area, i, j, combined.convex_hull))
        if not candidates:
            return pieces
        _, i, j, combined = max(candidates, key=lambda item: item[0])
        pieces = [piece for k, piece in enumerate(pieces) if k not in (i, j)] + [combined]


def main():
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    profiles = []
    for node_id, plane_axes in [('1/438/410', (0, 2)), ('1/438/396', (1, 2))]:
        node = manifest['nodes'][node_id]
        definition = manifest['definitions'][node['definition']]
        mesh = trimesh.load(ASSETS / definition['mesh'], process=False)
        faces = [Polygon(mesh.vertices[face][:, plane_axes]) for face in mesh.faces]
        projection = unary_union([face for face in faces if face.area > 1e-12])
        outline = Polygon(projection.exterior).simplify(TOLERANCE, preserve_topology=True)
        pieces = convex_pieces(outline)
        thickness_axis = next(axis for axis in range(3) if axis not in plane_axes)
        profiles.append({
            'source_node': node_id,
            'plane_axes': plane_axes,
            'thickness_axis': thickness_axis,
            'thickness_bounds': [float(mesh.vertices[:, thickness_axis].min()),
                                 float(mesh.vertices[:, thickness_axis].max())],
            'polygons': [list(map(list, piece.exterior.coords[:-1])) for piece in pieces],
        })
        assert abs(unary_union(pieces).area - outline.area) < 1e-12
        print(f'{node_id}: {len(pieces)} convex prisms')
    output = {
        'profiles': profiles,
        'simplification_tolerance_m': TOLERANCE,
        'limitations': ['Mounting holes are filled. Outline chord approximation is within 0.25 mm.'],
    }
    (ASSETS / 'aligner_profiles.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
