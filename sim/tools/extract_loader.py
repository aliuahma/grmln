"""Derive convex loader wall panels from packaged CAD.

From the repository root:
PYTHONPATH=sim uv run --project sim --with trimesh --with shapely --with mapbox-earcut python sim/tools/extract_loader.py
"""
import json

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

from override_sim.field import ASSETS


SOURCE = '276-9250-031_Red_Loader_0-1-1-82.obj'
# The two main outer side faces and the planar outer front face in this source mesh.
WALL_FACETS = (329, 220, 159)


def merge_convex_parts(parts):
    """Join neighboring triangles only when their union remains convex."""
    changed = True
    while changed:
        changed = False
        for first in range(len(parts)):
            if changed:
                break
            for second in range(first + 1, len(parts)):
                if parts[first].distance(parts[second]) > 1e-7:
                    continue
                combined = parts[first].union(parts[second])
                hull = combined.convex_hull
                if hull.area <= combined.area + 1e-8:
                    parts[first] = hull
                    parts.pop(second)
                    changed = True
                    break
    return parts


def main():
    mesh = trimesh.load(ASSETS / 'meshes' / SOURCE, force='mesh')
    panels, facets = [], []
    for index in WALL_FACETS:
        normal = mesh.facets_normal[index]
        axis = np.eye(3)[np.argmin(abs(normal))]
        horizontal = np.cross(normal, axis)
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(normal, horizontal)
        faces = mesh.faces[mesh.facets[index]]
        triangles = [Polygon(np.column_stack([mesh.vertices[face] @ horizontal,
                                             mesh.vertices[face] @ vertical])) for face in faces]
        polygon = unary_union(triangles).buffer(1e-9).simplify(.0001, preserve_topology=True)
        if polygon.geom_type != 'Polygon':
            raise ValueError(f'CAD wall facet {index} is not a connected polygon')
        vertices, triangles = trimesh.creation.triangulate_polygon(polygon, engine='earcut')
        parts = merge_convex_parts([Polygon(vertices[triangle]) for triangle in triangles])
        plane = float(mesh.vertices[faces[0, 0]] @ normal)
        for part in parts:
            xy = np.array(part.exterior.coords[:-1])
            points = xy[:, 0, None] * horizontal + xy[:, 1, None] * vertical + plane * normal
            panels.append(np.concatenate([points, points - .003 * normal]).tolist())
        facets.append({'facet': index, 'holes': len(polygon.interiors), 'panels': len(parts),
                       'area': polygon.area})
    profile = {'source_mesh': SOURCE, 'source_bounds_m': mesh.bounds.tolist(),
               'planar_boundary_tolerance_m': .0001, 'main_wall_thickness_m': .003,
               'panels': panels, 'facets': facets}
    (ASSETS / 'loader_contacts.json').write_text(json.dumps(profile, indent=2) + '\n')
    print(f'{len(panels)} convex wall panels; side-wall holes and lower cutouts retained')


if __name__ == '__main__':
    main()
