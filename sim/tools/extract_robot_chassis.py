"""Extract convex horizontal CAD sections for the chassis plate and goal guides.

uv run --project sim --extra cad --with trimesh==5.2.0 --with shapely==2.2.0 \
    --with mapbox-earcut==2.1.0 --with networkx==3.7 \
    python sim/tools/extract_robot_chassis.py
"""

import json
import hashlib
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon
from extract_loader import merge_convex_parts

ROOT = Path(__file__).resolve().parents[1] / "override_sim/assets/robot"
NODES = ("1/171", "1/201", "1/202", "1/197/1", "1/197/2", "1/77")


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    panels = []
    sources = []
    for node_id in NODES:
        node = manifest["nodes"][node_id]
        mesh = trimesh.load(
            ROOT / manifest["definitions"][node["definition"]]["mesh"], force="mesh"
        )
        mesh.apply_transform(node["world_transform"])
        levels = {float(mesh.bounds[0, 1]), float(mesh.bounds[1, 1])}
        for facet, normal, area in zip(
            mesh.facets, mesh.facets_normal, mesh.facets_area
        ):
            if abs(normal[1]) > 0.99999 and area > 1e-5:
                levels.add(float(np.mean(mesh.vertices[mesh.faces[facet].ravel(), 1])))
        levels = sorted(set(round(v, 7) for v in levels))
        count = 0
        for low, high in zip(levels, levels[1:]):
            if high - low < 1e-6:
                continue
            section = mesh.section(
                plane_origin=[0, (low + high) / 2, 0], plane_normal=[0, 1, 0]
            )
            if section is None:
                continue
            # Closed source section rings in the actual CAD X/Z plane.
            rings = [
                Polygon(np.asarray(points)[:, [0, 2]])
                for points in section.discrete
                if len(points) > 3
            ]
            rings = sorted(rings, key=lambda p: p.area, reverse=True)
            shape = None
            for ring in rings:
                shape = ring if shape is None else shape.symmetric_difference(ring)
            if shape is None:
                continue
            polygons = (
                list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
            )
            for polygon in polygons:
                # Sub-mm tessellation edges are simplified; all functional openings remain.
                polygon = polygon.simplify(0.0003, preserve_topology=True)
                polygon = Polygon(
                    polygon.exterior,
                    [r.coords for r in polygon.interiors if Polygon(r).area > 5e-5],
                )
                vertices, triangles = trimesh.creation.triangulate_polygon(
                    polygon, engine="earcut"
                )
                parts = merge_convex_parts([Polygon(vertices[t]) for t in triangles])
                for part in parts:
                    points = np.asarray(part.exterior.coords[:-1])
                    v = np.column_stack(
                        [points[:, 0], np.full(len(points), low), points[:, 1]]
                    )
                    panels.append(
                        {
                            "node": node_id,
                            "vertices": np.concatenate(
                                [v, v + [0, high - low, 0]]
                            ).tolist(),
                        }
                    )
                    count += 1
        sources.append(
            {
                "node": node_id,
                "mesh": manifest["definitions"][node["definition"]]["mesh"],
                "sha256": hashlib.sha256(
                    (
                        ROOT / manifest["definitions"][node["definition"]]["mesh"]
                    ).read_bytes()
                ).hexdigest(),
                "bounds_m": mesh.bounds.tolist(),
                "levels_m": levels,
                "panels": count,
            }
        )
    result = {
        "source": manifest["source"],
        "non_volume_source": {
            "1/197/1": "Original STEP COMPOUND contains 0 solids and 95 single-face shells; not a volume collider."
        },
        "tolerance_m": 0.0003,
        "filled_hole_max_area_m2": 5e-5,
        "sources": sources,
        "panels": panels,
        "limitations": [
            "Derived from packaged visual tessellation, not original BREP.",
            "Horizontal source sections are extruded between significant planar source levels; curved transitions between levels are approximated.",
            "Polygon boundaries simplified within .3mm; enclosed mounting holes under 50 mm² filled.",
            "Only baseplate, front guides, front crossbar and battery plate replace the former solid chassis box; existing rail proxies retained.",
        ],
    }
    (ROOT / "chassis_contacts.json").write_text(json.dumps(result, indent=2) + "\n")
    print([(s["node"], s["panels"], s["levels_m"]) for s in sources])


if __name__ == "__main__":
    main()
