"""Claw plate contacts preserve the actual CAD void that capsule proxies filled."""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from override_sim.field import numbers
from override_sim.robot import ROBOT_ASSETS


def test_exact_jaw_profiles_preserve_source_holes_and_material():
    profile = json.loads((ROBOT_ASSETS / 'jaw_profiles.json').read_text())
    manifest = json.loads((ROBOT_ASSETS / 'manifest.json').read_text())
    assert profile['source_file'] == manifest['source']
    definitions = {manifest['nodes'][node]['definition'] for node in profile['equivalent_nodes']}
    assert len(definitions) == 1
    definition = manifest['definitions'][next(iter(definitions))]
    vertices = np.asarray([point for polygon in profile['polygons'] for point in polygon])
    bounds = np.asarray(definition['local_bounds'])
    np.testing.assert_allclose(vertices.min(axis=0), bounds[[0, 2]], atol=.0000251, rtol=0)
    np.testing.assert_allclose(vertices.max(axis=0), bounds[[3, 5]], atol=.0000251, rtol=0)
    assert profile['holes_preserved'] == 5
    assert profile['boundary_hausdorff_m'] < .0000251
    assert abs(profile['profile_volume_m3'] / profile['source_volume_m3'] - 1) < .001
    assert profile['regression_void_exact_distance_m'] > .0022

    root = ET.Element('mujoco')
    assets = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    for index, polygon in enumerate(profile['polygons']):
        vertices = [[point[0], thickness, point[1]]
                    for thickness in profile['thickness_bounds'] for point in polygon]
        name = f'plate_{index}'
        ET.SubElement(assets, 'mesh', name=name, vertex=numbers(np.asarray(vertices).ravel()))
        ET.SubElement(world, 'geom', type='mesh', mesh=name, contype='1', conaffinity='2')
    body = ET.SubElement(world, 'body')
    ET.SubElement(body, 'freejoint')
    ET.SubElement(body, 'geom', type='sphere', size='.0001', mass='.01', contype='2', conaffinity='1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    for point, expected_contact in [(profile['regression_void_point_local_m'], False),
                                    (profile['material_point_local_m'], True)]:
        data.qpos[:3] = point
        mujoco.mj_forward(model, data)
        assert bool(data.ncon) == expected_contact
