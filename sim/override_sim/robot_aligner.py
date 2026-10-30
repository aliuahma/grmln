"""Convex contact prisms for the notched match-load aligner and its brackets."""
import json
import xml.etree.ElementTree as ET

import numpy as np

from .field import numbers
from .robot import CAD_ROTATION, ROBOT_ASSETS


def add_aligner_contacts(builder):
    profiles = json.loads((ROBOT_ASSETS / 'aligner_profiles.json').read_text())
    body_name = 'robot_lift_upper_second'
    element, pivot = builder.bodies[body_name]
    contacts = []
    for profile in profiles['profiles']:
        source_nodes = [profile['source_node']]
        if profile['source_node'] == '1/438/396':
            source_nodes.append('1/438/397')
        for node_id in source_nodes:
            transform = np.asarray(builder.nodes[node_id]['world_transform'])
            for index, polygon in enumerate(profile['polygons']):
                vertices = []
                for thickness in profile['thickness_bounds']:
                    for point in polygon:
                        vertex = np.zeros(3)
                        vertex[profile['plane_axes']] = point
                        vertex[profile['thickness_axis']] = thickness
                        world = transform[:3, :3] @ vertex + transform[:3, 3]
                        vertices.append(CAD_ROTATION @ (world - pivot))
                name = f'robot_aligner_{node_id.rsplit("/", 1)[-1]}_{index}'
                ET.SubElement(builder.asset, 'mesh', name=name, vertex=numbers(np.asarray(vertices).ravel()))
                ET.SubElement(element, 'geom', name=name, type='mesh', mesh=name,
                              group='3', rgba='.9 .3 .1 .3', contype='8', conaffinity='3', mass='0')
                contacts.append(name)
    return {'body': body_name, 'contact_geoms': contacts,
            'source_nodes': ['1/438/410', '1/438/396', '1/438/397'],
            'limitations': profiles['limitations']}
