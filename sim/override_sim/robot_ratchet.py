"""Passive CAD pawls and gear-tooth contacts for the intake jaw carriers."""
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

from .field import numbers
from .robot import CAD_ROTATION, ROBOT_ASSETS


def add_ratchet(builder, side, jaw_name, shaft_name, pawl_node, gear_node, axis):
    profiles = json.loads((ROBOT_ASSETS / 'ratchet_profiles.json').read_text())
    profile = next(p for p in profiles['profiles'] if p['source_node'] == pawl_node)
    pivot = np.asarray(profile['pivot_world_m'])
    name = f'robot_claw_{side}_pawl'
    pawl = builder.body(name, jaw_name, pivot, axis=axis, limits=(-.6, .6), damping=.000001)
    initial_deflection = .123 if side == 'left' else -.091
    quaternion = np.r_[math.cos(initial_deflection / 2),
                       math.sin(initial_deflection / 2) * (CAD_ROTATION @ axis)]
    pawl.set('quat', numbers(quaternion))
    pawl.find('joint').set('stiffness', '.002')
    pawl.find('joint').set('springref', str(-initial_deflection))
    pawl.find('joint').set('range', numbers([-.6 - initial_deflection, .6 - initial_deflection]))
    builder.assign(name, [pawl_node])
    geom_groups = []
    for source, owner, contype, affinity in [(gear_node, shaft_name, 64, 128), (pawl_node, name, 128, 64)]:
        body, body_pivot = builder.bodies[owner]
        profile = next(p for p in profiles['profiles'] if p['source_node'] == source)
        geoms = []
        for index, vertices in enumerate(profile['prisms_world_m']):
            mesh_name = f'robot_ratchet_{side}_{source.rsplit("/", 1)[-1]}_{index}'
            local = (CAD_ROTATION @ (np.asarray(vertices) - body_pivot).T).T
            ET.SubElement(builder.asset, 'mesh', name=mesh_name, vertex=numbers(local.ravel()))
            ET.SubElement(body, 'geom', name=mesh_name, type='mesh', mesh=mesh_name,
                          contype=str(contype), conaffinity=str(affinity), mass='0',
                          group='3', rgba='.3 .8 .8 .4', condim='3', friction='.3 .001 .0001',
                          solref='.002 1', solimp='.99 .999 .0001')
            geoms.append(mesh_name)
        geom_groups.append(geoms)
    return {'body': name, 'pawl_source': pawl_node, 'gear_source': gear_node,
            'gear_contact_geom_count': len(geom_groups[0]), 'pawl_contact_geom_count': len(geom_groups[1]),
            'contact_masks': {'gear': [64, 128], 'pawl': [128, 64]},
            'spring_nm_per_rad_assumed': .002, 'spring_reference_rad': -initial_deflection,
            'initial_deflection_from_cad_rad': initial_deflection,
            'initial_deflection_reason': 'Nearest clear hinge pose; exact source CAD has intersecting gear/pawl solids at assembly rest.',
            'travel_limits_rad_assumed': [-.6, .6],
            'limitations': ['Pawl elastic return, damping and travel stops are unmeasured estimates.',
                            'Ratchet direction follows CAD tooth contacts; no one-way constraint is imposed.']}
