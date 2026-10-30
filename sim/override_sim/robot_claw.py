"""CAD claw hinges, passive jaw carriers and geared intake roller transmission."""
from dataclasses import dataclass
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

from .field import numbers
from .robot import CAD_ROTATION, ROBOT_ASSETS
from .robot_ratchet import add_ratchet


CLAW = '1/468'
JAW_AXIS = np.array([0., .999995046737, -.003147459628])
MOUNT_ORIGIN = np.array([0., .0635, .115637])


@dataclass(frozen=True)
class ClawClosure:
    """Proposed paired torsion springs; zero disables jaw centering, retaining pawl return assumptions."""
    stiffness_nm_per_rad: float = .04
    preload_nm: float = .085

    def __post_init__(self):
        values = (self.stiffness_nm_per_rad, self.preload_nm)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError('Claw closure stiffness and preload must be finite and nonnegative')
        if self.stiffness_nm_per_rad == 0 and self.preload_nm != 0:
            raise ValueError('A zero-stiffness closure cannot have preload')


def _closure_springs(builder, closure):
    for side, sign in [('left', -1), ('right', 1)]:
        element, _ = builder.bodies[f'robot_claw_{side}_jaw']
        joint = element.find('joint')
        reference = sign * closure.preload_nm / closure.stiffness_nm_per_rad if closure.stiffness_nm_per_rad else 0.
        joint.set('stiffness', str(closure.stiffness_nm_per_rad))
        joint.set('springref', str(reference))
    return {
        'mechanism': 'Proposed paired mount-anchored torsion springs; not recovered CAD hardware.',
        'stiffness_nm_per_rad_assumed': closure.stiffness_nm_per_rad,
        'preload_nm_assumed': closure.preload_nm,
        'opening_torque_at_0_4_rad_nm': closure.preload_nm + .4 * closure.stiffness_nm_per_rad,
        'opening_torque_at_0_65_rad_nm': closure.preload_nm + .65 * closure.stiffness_nm_per_rad,
        'calibrated': False, 'source_cad_hardware': False,
        'torque_reference': 'Comparable to the previous 3 N direct-band assumption over 0–0.4 rad opening; not a measured spring.',
        'damping': 'Existing passive jaw joint damping; no additional spring damping.',
        'no_springs_configuration': 'ClawClosure(stiffness_nm_per_rad=0, preload_nm=0)',
    }


def _roller_contacts(builder, body, indices):
    element, pivot = builder.bodies[body]
    for index in indices:
        node = builder.nodes[f'{CLAW}/{index}']
        definition = builder.definitions[node['definition']]
        bounds = definition['local_bounds']
        transform = np.asarray(node['world_transform'])
        center = (np.array(bounds[:3]) + np.array(bounds[3:])) / 2
        axis = CAD_ROTATION @ transform[:3, 1]
        position = CAD_ROTATION @ (transform[:3, :3] @ center + transform[:3, 3] - pivot)
        radius = max(abs(bounds[i]) for i in (0, 2, 3, 5))
        half_length = (bounds[4] - bounds[1]) / 2
        ET.SubElement(element, 'geom', name=f'robot_intake_tire_{index}', type='cylinder',
                      pos=numbers(position), zaxis=numbers(axis), size=numbers([radius, half_length]),
                      mass='0', group='3', rgba='.9 .3 .1 .3', contype='8', conaffinity='3',
                      friction='1.2 .005 .0001', condim='4')


def _jaw_contacts(builder, body_name, source_indices):
    profile = json.loads((ROBOT_ASSETS / 'jaw_profiles.json').read_text())
    element, pivot = builder.bodies[body_name]
    for index in source_indices:
        transform = np.asarray(builder.nodes[f'{CLAW}/{index}']['world_transform'])
        for part, polygon in enumerate(profile['polygons']):
            vertices = []
            for thickness in profile['thickness_bounds']:
                for point in polygon:
                    local = np.array([point[0], thickness, point[1]])
                    world = transform[:3, :3] @ local + transform[:3, 3]
                    vertices.append(CAD_ROTATION @ (world - pivot))
            name = f'robot_claw_arm_{index}_{part}'
            ET.SubElement(builder.asset, 'mesh', name=name, vertex=numbers(np.asarray(vertices).ravel()))
            ET.SubElement(element, 'geom', name=name, type='mesh', mesh=name,
                          mass='0', group='3', rgba='.9 .3 .1 .3', contype='8', conaffinity='3')


def add_claw(builder, end_body='robot_lift_end', closure=ClawClosure()):
    """Fix the claw at its CAD rest pose and retain the roller transmission."""
    parent, parent_pivot = builder.bodies[end_body]
    mount = ET.SubElement(parent, 'body', name='robot_claw_mount',
                          pos=numbers(CAD_ROTATION @ (MOUNT_ORIGIN - parent_pivot)))
    builder.bodies['robot_claw_mount'] = (mount, MOUNT_ORIGIN)
    builder.assign('robot_claw_mount', [CLAW, '1/439'])
    builder.assign(end_body, ['1/463'])
    tendon = builder.root.find('tendon')
    if tendon is None:
        tendon = ET.SubElement(builder.root, 'tendon')
    shafts = []
    ratchets = []
    for side, pivot_x, arms, gear_nodes, shaft_node, tires in (
        ('left', .019456400019, [50, 53, 72, 75, 76, 77, 78, 79, 80, 81, 82, 83, 92, 94, 95, 96, 97, 98, 99, 100, 101, 68, 69, 113, 115, 117, 119],
         [60, 61, 62, 63, 64], 179, [135, 150, 153, 156, 169, 172]),
        ('right', -.018643599981, [51, 52, 73, 74, 84, 85, 86, 87, 88, 89, 90, 91, 93, 102, 103, 104, 105, 106, 107, 108, 109, 70, 71, 111, 114, 116, 118],
         [56, 57, 58, 59, 67], 178, [133, 139, 142, 146, 163, 166]),
    ):
        pivot = np.array([pivot_x, .09700096212, .115531302945])
        jaw_name = f'robot_claw_{side}_jaw'
        jaw = builder.body(jaw_name, 'robot_claw_mount', pivot, axis=JAW_AXIS,
                           limits=(-.12, .65) if side == 'left' else (-.65, .12), damping=.01)
        builder.assign(jaw_name, [f'{CLAW}/{index}' for index in arms])
        shaft_name = f'robot_claw_{side}_shaft'
        builder.body(shaft_name, 'robot_claw_mount', pivot, axis=JAW_AXIS, damping=.0002)
        builder.assign(shaft_name, [f'{CLAW}/{index}' for index in gear_nodes])
        shafts.append(shaft_name)
        pawl_index, gear_index = (112, 60) if side == 'left' else (110, 56)
        ratchets.append(add_ratchet(builder, side, jaw_name, shaft_name,
                                    f'{CLAW}/{pawl_index}', f'{CLAW}/{gear_index}', JAW_AXIS))
        roller_name = f'robot_claw_{side}_roller'
        roller_pivot = np.array(builder.nodes[f'{CLAW}/{shaft_node}']['world_transform'])[:3, 3]
        builder.body(roller_name, jaw_name, roller_pivot, axis=JAW_AXIS, damping=.0005)
        if side == 'left':
            roller_parts = [121, 126, 127, 128, 129, 131, 134, 135, 137, 149, 150, 151,
                            152, 153, 154, 155, 156, 157, 158, 159, 161, 168, 169, 170, 171, 172,
                            173, 176, 177, 179]
        else:
            roller_parts = [120, 122, 123, 124, 125, 130, 132, 133, 136, 138, 139, 140,
                            141, 142, 143, 144, 145, 146, 147, 148, 160, 162, 163, 164, 165, 166,
                            167, 174, 175, 178]
        builder.assign(roller_name, [f'{CLAW}/{index}' for index in roller_parts])
        _roller_contacts(builder, roller_name, tires)
        chain_name = f'robot_claw_{side}_chain'
        fixed = ET.SubElement(tendon, 'fixed', name=chain_name)
        for name, coefficient in ((roller_name, 1), (jaw_name, 1), (shaft_name, -1)):
            ET.SubElement(fixed, 'joint', joint=name, coef=str(coefficient))
        # Default equality impedance lets these low-inertia shafts slip under motor torque.
        # Use a stiff transmission at 1 ms without adding inertia to the passive carriers.
        ET.SubElement(builder.equality, 'tendon', name=f'{chain_name}_constraint', tendon1=chain_name,
                      polycoef='0 0 0 0 0', solref='.002 .5', solimp='.999 .999 .0001')
        _jaw_contacts(builder, jaw_name, [50, 53] if side == 'left' else [51, 52])
    closure_metadata = _closure_springs(builder, closure)
    ET.SubElement(builder.equality, 'joint', name='robot_intake_24t_gears', joint1=shafts[0],
                  joint2=shafts[1], polycoef='0 -1 0 0 0', solref='.002 .5', solimp='.999 .999 .0001')
    ET.SubElement(builder.actuator, 'motor', name='robot_intake_drive', joint=shafts[0],
                  ctrllimited='true', ctrlrange='-1.05 1.05', forcelimited='true', forcerange='-1.05 1.05')
    builder.box('robot_claw_mount', 'robot_claw_backplate_contact', [.0004064, .131, .0925], [.03175, .059, .001])
    return {
        'actuators': {'intake': 'robot_intake_drive'},
        'mount': 'Fixed to the lift end at the CAD rest orientation.',
        'intake_motor_count': 1, 'intake_cartridge_rpm_assumed': 200,
        'intake_stall_torque_nm': 1.05, 'intake_no_load_rad_s': 200 * 2 * math.pi / 60,
        'intake_gearing': {'opposed_pivot_gears': [24, 24], 'roller_chains': [8, 8]},
        'jaw_actuation': 'Passive hinges with proposed paired centering torsion springs and contacting CAD ratchet pawls; no jaw servos.',
        'ratchets': ratchets, 'closure_springs': closure_metadata,
        'jaw_contacts': {'source_nodes': [f'{CLAW}/{index}' for index in [50, 51, 52, 53]],
                         'profile_asset': 'jaw_profiles.json', 'holes_preserved': True,
                         'outline_tolerance_m': .000025, 'added_mass_kg': 0.},
        'roller_radius_m': .0254000001, 'roller_tire_count': 12,
        'wheel_material': {
            'cad_definition': builder.nodes[f'{CLAW}/135']['definition'],
            'cad_part_name': builder.definitions[builder.nodes[f'{CLAW}/135']['definition']]['name'],
            'hardness_shore_a_assumed': 40,
            'hardness_reference': 'User-specified assumption; not recovered from CAD or measured.',
            'compound_reference': 'CAD names custom urethane; exact formulation is unknown.',
            'sliding_friction_assumed': 1.2,
            'compression_model': 'Rigid cylinders with solver compliance; no calibrated force-displacement curve or contact-patch deformation.',
        },
        'limitations': [
            'Original pneumatic CAD part remains fixed to the lift end; hardware removal is unspecified.',
            'Paired centering springs are a proposed hardware addition; their stiffness and preload are uncalibrated estimates. Pawl return stiffness and travel limits are also estimates.',
            'Circular pivot inserts allow passive jaw swivel; shaft-to-arm retention is mediated by local pawl-tooth contacts.',
            'Flexwheel collision cylinders use exact CAD outer radius and width but do not deform.',
            'Single motor torque is bounded at the assumed 200 RPM cartridge stall torque; speed envelope belongs to controller.',
            'Gear and chain equalities approximate rigid transmission at the 1 ms timestep; physical backlash and compliance are unmeasured.',
        ],
    }
