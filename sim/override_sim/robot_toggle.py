"""Passive wall-toggle arms with CAD pivots and independently rolling end wheels."""
import xml.etree.ElementTree as ET

import numpy as np

from .field import numbers
from .robot import CAD_ROTATION


def add_toggle_arms(builder):
    """Add two unpowered arms; positive travel raises each arm from its lower stop."""
    metadata = {'arms': [], 'actuators': [], 'limitations': [
        'User-confirmed pink lower stops are represented by hinge limits at the supplied CAD rest pose.',
        'Only the lower stop is established; the numerical upper bound of 1e6 radians is never intended to engage.',
        'End flexwheels use exact CAD outer radius and width as rigid cylinders without urethane compliance.',
        'Bearing damping is estimated; neither arm nor end wheel has a motor or centering spring.',
        'Lower flipout stop pieces remain fixed at their supplied CAD pose; their optional deployment is not simulated.',
    ]}
    for side, arm_index, wheel_index, fittings in (
        ('left', 412, 421, [414, 415, 416, 417, 418, 419, 420, 422, 423]),
        ('right', 424, 431, [426, 427, 428, 429, 430, 432, 433, 434, 435]),
    ):
        arm_node = builder.nodes[f'1/{arm_index}']
        arm_transform = np.asarray(arm_node['world_transform'])
        pivot = arm_transform[:3, 3]
        arm_name = f'robot_toggle_{side}_arm'
        arm = builder.body(arm_name, 'robot', pivot, axis=(-1., 0., 0.),
                           limits=(0., 1e6), damping=.0005)
        arm.find('joint').set('solreflimit', '.004 1')
        builder.assign(arm_name, [f'1/{index}' for index in [arm_index, *fittings]])
        wheel_node = builder.nodes[f'1/{wheel_index}']
        wheel_transform = np.asarray(wheel_node['world_transform'])
        definition = builder.definitions[wheel_node['definition']]
        bounds = np.asarray(definition['local_bounds'])
        local_center = (bounds[:3] + bounds[3:]) / 2
        center = wheel_transform[:3, :3] @ local_center + wheel_transform[:3, 3]
        axis = wheel_transform[:3, 1]
        radius = max(abs(bounds[index]) for index in (0, 2, 3, 5))
        half_width = (bounds[4] - bounds[1]) / 2
        wheel_name = f'robot_toggle_{side}_wheel'
        wheel = builder.body(wheel_name, arm_name, center, axis=axis, damping=.00005)
        builder.assign(wheel_name, [f'1/{wheel_index}'])
        ET.SubElement(wheel, 'geom', name=f'{wheel_name}_contact', type='cylinder',
                      zaxis=numbers(CAD_ROTATION @ axis), size=numbers([radius, half_width]),
                      group='3', rgba='.9 .3 .1 .3', mass='0', contype='8', conaffinity='3',
                      friction='.9 .005 .0001')
        arm_midplane = pivot + arm_transform[:3, 0] * .003175
        wheel_mount = wheel_transform[:3, 3]
        builder.capsule(arm_name, f'{arm_name}_contact', arm_midplane, wheel_mount, .003175)
        metadata['arms'].append({'name': arm_name, 'wheel': wheel_name,
                                 'pivot_cad_m': pivot.tolist(), 'axis_cad': [-1., 0., 0.],
                                 'range_rad': [0., 1e6], 'wheel_axis_cad': axis.tolist(),
                                 'wheel_center_cad_m': center.tolist(),
                                 'wheel_radius_m': float(radius), 'wheel_width_m': float(2 * half_width)})
    return metadata
