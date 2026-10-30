"""CAD-grounded robot bodies. All source coordinates are meters in the STEP frame."""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .field import numbers, transform_attrs

ROBOT_ASSETS = Path(__file__).resolve().parent / 'assets' / 'robot'
CAD_ROTATION = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


class RobotBuilder:
    def __init__(self, root, start=(0., 0., 0.), fixed=False, visual_mode='simple'):
        self.root = root
        self.manifest = json.loads((ROBOT_ASSETS / 'manifest.json').read_text())
        self.nodes = self.manifest['nodes']
        self.definitions = self.manifest['definitions']
        from .robot_visuals import RobotVisuals
        self.visuals = RobotVisuals(ROBOT_ASSETS, visual_mode)
        self.visual_meshes = {}
        self.asset = root.find('asset')
        if self.asset is None:
            self.asset = ET.SubElement(root, 'asset')
        self.actuator = root.find('actuator')
        if self.actuator is None:
            self.actuator = ET.SubElement(root, 'actuator')
        self.equality = root.find('equality')
        if self.equality is None:
            self.equality = ET.SubElement(root, 'equality')
        self.excluded = set()
        self.assignments = {}
        self.members = {}
        self.mass_report = {}
        self.mass_transfers = []
        x, y, yaw = start
        chassis = ET.SubElement(root.find('worldbody'), 'body', name='robot', pos=numbers([x, y, .001]),
                                quat=numbers([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
        if not fixed:
            ET.SubElement(chassis, 'freejoint', name='robot_free')
        self.bodies = {'robot': (chassis, np.zeros(3))}
        for key, definition in self.definitions.items():
            if definition.get('mesh'):
                path = self.visuals.mesh_path(key, definition['mesh'])
                if path is not None:
                    self.visual_meshes[key] = 'robot_cad_' + key.replace(':', '_')
                    ET.SubElement(self.asset, 'mesh', name=self.visual_meshes[key], file=str(path))

    def body(self, name, parent, pivot, axis=(1., 0., 0.), limits=None, damping=.02, armature=0.):
        pivot = np.asarray(pivot)
        parent_element, parent_pivot = self.bodies[parent]
        element = ET.SubElement(parent_element, 'body', name=name, pos=numbers(CAD_ROTATION @ (pivot - parent_pivot)))
        attrs = {'limited': 'false'} if limits is None else {'range': numbers(limits), 'limited': 'true'}
        ET.SubElement(element, 'joint', name=name, type='hinge', axis=numbers(CAD_ROTATION @ axis),
                      damping=str(damping), armature=str(armature), **attrs)
        self.bodies[name] = (element, pivot)
        return element

    def assign(self, body, prefixes):
        for prefix in prefixes:
            if prefix in self.assignments:
                raise ValueError(f'Duplicate robot part assignment: {prefix}')
            self.assignments[prefix] = body

    def connect(self, name, body1, body2, point):
        element, pivot = self.bodies[body1]
        ET.SubElement(self.equality, 'connect', name=name, body1=body1, body2=body2,
                      anchor=numbers(CAD_ROTATION @ (np.asarray(point) - pivot)), solref='.004 1', solimp='.99 .999 .0001')

    def capsule(self, body, name, start, end, radius, friction='.7 .005 .0001'):
        element, pivot = self.bodies[body]
        endpoints = np.r_[CAD_ROTATION @ (np.asarray(start) - pivot), CAD_ROTATION @ (np.asarray(end) - pivot)]
        ET.SubElement(element, 'geom', name=name, type='capsule', fromto=numbers(endpoints), size=str(radius),
                      group='3', rgba='.9 .3 .1 .3', contype='8', conaffinity='3', mass='0', friction=friction)

    def box(self, body, name, center, halfsize):
        element, pivot = self.bodies[body]
        ET.SubElement(element, 'geom', name=name, type='box', pos=numbers(CAD_ROTATION @ (np.asarray(center) - pivot)),
                      size=numbers(np.abs(CAD_ROTATION) @ halfsize), group='3', rgba='.9 .3 .1 .3',
                      contype='8', conaffinity='3', mass='0')

    def finish(self):
        weighted = {name: [] for name in self.bodies}
        for node_id, node in self.nodes.items():
            if node['children'] or any(node_id == prefix or node_id.startswith(prefix + '/') for prefix in self.excluded):
                continue
            definition = self.definitions[node['definition']]
            matching = [prefix for prefix in self.assignments if node_id == prefix or node_id.startswith(prefix + '/')]
            owner = self.assignments[max(matching, key=len)] if matching else 'robot'
            self.members[node_id] = owner
            element, pivot = self.bodies[owner]
            transform = np.array(node['world_transform'])
            rotation = CAD_ROTATION @ transform[:3, :3]
            position = CAD_ROTATION @ (transform[:3, 3] - pivot)
            identity = definition['part_identity']
            if node['definition'] in self.visual_meshes:
                pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = position
                ET.SubElement(element, 'geom', name='robot_visual_' + node_id.replace('/', '_'), type='mesh',
                              mesh=self.visual_meshes[node['definition']], **transform_attrs(pose),
                              rgba=material_color(identity), group='1', contype='0', conaffinity='0', mass='0')
            properties = definition.get('geometric_properties')
            if properties and properties['volume_m3'] > 1e-12:
                volume = properties['volume_m3']
                mass = part_mass(identity, volume)
                center = position + rotation @ properties['volume_centroid_m']
                inertia = rotation @ np.array(properties['volume_inertia_m5']) @ rotation.T * (mass / volume)
                weighted[owner].append((mass, center, inertia))
        for owner, mass, center, inertia in self.mass_transfers:
            weighted[owner].append((-mass, center, -inertia))
        for name, element_and_pivot in self.bodies.items():
            parts = weighted[name]
            if not parts:
                continue
            mass = sum(part[0] for part in parts)
            center = sum(m * c for m, c, _ in parts) / mass
            inertia = sum(i + m * ((c - center) @ (c - center) * np.eye(3) - np.outer(c - center, c - center))
                          for m, c, i in parts)
            ET.SubElement(element_and_pivot[0], 'inertial', mass=str(mass), pos=numbers(center),
                          fullinertia=numbers([inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]))
            self.mass_report[name] = mass
        for name, (element, _) in self.bodies.items():
            if name not in self.mass_report:
                self.mass_report[name] = sum(float(geom.get('mass', '0')) for geom in element.findall('geom'))
        return {'source': self.manifest['source'], 'cad_rotation': CAD_ROTATION.tolist(),
                'visuals': self.visuals.metadata,
                'body_masses_kg': self.mass_report, 'part_bodies': self.members,
                'limitations': ['Material densities, infill, friction and damping are estimates.',
                               'Robot self-contact is omitted except isolated ratchet gear/pawl contacts; field and game-piece collisions remain enabled.',
                               'Motor cartridge defaults must be confirmed against the built robot.']}


def material_color(name):
    if name in ('Wanda', 'Darnyoudinkleberg'):
        return '.93 .93 .93 1'
    if 'PLA' in name:
        return '.94 .30 .65 1'
    if any(token in name for token in ('Poly', 'pOLY', 'Urethane', 'Wanda', 'Darn', 'Wheel', 'Omni', 'Motor', 'Brain', 'Battery')):
        return '.055 .06 .075 1'
    if 'License' in name:
        return '.05 .65 .85 1'
    return '.63 .66 .70 1'


def part_mass(name, volume):
    if 'V5 Motor' in name:
        return .155
    if 'Battery' in name:
        return .335
    if 'Brain' in name:
        return .220
    if 'Sensor' in name:
        return .020
    density = 2700 if any(token in name for token in ('Alu', 'Aluminum')) else 1200
    if 'PLA' in name:
        density = 700
    if any(token in name for token in ('Screw', 'Shaft', 'Bearing', 'Standoff', 'Nut Thin')):
        density = 7800
    return max(1e-6, volume * density)


def add_lift(builder):
    graph = json.loads((ROBOT_ASSETS / 'lift_graph.json').read_text())
    for body in graph['bodies']:
        limits = (-.03, 2.05) if body['name'] == 'robot_lift_lower_c' else None
        builder.body(body['name'], body['parent'], body['pivot'], limits=limits,
                     armature=.00002 if body['name'] == 'robot_lift_crank' else 0.)
        builder.assign(body['name'], body['source_prefixes'])
    spans = {
        'lower_c': ('G_hi', 'M_lo'), 'lower_first': ('G_lo', 'I_lo'),
        'lower_bridge': ('I_lo', 'I_hi'), 'lower_second': ('I_hi', 'M_gear_lo'),
        'midsection': ('M_lo', 'M_hi'), 'upper_c': ('M_hi', 'E_lo'),
        'upper_first': ('M_gear_hi', 'U_lo'), 'upper_bridge': ('U_lo', 'U_hi'),
        'upper_second': ('U_hi', 'E_hi'), 'end': ('E_lo', 'E_hi'),
        'crank': ('O', 'D'), 'rod': ('D', 'P'),
    }
    for body in graph['bodies']:
        first, second = spans[body['name'].removeprefix('robot_lift_')]
        for index, source in enumerate(body['core_node_prefixes'][:2]):
            a, b = np.array(graph['pivots'][first]), np.array(graph['pivots'][second])
            a[0] = b[0] = builder.nodes[source]['world_transform'][0][3]
            builder.capsule(body['name'], body['name'] + f'_contact_{index}', a, b, .008)
    builder.assign('robot_lift_end', [f'1/{i}' for i in [*range(440, 443), *range(445, 463), *range(464, 468)]])
    builder.assign('robot_lift_rod', [f'1/{i}' for i in (401, 403, 404, 405, 406, 407)])
    builder.assign('robot_lift_lower_c', ['1/408', '1/409'])
    for closure in graph['closures']:
        builder.connect(closure['name'], closure['body1'], closure['body2'], closure['point'])
    for coupling in graph['couplings']:
        ET.SubElement(builder.equality, 'joint', name='robot_lift_gears', joint1=coupling['joint1'],
                      joint2=coupling['joint2'], polycoef=numbers(coupling['polycoef']), solref='.004 1')
    for constraint in builder.equality:
        if constraint.get('name', '').startswith('robot_lift_'):
            constraint.set('solref', '.002 1')
            constraint.set('solimp', '.9999 .9999 .0001')
    ET.SubElement(builder.actuator, 'motor', name='robot_lift_drive', joint='robot_lift_crank',
                  gear='1', ctrllimited='true', ctrlrange='-14.7 14.7')
    return graph


def add_wheels(builder):
    profile = json.loads((ROBOT_ASSETS / 'roller_profiles.json').read_text())
    vertices = [[r * math.cos(a), r * math.sin(a), z]
                for z, r in [(-.0066, .0046), (-.00318, .005153), (0, .0053), (.00318, .005153), (.0066, .0046)]
                for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    vertex_text = numbers(np.array(vertices).ravel())
    ET.SubElement(builder.asset, 'mesh', name='robot_omni_roller', vertex=vertex_text)
    roller_model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><mesh name="roller" vertex="' + vertex_text + '"/></asset>'
        '<worldbody><body><freejoint/><geom type="mesh" mesh="roller" mass=".001"/></body></worldbody></mujoco>')
    principal_rotation = np.empty(9)
    mujoco.mju_quat2Mat(principal_rotation, roller_model.body_iquat[1])
    principal_rotation = principal_rotation.reshape(3, 3)
    roller_inertia = principal_rotation @ np.diag(roller_model.body_inertia[1]) @ principal_rotation.T
    wheels = []
    for i in range(1, 5):
        node = builder.nodes[f'1/{i}/2/1/2']
        transform = np.array(node['world_transform'])
        pivot = transform[:3, 3]
        name = f'robot_wheel_{i}'
        wheel = builder.body(name, 'robot', pivot, damping=.002, armature=.00002)
        builder.assign(name, [f'1/{i}/1', f'1/{i}/2'])
        for j, roller in enumerate(profile['rollers']):
            center = transform[:3, :3] @ roller['center'] + pivot
            axis = transform[:3, :3] @ roller['axis']
            element = builder.body(f'{name}_roller_{j}', name, center, axis=axis, damping=.000002)
            quat = np.empty(4)
            mujoco.mju_quatZ2Vec(quat, CAD_ROTATION @ axis)
            roller_rotation = np.empty(9)
            mujoco.mju_quat2Mat(roller_rotation, quat)
            roller_rotation = roller_rotation.reshape(3, 3)
            center_in_wheel = CAD_ROTATION @ (center - pivot) + roller_rotation @ roller_model.body_ipos[1]
            builder.mass_transfers.append((name, .001, center_in_wheel,
                                           roller_rotation @ roller_inertia @ roller_rotation.T))
            ET.SubElement(element, 'geom', type='mesh', mesh='robot_omni_roller', quat=numbers(quat),
                          group='3', rgba='.9 .3 .1 .3', mass='.001', contype='8', conaffinity='3',
                          friction='1 .002 .00001', solref='.004 1')
        ET.SubElement(builder.actuator, 'motor', name=name + '_drive', joint=name, ctrlrange='-2.8 2.8',
                      ctrllimited='true')
        wheels.append({'joint': name, 'actuator': name + '_drive', 'side': 'left' if pivot[0] > 0 else 'right'})
    return wheels


def build_robot_scene(layout=None, start=(0., 0., 0.), fixed=False, visual_mode='simple', claw_closure=None):
    if layout is None:
        root = ET.Element('mujoco', model='grmln-kiss-v2')
        ET.SubElement(root, 'compiler', angle='radian', balanceinertia='true', autolimits='true')
        ET.SubElement(root, 'option', timestep='.001', integrator='implicitfast', solver='Newton', iterations='60',
                      tolerance='1e-8', cone='elliptic')
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'global', offwidth='1280', offheight='960')
        default = ET.SubElement(root, 'default')
        ET.SubElement(default, 'geom', solref='.004 1', solimp='.95 .99 .001')
        world = ET.SubElement(root, 'worldbody')
        ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 .01', rgba='.22 .24 .27 1',
                      contype='1', conaffinity='4')
        ET.SubElement(world, 'light', pos='1 -1 3', dir='-1 1 -3', diffuse='.9 .9 .9')
    else:
        from .field import build_scene
        root = ET.fromstring(build_scene(layout, timestep=.001).xml)
    builder = RobotBuilder(root, start, fixed, visual_mode)
    graph = add_lift(builder)
    from .robot_aligner import add_aligner_contacts
    aligner = add_aligner_contacts(builder)
    wheels = add_wheels(builder)
    from .robot_claw import add_claw
    claw = add_claw(builder, **({'closure': claw_closure} if claw_closure is not None else {}))
    from .robot_toggle import add_toggle_arms
    toggle = add_toggle_arms(builder)
    from .robot_chassis import add_chassis_contacts
    chassis_contacts = add_chassis_contacts(builder)
    for side in (-1, 1):
        builder.box('robot', f'robot_rail_{side}', [side * .133, .08, 0], [.009, .035, .17])
    metadata = builder.finish()
    metadata.update({'chassis_contacts': chassis_contacts, 'wheels': wheels, 'lift': graph, 'claw': claw, 'toggle_arms': toggle, 'aligner': aligner})
    metadata['limitations'].extend(claw['limitations'])
    return ET.tostring(root, encoding='unicode'), metadata
