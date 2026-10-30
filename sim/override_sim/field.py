from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent / 'assets' / 'cad'
CAD_TO_FIELD = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, .2945892], [0, 0, 0, 1.]])
COLORS = {'red': '.88 .035 .10 1', 'blue': '0 .55 .86 1', 'yellow': '1 .84 .04 1',
          'dark': '.09 .10 .12 1', 'clear': '.72 .82 .86 .48', 'silver': '.62 .65 .69 1'}


class Layout(str, Enum):
    MATCH = 'university-match'
    SKILLS = 'university-skills'


@dataclass(frozen=True)
class SceneSpec:
    xml: str
    metadata: dict


def numbers(values):
    return ' '.join(f'{float(v):.10g}' for v in values)


def transform_attrs(transform):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(transform[:3, :3].reshape(9)))
    return {'pos': numbers(transform[:3, 3]), 'quat': numbers(q)}


def color(name):
    for token, value in [('Red', 'red'), ('199c', 'red'), ('Blue', 'blue'), ('299c', 'blue'),
                         ('Yellow', 'yellow'), ('Clear', 'clear')]:
        if token in name:
            return COLORS[value]
    return COLORS['dark']


class FieldBuilder:
    def __init__(self, layout: Layout, timestep: float):
        self.layout = Layout(layout)
        self.manifest = json.loads((ASSETS / 'manifest.json').read_text())
        self.nodes = self.manifest['nodes']
        self.definitions = self.manifest['definitions']
        self.root = ET.Element('mujoco', model=f'override-{layout.value}')
        ET.SubElement(self.root, 'compiler', angle='radian', autolimits='true', balanceinertia='true')
        ET.SubElement(self.root, 'option', timestep=str(timestep), integrator='implicitfast',
                      solver='Newton', iterations='30', tolerance='1e-8', cone='elliptic')
        visual = ET.SubElement(self.root, 'visual')
        ET.SubElement(visual, 'global', offwidth='1280', offheight='960')
        ET.SubElement(visual, 'headlight', diffuse='.7 .7 .7', ambient='.35 .35 .35')
        ET.SubElement(visual, 'quality', shadowsize='2048')
        default = ET.SubElement(self.root, 'default')
        ET.SubElement(default, 'geom', friction='.6 .005 .0001', solref='.004 1', solimp='.95 .99 .001',
                      condim='3', margin='0', density='0', contype='2', conaffinity='3')
        collision = ET.SubElement(default, 'default', **{'class': 'collision'})
        ET.SubElement(collision, 'geom', group='3', rgba='.9 .3 .1 .35')
        display = ET.SubElement(default, 'default', **{'class': 'display'})
        ET.SubElement(display, 'geom', group='1', contype='0', conaffinity='0', mass='0')
        self.asset = ET.SubElement(self.root, 'asset')
        self.world = ET.SubElement(self.root, 'worldbody')
        ET.SubElement(self.world, 'light', pos='0 0 4', dir='0 0 -1', directional='true', castshadow='true')
        ET.SubElement(self.world, 'camera', name='overhead', pos='0 0 5.7', xyaxes='1 0 0 0 1 0', fovy='43')
        ET.SubElement(self.world, 'camera', name='oblique', pos='4 -5 4.4', xyaxes='0.7809 0.6247 0 -0.3536 0.4420 0.8245', fovy='43')
        self.actuator = ET.SubElement(self.root, 'actuator')
        self.mesh_ids = {}
        self.proxy_ids = {}
        self.piece_contacts = json.loads((ASSETS / 'piece_contacts.json').read_text())
        self.placements = []
        for key, definition in self.definitions.items():
            if definition.get('mesh') and definition['semantic'] != 'foam_tile':
                mesh_id = 'cad_' + key.replace(':', '_')
                ET.SubElement(self.asset, 'mesh', name=mesh_id, file=str(ASSETS / definition['mesh']))
                self.mesh_ids[key] = mesh_id

    def leaves(self, node_id):
        node = self.nodes[node_id]
        if not node['children']:
            yield node_id, node
        for child in node['children']:
            yield from self.leaves(child)

    def world_transform(self, node):
        return CAD_TO_FIELD @ np.array(node['world_transform'])

    def display(self, body, node, reference, override=None):
        if node['definition'] not in self.mesh_ids:
            return
        pose = np.linalg.inv(reference) @ self.world_transform(node)
        ET.SubElement(body, 'geom', **{'class': 'display'}, type='mesh',
                      mesh=self.mesh_ids[node['definition']], rgba=override or color(node['name']),
                      **transform_attrs(pose))

    def convex(self, body, key, vertices, pose=None, rgba=None, display=False, masks=None):
        if key not in self.proxy_ids:
            mesh = f'proxy_{len(self.proxy_ids)}'
            ET.SubElement(self.asset, 'mesh', name=mesh, vertex=numbers(np.array(vertices).flatten()))
            self.proxy_ids[key] = mesh
        attrs = {} if pose is None else transform_attrs(pose)
        ET.SubElement(body, 'geom', **{'class': 'display' if display else 'collision'}, type='mesh',
                      mesh=self.proxy_ids[key], **attrs, **({'rgba': rgba} if rgba else {}), **({'contype': str(masks[0]), 'conaffinity': str(masks[1])} if masks else {}))

    def ring(self, body, key, y0, y1, inner0, inner1, outer0, outer1, pose, segments=10):
        for j in range(segments):
            vertices = []
            for y, inner, outer in [(y0, inner0, outer0), (y1, inner1, outer1)]:
                for radius in (inner, outer):
                    for angle in (2 * math.pi * j / segments, 2 * math.pi * (j + 1) / segments):
                        vertices.append([radius * math.cos(angle), y, radius * math.sin(angle)])
            self.convex(body, (key, j), vertices, pose, masks=(2, 2) if key == 'cup' else None)

    def fixed_field(self):
        ET.SubElement(self.world, 'geom', name='floor', type='box', pos='0 0 -.008', size='1.834 1.834 .008',
                      rgba='.32 .34 .36 1', friction='.8 .005 .0001', contype='1', conaffinity='4')
        pitch = .59805843485
        for i in range(6):
            for j in range(6):
                shade = .34 + .025 * ((i + j) % 2)
                ET.SubElement(self.world, 'geom', **{'class': 'display'}, type='box',
                              pos=numbers([(i - 2.5) * pitch, (j - 2.5) * pitch, -.0004]),
                              size=numbers([pitch / 2 - .0008, pitch / 2 - .0008, .0005]),
                              rgba=numbers([shade, shade, shade + .01, 1]))
        for axis in (0, 1):
            for sign in (-1, 1):
                pos = [0., 0., .149]
                pos[axis] = sign * 1.844
                size = [1.854, 1.854, .149]
                size[axis] = .010
                ET.SubElement(self.world, 'geom', name=f'wall_{axis}_{sign}', type='box',
                              pos=numbers(pos), size=numbers(size), rgba='.65 .68 .72 .55')
                for z in (.014, .287):
                    pos[2] = z; size[2] = .012
                    ET.SubElement(self.world, 'geom', **{'class': 'display'}, type='box',
                                  pos=numbers(pos), size=numbers(size), rgba=COLORS['silver'])
        for node_id in self.nodes['1']['children']:
            node = self.nodes[node_id]
            if node['name'].startswith('Tape-'):
                self.display(self.world, node, np.eye(4), color(node['name']) if 'Zone_' in node['name'] else '.94 .94 .94 1')
            elif '840_' in node['name']:
                self.goal(node_id)

    def goal(self, node_id):
        node = self.nodes[node_id]
        pose = self.world_transform(node)
        x, y = pose[:2, 3]
        ring = next(n for _, n in self.leaves(node_id) if n['name'].startswith('276-9250-003'))
        ring_pose = self.world_transform(ring)
        for _, leaf in self.leaves(node_id):
            self.display(self.world, leaf, np.eye(4), color(node['name']))
        for index, vertices in enumerate(self.piece_contacts['goal']['pieces']):
            self.convex(self.world, ('goal_cad', index), vertices, ring_pose)
        stand = ring_pose[2, 3]
        if stand > .001:
            ET.SubElement(self.world, 'geom', **{'class': 'collision'}, type='cylinder',
                          pos=numbers([x, y, stand / 2]), size=numbers([.069, stand / 2]))
        self.placements.append({'kind': 'goal', 'source': node_id, 'name': node['name'], 'position': [x, y, stand]})

    def game_piece(self, node_id, kind):
        node = self.nodes[node_id]
        pose = self.world_transform(node)
        body = ET.SubElement(self.world, 'body', name=f'{kind}_{node_id.replace("/", "_")}', **transform_attrs(pose))
        ET.SubElement(body, 'freejoint')
        mass = .06 if kind == 'pin' else .10
        ET.SubElement(body, 'inertial', pos='0 0 0', mass=str(mass), diaginertia='0.00025 0.00005 0.00025')
        for _, leaf in self.leaves(node_id):
            self.display(body, leaf, pose)
            local = np.linalg.inv(pose) @ self.world_transform(leaf)
            if kind == 'cup':
                for index, vertices in enumerate(self.piece_contacts['cup']['pieces']):
                    self.convex(body, ('cup_cad', index), vertices, local, masks=(2, 2))
                floor_vertices = [[r * math.cos(t), h, r * math.sin(t)]
                                  for h, r in [(.00275, .0401), (.085, .0195)]
                                  for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
                self.convex(body, 'cup_floor', floor_vertices, local, masks=(4, 1))
            else:
                # Partitioned CAD hulls retain shoulder relief and hollow tips.
                for index, vertices in enumerate(self.piece_contacts['pin']['pieces']):
                    self.convex(body, ('pin_cad', index), vertices, local)
        self.placements.append({'kind': kind, 'source': node_id, 'name': node['name'],
                                'position': pose[:3, 3].tolist(), 'rotation': pose[:3, :3].tolist()})

    def toggle(self, node_id, index):
        leaves = list(self.leaves(node_id))
        bar = next(n for _, n in leaves if '023_GREY' in n['name'])
        pose = self.world_transform(bar)
        body = ET.SubElement(self.world, 'body', name=f'toggle_{index}', **transform_attrs(pose))
        ET.SubElement(body, 'inertial', pos='0 0 0', mass='.32', diaginertia='.00016 .012 .012')
        ET.SubElement(body, 'joint', name=f'toggle_{index}', type='hinge', axis='1 0 0', limited='false', damping='.005', frictionloss='.005')
        vertices = [[x, y, z] for x in (-.3281, .3281) for y, z in [(-.03016, 0), (.02143, -.02782), (.02143, .02782)]]
        self.convex(body, 'toggle', vertices)
        for _, leaf in leaves:
            if '023_GREY' in leaf['name']:
                continue
            if '024' in leaf['name']:
                self.display(body, leaf, pose)
            else:
                self.display(self.world, leaf, np.eye(4))
        cross = np.array([[-.0305, 0], [.0218, -.0282], [.0218, .0282]])
        for j, c in enumerate(['blue', 'yellow', 'red']):
            a, b = cross[j], cross[(j + 1) % 3]
            vertices = [[x, yz[0], yz[1]] for x in (-.327, .327) for yz in (a, b, (a + b) / 2 * 1.02)]
            self.convex(body, ('toggle_face', j), vertices, rgba=COLORS[c], display=True)
        self.placements.append({'kind': 'toggle', 'source': node_id, 'position': pose[:3, 3].tolist(), 'axis': pose[:3, 0].tolist()})

    def loader(self, node_id, index):
        leaves = list(self.leaves(node_id))
        chute = next(n for _, n in leaves if '031_' in n['name'])
        guide = next(n for _, n in leaves if n['name'] == '276-9250-032')
        guide_pose = self.world_transform(guide)
        for rail, x in enumerate((-.056, .056)):
            rail_pose = guide_pose.copy()
            rail_pose[:3, 3] += guide_pose[:3, :3] @ np.array([x, .1575, .032])
            ET.SubElement(self.world, 'geom', name=f'loader_guide_{index}_{rail}',
                          **{'class': 'collision'}, type='box', size='.003 .1575 .017',
                          **transform_attrs(rail_pose))
        pose = self.world_transform(chute)
        definition = self.definitions[chute['definition']]
        bounds = definition.get('mesh_bounds', definition['local_bounds'])
        pose[2, 3] = -bounds[1]
        body = ET.SubElement(self.world, 'body', name=f'loader_{index}', **transform_attrs(pose))
        ET.SubElement(body, 'inertial', pos='0 .17 .02', mass='.45', diaginertia='.006 .001 .006')
        ET.SubElement(body, 'joint', name=f'loader_{index}', type='slide', axis='0 1 0', range='0 .2185', damping='3', frictionloss='.5')
        ET.SubElement(self.actuator, 'position', name=f'loader_{index}_lift', joint=f'loader_{index}',
                      kp='1800', kv='55', ctrlrange='0 .2185', forcerange='-50 50')
        for _, leaf in leaves:
            if '031_' in leaf['name']:
                ET.SubElement(body, 'geom', **{'class': 'display'}, type='mesh',
                              mesh=self.mesh_ids[leaf['definition']], rgba=color(leaf['name'])[:-1] + '.55')
            elif '034' in leaf['name']:
                original = self.world_transform(chute)
                self.display(body, leaf, original)
            else:
                self.display(self.world, leaf, np.eye(4))
        for x in (-.049, .049):
            ET.SubElement(body, 'geom', **{'class': 'collision'}, type='box',
                          pos=numbers([x, .177, .006]), size='.002 .18 .045')
        ET.SubElement(body, 'geom', **{'class': 'collision'}, type='box', pos='0 .177 .049', size='.049 .18 .002')
        ET.SubElement(body, 'geom', **{'class': 'collision'}, type='box', pos='0 .215 -.039', size='.049 .139 .002')
        self.placements.append({'kind': 'loader', 'source': node_id, 'position': pose[:3, 3].tolist(), 'travel_m': .2185})

    def build(self, robot_start=None):
        self.fixed_field()
        short_goals = [np.array(p['position'][:2]) for p in self.placements if p['kind'] == 'goal' and 'Neutral' in p['name'] and 'Goal 0' not in p['name']]
        ti = li = 0
        for node_id in self.nodes['1']['children']:
            node = self.nodes[node_id]
            name = node['name']; pose = self.world_transform(node)
            if name.startswith('276-9250-120'):
                self.toggle(node_id, ti); ti += 1
            elif name.startswith('276-9250-830'):
                self.loader(node_id, li); li += 1
            elif name.startswith(('276-9250-81x', '276-9250-80x')) and max(abs(pose[:2, 3])) < 1.8:
                kind = 'cup' if '81x' in name else 'pin'
                if kind == 'pin' and any(np.linalg.norm(pose[:2, 3] - xy) < .01 for xy in short_goals):
                    continue
                if kind == 'pin' and self.layout == Layout.SKILLS and np.linalg.norm(pose[:2, 3]) < .01:
                    continue
                self.game_piece(node_id, kind)
        if robot_start is not None:
            from .route import add_robot
            add_robot(self.world, self.actuator, robot_start)
        from .loader_contacts import replace_loader_contacts
        loader_contacts = replace_loader_contacts(self.root)
        metadata = {'layout': self.layout.value, 'frame': 'field_center_m_rad', 'cad_to_field': CAD_TO_FIELD.tolist(),
                    'source': self.manifest['source'], 'placements': self.placements,
                    'loader_contacts': loader_contacts,
                    'assumptions': ['Unmeasured masses, inertias, friction and damping.',
                                    'Game-piece CAD hull partitions preserve openings and shoulder relief while approximating curvature.',
                                    'Loader servo models human input; lowered sleeve bottom normalized to floor.',
                                    'CAD-derived loader walls retain cutouts; the slide joint owns sleeve/guide bearing contact.',
                                    'Toggle friction approximates seating; no calibrated detent.',
                                    'No official scoring, GPS texture, robot CAD or PROS controller parity.']}
        return SceneSpec(ET.tostring(self.root, encoding='unicode'), metadata)


def build_scene(layout=Layout.MATCH, *, timestep=.002, robot_start=None):
    if not math.isfinite(timestep) or not 0 < timestep <= .01:
        raise ValueError('timestep must be finite and between 0 and 0.01 seconds')
    return FieldBuilder(Layout(layout), timestep).build(robot_start)
