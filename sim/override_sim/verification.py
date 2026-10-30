"""Reproducible contact experiments using the field's actual CAD bodies.

These checks validate simulation behavior. They do not calibrate physical masses,
friction, compliance, actuator forces, or match performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .field import build_scene, numbers
from .scene import check_state, frame

LABEL = 'VALIDATED SIMULATION BEHAVIOR; NOT PHYSICALLY CALIBRATED'


@dataclass(frozen=True)
class Experiment:
    name: str
    bodies: tuple[str, ...]
    seconds: float
    description: str


EXPERIMENTS = (
    Experiment('pin_into_cup', ('pin_1_15', 'cup_1_16'), 3.,
               'Free pin falls point down into the clear upward cup.'),
    Experiment('pin_rim_contact', ('pin_1_15', 'cup_1_16'), 3.,
               'An off-axis falling pin hits the clear cup rim and deflects.'),
    Experiment('cup_over_pin', ('pin_1_15', 'cup_1_16'), 3.,
               'Inverted free cup descends over an upright pin.'),
    Experiment('rolling', ('pin_1_15', 'cup_1_16'), 3.,
               'Sideways pin and cup roll under initial linear and angular velocity.'),
    Experiment('tumbling', ('pin_1_15', 'cup_1_16'), 3.,
               'Tilted pin and cup tumble after an initial angular impulse.'),
    Experiment('toggle_rotation', ('toggle_0',), 3.,
               'Original unlimited toggle hinge receives a torque pulse then coasts.'),
    Experiment('loader_cycle', ('loader_0',), 6.,
               'Human lift servo raises and lowers the sleeve while its guides remain fixed.'),
    Experiment('loader_aperture', ('loader_0', 'pin_1_15'), 2.5,
               'A free pin drops through the bottom aperture of the raised sleeve.'),
)


def _quat_product(a, b):
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def _isolated(experiment):
    """Prune a field scene; never substitute collision primitives for game pieces."""
    spec = build_scene(timestep=.001 if experiment.name == 'loader_aperture' else .002)
    root = ET.fromstring(spec.xml)
    world = root.find('worldbody')
    selected = [b for b in world.findall('body') if b.get('name') in experiment.bodies]
    if len(selected) != len(experiment.bodies):
        raise ValueError(f'Missing original field bodies for {experiment.name}')
    reference = next((body for body in selected if body.get('name') == 'loader_0'), selected[0])
    origin = np.fromstring(reference.get('pos'), sep=' ')
    for node in list(world):
        keep = (node in selected or node.tag == 'light' or node.get('name') == 'floor'
                or (experiment.name.startswith('loader') and node.get('name', '').startswith('loader_guide_0_')))
        if experiment.name.startswith(('loader', 'toggle')) and node.tag == 'geom' and node.get('class') == 'display':
            pos = np.fromstring(node.get('pos', '0 0 0'), sep=' ')
            keep = np.linalg.norm(pos - origin) < .6
        if not keep:
            world.remove(node)
    actuator = root.find('actuator')
    for node in list(actuator):
        if node.get('joint') not in experiment.bodies:
            actuator.remove(node)
    if not experiment.name.startswith(('loader', 'toggle')):
        origin = np.zeros(3)
    height = .25 if experiment.name.startswith('loader') else (0. if experiment.name.startswith('toggle') else .14)
    target = origin + np.array([0, 0, height])
    camera_pos = target + np.array([.48, -.73, .28])
    z = camera_pos - target
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(world, 'camera', name='experiment', pos=numbers(camera_pos),
                  xyaxes=numbers(np.r_[x, y]), fovy='48')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return spec, root, model, data


def _free_pose(model, data, name, xyz, quat=None, velocity=None):
    body = model.body(name)
    joint = int(body.jntadr[0])
    qadr = int(model.jnt_qposadr[joint])
    dadr = int(model.jnt_dofadr[joint])
    data.qpos[qadr:qadr + 3] = xyz
    if quat is not None:
        data.qpos[qadr + 3:qadr + 7] = quat
    if velocity is not None:
        data.qvel[dadr:dadr + 6] = velocity


def _initialize(experiment, model, data):
    name = experiment.name
    if name in ('pin_into_cup', 'pin_rim_contact'):
        _free_pose(model, data, 'cup_1_16', [0, 0, .0823])
        _free_pose(model, data, 'pin_1_15', [.045 if name == 'pin_rim_contact' else 0, 0, .335])
    elif name == 'cup_over_pin':
        _free_pose(model, data, 'pin_1_15', [0, 0, .0825])
        body = model.body('cup_1_16')
        qadr = int(model.jnt_qposadr[int(body.jntadr[0])])
        flipped = _quat_product([0., 1., 0., 0.], data.qpos[qadr + 3:qadr + 7])
        _free_pose(model, data, 'cup_1_16', [0, 0, .335], flipped)
    elif name in ('rolling', 'tumbling'):
        for i, body in enumerate(experiment.bodies):
            quat = [1., 0., 0., 0.] if name == 'rolling' else [math.cos(.3), math.sin(.3), 0., 0.]
            velocity = [.35, 0, 0, 0, 8.5, 0] if name == 'rolling' else [.4, 0, 0, 12., 3., 0]
            _free_pose(model, data, body, [-.20 + i * .4, 0, .045 if name == 'rolling' else .13], quat, velocity)
    elif name == 'loader_aperture':
        data.joint('loader_0').qpos[0] = .2185
        data.ctrl[model.actuator('loader_0_lift').id] = .2185
        mujoco.mj_forward(model, data)
        sleeve = data.body('loader_0')
        rotation = sleeve.xmat.reshape(3, 3)
        center = sleeve.xpos + rotation @ np.array([0., .045, .006])
        _free_pose(model, data, 'pin_1_15', center + [0, 0, .10])
    mujoco.mj_forward(model, data)


def _control(experiment, model, data):
    if experiment.name == 'toggle_rotation':
        data.qfrc_applied[model.joint('toggle_0').dofadr[0]] = .08 if data.time < .55 else 0.
    elif experiment.name == 'loader_cycle':
        target = .2185 if .5 <= data.time < 3.2 else 0.
        data.ctrl[model.actuator('loader_0_lift').id] = target
    elif experiment.name == 'loader_aperture':
        data.ctrl[model.actuator('loader_0_lift').id] = .2185


def _record(experiment, model, data):
    bodies = {}
    for name in experiment.bodies:
        body = data.body(name)
        bodies[name] = {'position_m': body.xpos.tolist(), 'quaternion_wxyz': body.xquat.tolist()}
    return {'time_s': float(data.time), 'bodies': bodies,
            'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist(),
            'actuator_force_n': data.actuator_force.tolist(),
            'joints': {name: float(data.joint(name).qpos[0]) for name in ('loader_0', 'toggle_0')
                       if name in experiment.bodies}}


def _evaluate(experiment, records, contacts, penetration, warnings, fixed_drift, pair_penetrations):
    name = experiment.name
    first, last = records[0], records[-1]
    metrics = {'max_penetration_m': penetration, 'warning_count': warnings,
               'contact_body_pairs': sorted([list(pair) for pair in contacts]),
               'fixed_guide_drift_m': fixed_drift,
               'max_penetration_by_body_pair_m': {' / '.join(pair): value for pair, value in pair_penetrations.items()}}
    checks = {'finite_states': True, 'zero_warnings': warnings == 0,
              ('penetration_below_5mm' if name == 'loader_aperture' else 'penetration_below_10mm'):
                  penetration < (.005 if name == 'loader_aperture' else .01),
              'fixed_guides_stationary': fixed_drift < 1e-10}
    positions = lambda body: np.array([r['bodies'][body]['position_m'] for r in records])
    if name in ('pin_into_cup', 'cup_over_pin'):
        pin, cup = positions('pin_1_15'), positions('cup_1_16')
        moving = pin if name == 'pin_into_cup' else cup
        separation = float(np.linalg.norm(pin[-1, :2] - cup[-1, :2]))
        height = float(abs(pin[-1, 2] - cup[-1, 2]))
        speed = float(np.linalg.norm(last['qvel']))
        metrics.update(drop_m=float(moving[0, 2] - moving[-1, 2]),
                       final_axis_separation_m=separation, final_center_height_difference_m=height,
                       final_speed=speed)
        checks.update(fell_under_gravity=metrics['drop_m'] > .08,
                      pieces_contacted=('cup_1_16', 'pin_1_15') in contacts,
                      nested_axes=separation < .015,
                      nested_height=0.025 < height < .12,
                      settled=speed < .15)
    elif name == 'pin_rim_contact':
        pin, cup = positions('pin_1_15'), positions('cup_1_16')
        deflection = float(np.max(np.linalg.norm(pin[:, :2] - pin[0, :2], axis=1)))
        metrics.update(initial_axis_offset_m=.045, pin_lateral_deflection_m=deflection)
        checks.update(hit_cup_rim=('cup_1_16', 'pin_1_15') in contacts,
                      deflected_after_contact=deflection > .01)
    elif name in ('rolling', 'tumbling'):
        for body in experiment.bodies:
            path = positions(body)
            q0 = np.array(first['bodies'][body]['quaternion_wxyz'])
            angles = [2 * math.acos(min(1., abs(float(np.dot(q0, r['bodies'][body]['quaternion_wxyz'])))))
                      for r in records]
            distance = float(np.max(np.linalg.norm(path[:, :2] - path[0, :2], axis=1)))
            metrics[body] = {'max_translation_m': distance, 'max_rotation_rad': max(angles)}
            checks[body + '_translated'] = distance > .04
            checks[body + '_rotated'] = max(angles) > .5
            checks[body + '_floor_contact'] = tuple(sorted((body, 'world'))) in contacts
    elif name == 'toggle_rotation':
        angles = np.array([r['qpos'][0] for r in records])
        pulse_end = min(records, key=lambda r: abs(r['time_s'] - .55))
        coast_angle = float(angles[-1] - pulse_end['qpos'][0])
        metrics.update(total_rotation_rad=float(angles[-1] - angles[0]), coast_rotation_rad=coast_angle,
                       pulse_torque_nm=.08, pulse_seconds=.55)
        checks.update(rotated_beyond_full_turn=metrics['total_rotation_rad'] > 2 * math.pi,
                      coasted_after_pulse=coast_angle > .2,
                      continuous_monotonic_rotation=bool(np.min(np.diff(angles)) >= -1e-5))
    elif name == 'loader_cycle':
        heights = np.array([r['qpos'][0] for r in records])
        high = min(records, key=lambda r: abs(r['time_s'] - 3.1))['qpos'][0]
        low = last['qpos'][0]
        metrics.update(upper_endpoint_m=high, lower_endpoint_m=low, prescribed_travel_m=.2185,
                       min_slide_m=float(min(heights)), max_slide_m=float(max(heights)),
                       peak_human_lift_force_n=max(abs(r['actuator_force_n'][0]) for r in records))
        checks.update(upper_endpoint_within_5mm=abs(high - .2185) < .005,
                      lower_endpoint_within_5mm=abs(low) < .005,
                      slide_respects_bounds=min(heights) > -.005 and max(heights) < .2235)
    elif name == 'loader_aperture':
        pin = positions('pin_1_15')
        metrics.update(pin_drop_m=float(pin[0, 2] - pin[-1, 2]), final_pin_height_m=float(pin[-1, 2]),
                       final_loader_height_m=float(last['joints']['loader_0']))
        checks.update(aperture_wall_penetration_below_5mm=pair_penetrations.get(('loader_0', 'pin_1_15'), 0.) < .005,
                      passed_open_bottom=pin[-1, 2] < .15,
                      pin_fell=metrics['pin_drop_m'] > .15,
                      loader_held_raised=last['joints']['loader_0'] > .20)
    return metrics, {key: bool(value) for key, value in checks.items()}


def _video_frame(renderer, data, experiment):
    picture = Image.fromarray(frame(renderer, data, 'experiment'))
    draw = ImageDraw.Draw(picture)
    font = ImageFont.load_default(size=max(12, picture.width // 58))
    draw.rectangle((0, 0, picture.width, 35), fill=(16, 20, 26))
    draw.text((12, 8), f'{experiment.name.replace("_", " ").upper()}   t = {data.time:.2f} s',
              fill='white', font=font)
    draw.rectangle((0, picture.height - 31, picture.width, picture.height), fill=(16, 20, 26))
    draw.text((12, picture.height - 24), LABEL, fill=(255, 205, 92), font=font)
    return np.asarray(picture)


def run_scenario(name, output=None, *, width=960, height=720, fps=30):
    """Run one experiment; an output directory enables MP4 and still rendering."""
    if not 1 <= fps <= 120 or width < 64 or height < 64:
        raise ValueError('fps must be 1..120 and frame dimensions at least 64 pixels')
    experiment = next((e for e in EXPERIMENTS if e.name == name), None)
    if experiment is None:
        raise ValueError(f'Unknown experiment {name!r}')
    spec, xml, model, data = _isolated(experiment)
    _initialize(experiment, model, data)
    output = None if output is None else Path(output)
    renderer = writer = None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        initial = ET.SubElement(xml, 'keyframe')
        ET.SubElement(initial, 'key', name='experiment_initial', qpos=numbers(data.qpos),
                      qvel=numbers(data.qvel), ctrl=numbers(data.ctrl))
        (output / f'{name}.xml').write_text(ET.tostring(xml, encoding='unicode'))
        renderer = mujoco.Renderer(model, height, width)
        writer = imageio.get_writer(output / f'{name}.mp4', fps=fps, codec='libx264', quality=8, macro_block_size=1)
    records = [_record(experiment, model, data)]
    contacts = set()
    pair_penetrations = {}
    penetration = 0.
    fixed = np.flatnonzero(model.geom_bodyid == 0)
    fixed_initial = data.geom_xpos[fixed].copy()
    drift = 0.
    error = None
    next_sample = 1 / fps
    try:
        if writer:
            writer.append_data(_video_frame(renderer, data, experiment))
        while data.time < experiment.seconds - model.opt.timestep / 2:
            _control(experiment, model, data)
            mujoco.mj_step(model, data)
            check_state(model, data)
            if not np.isfinite(data.qacc).all():
                raise RuntimeError('Nonfinite acceleration')
            drift = max(drift, float(np.max(np.linalg.norm(data.geom_xpos[fixed] - fixed_initial, axis=1))))
            for c in data.contact:
                penetration = max(penetration, -float(c.dist))
                pair = tuple(sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or 'world'
                                    for g in (c.geom1, c.geom2)))
                contacts.add(pair)
                pair_penetrations[pair] = max(pair_penetrations.get(pair, 0.), -float(c.dist))
            if data.time >= next_sample - model.opt.timestep / 2:
                records.append(_record(experiment, model, data))
                if writer:
                    writer.append_data(_video_frame(renderer, data, experiment))
                next_sample += 1 / fps
        if renderer:
            Image.fromarray(_video_frame(renderer, data, experiment)).save(output / f'{name}.png')
    except RuntimeError as exc:
        error = str(exc)
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    metrics, checks = _evaluate(experiment, records, contacts, penetration,
                                sum(int(w.number) for w in data.warning), drift, pair_penetrations)
    if error:
        checks['finite_states'] = False
    result = {'name': name, 'description': experiment.description, 'label': LABEL,
              'passed': all(checks.values()), 'checks': checks, 'metrics': metrics,
              'source_bodies': list(experiment.bodies), 'source': spec.metadata['source'],
              'trajectory': records, 'error': error,
              'timestep_s': float(model.opt.timestep),
              'contact_timeconstant_s': float(model.geom_solref[model.geom('floor').id, 0]),
              'replay': 'XML keyframe 0 stores initial qpos/qvel/ctrl. Reset it with mj_resetDataKeyframe, then apply this experiment control schedule from run_scenario.',
              'artifacts': {ext: str(output / f'{name}.{ext}') for ext in ('mp4', 'png', 'xml')} if output else {}}
    if output:
        (output / f'{name}.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def verify(output: Path, width=960, height=720, fps=30):
    """Write separate real dynamics videos, stills, and measured outcomes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = [run_scenario(e.name, output, width=width, height=height, fps=fps) for e in EXPERIMENTS]
    report = {'label': LABEL, 'passed': all(r['passed'] for r in scenarios),
              'scenarios': [{k: v for k, v in r.items() if k != 'trajectory'} for r in scenarios],
              'method': 'Original build_scene bodies, meshes and colliders. Initial poses/velocities only; free evolution and mechanism actuation thereafter.',
              'limits': 'Masses, inertia, friction, damping and human lift forces are unmeasured. These results do not establish physical realism or official scoring.',
              'output': str(output)}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
