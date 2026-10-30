"""Torque-driven CAD robot demonstrations with explicit simulation evidence."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .field import Layout
from .robot import build_robot_scene
from .robot_control import RobotControl
from .scene import check_state


def _model(fixed, pieces=False, visual_mode='simple'):
    xml, metadata = build_robot_scene(layout=Layout.MATCH if pieces else None, fixed=fixed, visual_mode=visual_mode)
    root = ET.fromstring(xml)
    if pieces:
        world = root.find('worldbody')
        for node in list(world):
            if node.tag != 'light' and node.get('name') not in ('robot', 'floor', 'pin_1_15', 'cup_1_16'):
                world.remove(node)
        actuator = root.find('actuator')
        for node in list(actuator):
            if not node.get('name', '').startswith('robot_'):
                actuator.remove(node)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, metadata


def _pose(model, data, body_name, position):
    body = model.body(body_name)
    address = model.jnt_qposadr[int(body.jntadr[0])]
    data.qpos[address:address + 3] = position


def _sample(model, data):
    errors = {'lift_connect_m': 0., 'lift_gear_rad': 0., 'claw_transmission_rad': 0.}
    for row in np.flatnonzero(data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY):
        equality = int(data.efc_id[row])
        name = model.equality(equality).name or ''
        connect = model.eq_type[equality] == mujoco.mjtEq.mjEQ_CONNECT
        if name.startswith('robot_lift_'):
            key = 'lift_connect_m' if connect else 'lift_gear_rad'
        elif name.startswith(('robot_claw_', 'robot_intake_')):
            key = 'claw_transmission_rad'
        else:
            continue
        errors[key] = max(errors[key], abs(float(data.efc_pos[row])))
    names = ('robot_lift_crank', 'robot_claw_left_jaw', 'robot_claw_right_jaw',
             'robot_claw_left_shaft', 'robot_claw_right_shaft',
             'robot_claw_left_roller', 'robot_claw_right_roller')
    return {'time_s': float(data.time), 'robot_position_m': data.body('robot').xpos.tolist(),
            'lift_end_height_m': float(data.body('robot_lift_end').xpos[2]),
            'claw_relative_rotation': (data.body('robot_lift_end').xmat.reshape(3, 3).T @
                                       data.body('robot_claw_mount').xmat.reshape(3, 3)).tolist(),
            'joints_rad': {name: float(data.joint(name).qpos[0]) for name in names},
            'equality_errors': errors}


def _commands(name, t):
    if name == 'lift_cycle':
        target = 1.9 * np.clip((t - .5) / 4., 0, 1) if t < 6 else 1.9 * np.clip((11 - t) / 5., 0, 1)
        return dict(lift=float(target))
    if name == 'claw_cycle':
        return dict(intake=1. if t < 3 else -1., lift=.9 * np.clip(t / 3., 0, 1))
    if name == 'drivetrain':
        return dict(forward=.22 if .5 < t < 2.5 else 0., turn=.6 if 3 < t < 5 else 0.)
    return dict(forward=.06 if .5 < t < 2 else 0., intake=-1. if t < 5 else 1.,
                lift=.9 * np.clip((t - 2) / 3., 0, 1))


def _run(name, seconds, output, video, lift_rpm, visual_mode='simple'):
    pieces = name == 'pickup_attempt'
    model, data, metadata = _model(name in ('lift_cycle', 'claw_cycle'), pieces, visual_mode)
    if pieces:
        _pose(model, data, 'pin_1_15', [.27, 0., .083])
        _pose(model, data, 'cup_1_16', [.48, 0., .083])
        mujoco.mj_forward(model, data)
    control = RobotControl(model, metadata, lift_rpm=lift_rpm)
    xml_path = output / f'{name}.xml'
    mujoco.mj_saveLastXML(str(xml_path), model)
    root = ET.fromstring(xml_path.read_text())
    keyframes = ET.SubElement(root, 'keyframe')
    ET.SubElement(keyframes, 'key', name='experiment_initial',
                  qpos=' '.join(map(str, data.qpos)), qvel=' '.join(map(str, data.qvel)))
    xml_path.write_text(ET.tostring(root, encoding='unicode'))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.03, 0., .55]
    camera.distance = 2.3
    camera.azimuth, camera.elevation = 135, -15
    if name == 'claw_cycle':
        camera.lookat[:] = [.17, 0., .15]
        camera.distance, camera.azimuth, camera.elevation = .68, 125, -20
    if pieces:
        camera.lookat[:] = [.18, 0., .32]
        camera.distance = 1.5
    options = mujoco.MjvOption()
    options.geomgroup[3] = False
    renderer = mujoco.Renderer(model, 720, 960) if video else None
    writer = imageio.get_writer(output / f'{name}.mp4', fps=30, codec='libx264', macro_block_size=1) if video else None
    records, contacts = [], set()
    maxima = dict.fromkeys(_sample(model, data)['equality_errors'], 0.)
    initial = _sample(model, data)
    pin_height_max = 0.
    try:
        for frame in range(round(seconds * 30)):
            while data.time < (frame + 1) / 30 - model.opt.timestep / 2:
                control.update(data, **_commands(name, data.time))
                mujoco.mj_step(model, data)
                check_state(model, data)
                sample = _sample(model, data)
                for key, value in sample['equality_errors'].items():
                    maxima[key] = max(maxima[key], value)
                if pieces:
                    pin_height_max = max(pin_height_max, float(data.body('pin_1_15').xpos[2]))
                    for contact in data.contact:
                        bodies = [model.body(int(model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
                        if any(n.startswith('robot') for n in bodies) and 'pin_1_15' in bodies:
                            contacts.add(tuple(sorted(bodies)))
            records.append(sample)
            if renderer is not None:
                if name == 'claw_cycle':
                    camera.lookat[:] = data.body('robot_claw_mount').xpos + [0, 0, .10]
                renderer.update_scene(data, camera=camera, scene_option=options)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 960, 47), fill=(15, 18, 25))
                draw.text((16, 8), f'{name.replace("_", " ")} | {data.time:.1f} s | lift {sample["lift_end_height_m"]:.3f} m', fill='white')
                draw.text((16, 28), 'CAD simulation. Motor response and elastic band strength are uncalibrated.', fill='white')
                writer.append_data(np.asarray(image))
                if frame in (0, round(seconds * 15), round(seconds * 30) - 1):
                    image.save(output / f'{name}-{frame:03}.png')
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    report = {'name': name, 'duration_s': float(data.time), 'initial': initial, 'final': records[-1],
              'claw_closure': metadata['claw']['closure_springs'], 'limitations': metadata['limitations'],
              'max_equality_errors': maxima, 'warnings': [int(w.number) for w in data.warning],
              'lift_height_range_m': [min(r['lift_end_height_m'] for r in records), max(r['lift_end_height_m'] for r in records)],
              'joint_ranges_rad': {joint: [min(r['joints_rad'][joint] for r in records), max(r['joints_rad'][joint] for r in records)] for joint in initial['joints_rad']},
              'video': str(output / f'{name}.mp4') if video else None,
              'robot_displacement_m': float(np.linalg.norm(data.body('robot').xpos[:2] - np.array(initial['robot_position_m'][:2])))}
    if name == 'lift_cycle':
        report['checks'] = {'raises_over_085_m': report['lift_height_range_m'][1] - initial['lift_end_height_m'] > .85,
                            'returns_within_002_m': abs(records[-1]['lift_end_height_m'] - initial['lift_end_height_m']) < .02,
                            'lift_connect_within_02_mm': maxima['lift_connect_m'] < .0002,
                            'lift_gear_within_003_rad': maxima['lift_gear_rad'] < .003}
    elif name == 'claw_cycle':
        fixed_orientation = all(np.allclose(r['claw_relative_rotation'], initial['claw_relative_rotation'],
                                           atol=1e-12, rtol=0.) for r in records)
        report['checks'] = {'claw_orientation_fixed_to_lift': fixed_orientation,
                            'lift_moves_over_01_m': np.ptp(report['lift_height_range_m']) > .1,
                            'left_roller_moves_over_10_rad': np.ptp(report['joint_ranges_rad']['robot_claw_left_roller']) > 10.,
                            'right_roller_moves_over_10_rad': np.ptp(report['joint_ranges_rad']['robot_claw_right_roller']) > 10.,
                            'transmission_within_004_rad': maxima['claw_transmission_rad'] < .004}
    elif name == 'drivetrain':
        report['checks'] = {'moves_over_03_m': report['robot_displacement_m'] > .3}
    if not pieces:
        report['checks'] = {key: bool(value) for key, value in report['checks'].items()}
        if not all(report['checks'].values()):
            raise AssertionError(f'{name} failed behavioral checks: {report["checks"]}')
    if pieces:
        report.update({'robot_pin_contacts': sorted(contacts), 'pin_max_center_height_m': pin_height_max,
                       'pin_final_position_m': data.body('pin_1_15').xpos.tolist(),
                       'cup_final_position_m': data.body('cup_1_16').xpos.tolist(),
                       'pickup_verified': False, 'placement_attempted': False, 'placement_verified': False,
                       'limitation': 'Pickup failed to retain the pin. The scripted lift/release continued without a grasp, so placement was not attempted. No grasp weld or pose changes after initialization.'})
    (output / f'{name}.json').write_text(json.dumps({'summary': report, 'samples': records}, indent=2))
    return report


def verify_robot(output, video=True, lift_rpm=100, visual_mode='simple'):
    """Record mechanism cycles and a free contact pickup attempt; never imply hardware validation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = [_run(name, seconds, output, video, lift_rpm, visual_mode) for name, seconds in
               (('lift_cycle', 12.), ('claw_cycle', 6.), ('drivetrain', 6.), ('pickup_attempt', 8.))]
    result = {'lift_cartridge_rpm': lift_rpm, 'experiments': reports,
              'limitations': ['Actuator and contact parameters are uncalibrated.',
                              'Jaw hinges are passive. The claw mount is fixed to the lift end with no tilt actuator.',
                              'The pickup trial is an attempt and does not prove scoring or hardware feasibility.']}
    (output / 'robot-verification.json').write_text(json.dumps(result, indent=2))
    return result
