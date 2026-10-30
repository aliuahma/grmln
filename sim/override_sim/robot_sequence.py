"""Continuous wheel-torque-only verification on the unchanged university field."""
from pathlib import Path
import json
import math
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .field import Layout
from .robot import build_robot_scene
from .robot_control import RobotControl
from .scene import check_state, frame


def run_sequence(output, video=True, *, seconds=15., visual_mode='simple'):
    """Rest, accelerate through loose pins, turn under momentum, then approach a toggle.

    The selected initial pose is a clear lane west of the southeast loose-pin group.
    No state writes or applied body forces occur after initialization.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    xml, metadata = build_robot_scene(Layout.MATCH, start=(-.15, -.90, 0.), visual_mode=visual_mode)
    (output / 'sequence.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for contact in data.contact:
        if contact.dist < -.003 and any(model.body(int(model.geom_bodyid[g])).name.startswith('robot') for g in (contact.geom1, contact.geom2)):
            raise ValueError('Initial robot pose overlaps field geometry')
    control = RobotControl(model, metadata)
    pins = [i for i in range(model.nbody) if model.body(i).name.startswith('pin_')]
    initial = data.xpos[pins].copy()
    toggles = np.array([data.joint(f'toggle_{i}').qpos[0] for i in range(4)])
    contacts = set()
    drift_contacts = set()
    drift_final = initial.copy()
    first_toggle_contacts = {}
    toggle_contact_impulses = {}
    arm_travel = {side: [0., 0.] for side in ('left', 'right')}
    lift_rest_range = [float('inf'), -float('inf')]
    phases = []
    records = []
    renderer = mujoco.Renderer(model, 720, 960) if video else None
    writer = imageio.get_writer(output / 'sequence.mp4', fps=30, codec='libx264', macro_block_size=1) if video else None
    camera = mujoco.MjvCamera()
    camera.distance = 2.8
    camera.azimuth = 125
    camera.elevation = -38
    next_frame = next_control = 0.
    saved_drift_slip = 0.
    phase = 'rest'
    waypoint = 0
    points = [(.60, -.90), (.98, -.08), (1.40, 0.), (1.72, 0.)]
    max_slip = 0.
    max_drift_slip = 0.
    max_toggle = np.zeros(4)
    wall = time.perf_counter()
    forward = turn = 0.
    try:
        while data.time < seconds:
            if data.time >= next_control:
                pose = data.body('robot').xpos
                rotation = data.body('robot').xmat.reshape(3, 3)
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                velocity = data.qvel[model.joint('robot_free').dofadr[0]:][:2]
                slip = abs(-math.sin(yaw) * velocity[0] + math.cos(yaw) * velocity[1])
                max_slip = max(max_slip, float(slip))
                for side, travel in arm_travel.items():
                    angle = float(data.joint(f'robot_toggle_{side}_arm').qpos[0])
                    travel[0], travel[1] = min(travel[0], angle), max(travel[1], angle)
                if data.time < 1.:
                    lift_angle = float(data.joint('robot_lift_crank').qpos[0])
                    lift_rest_range[0] = min(lift_rest_range[0], lift_angle)
                    lift_rest_range[1] = max(lift_rest_range[1], lift_angle)
                    initial = data.xpos[pins].copy()
                if 1. <= data.time < 3.8:
                    max_drift_slip = max(max_drift_slip, float(slip))
                    drift_final = data.xpos[pins].copy()
                forward = turn = 0.
                if data.time >= 1.:
                    phase = 'drive' if waypoint == 0 else 'drift' if waypoint == 1 else 'toggle approach'
                    target = np.array(points[waypoint])
                    delta = target - pose[:2]
                    distance = np.linalg.norm(delta)
                    desired = math.atan2(delta[1], delta[0]) if waypoint < len(points)-1 else 0.
                    error = math.atan2(math.sin(desired-yaw), math.cos(desired-yaw))
                    turn = float(np.clip(5 * error, -5., 5.))
                    forward = min(1.25 if waypoint < 2 else .45, 2 * distance) * max(.05, math.cos(error))
                    if distance < (.15 if waypoint < 2 else .08) and waypoint < len(points)-1:
                        waypoint += 1
                current = np.array([data.joint(f'toggle_{i}').qpos[0] for i in range(4)])
                max_toggle = np.maximum(max_toggle, abs(current - toggles))
                if not phases or phases[-1]['phase'] != phase:
                    phases.append({'time_s': float(data.time), 'phase': phase})
                records.append([float(data.time), *pose.tolist(), yaw, float(slip), waypoint])
                next_control += .01
            control.update(data, forward=forward, turn=turn)
            mujoco.mj_step(model, data)
            check_state(model, data)
            for contact_index, contact in enumerate(data.contact):
                names = [model.body(int(model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
                if any(n.startswith('robot') for n in names) and any(n.startswith(('pin_', 'toggle_')) for n in names):
                    contacts.add(tuple(sorted(names)))
                    if 1. <= data.time < 3.8:
                        drift_contacts.update(n for n in names if n.startswith('pin_'))
                    if 'toggle_1' in names:
                        robot_name = next(n for n in names if n.startswith('robot'))
                        first_toggle_contacts.setdefault(robot_name, {'time_s': float(data.time), 'angle_rad': float(data.joint('toggle_1').qpos[0])})
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, contact_index, force)
                        world_force = contact.frame.reshape(3, 3).T @ force[:3]
                        world_force *= 1 if names[1] == 'toggle_1' else -1
                        axis = data.body('toggle_1').xmat.reshape(3, 3) @ model.jnt_axis[model.joint('toggle_1').id]
                        torque = float(np.dot(np.cross(contact.pos - data.body('toggle_1').xpos, world_force), axis))
                        toggle_contact_impulses[robot_name] = toggle_contact_impulses.get(robot_name, 0.) + abs(torque) * model.opt.timestep
            if writer and data.time >= next_frame:
                camera.lookat[:] = data.body('robot').xpos + [0, 0, .15]
                picture = Image.fromarray(frame(renderer, data, camera))
                caption = ImageDraw.Draw(picture)
                caption.rectangle((12, 12, 655, 70), fill=(15, 20, 30))
                label = ('physical slide into pins' if slip > .15 else 'turn toward pins') if phase == 'drift' else phase
                caption.text((24, 20), f'{data.time:05.2f}s | {label} | lateral {slip:.2f} m/s', fill='white', font=ImageFont.load_default(size=24))
                writer.append_data(np.asarray(picture))
                if 1. < data.time < 3.8 and slip > saved_drift_slip:
                    picture.save(output / 'sequence-drift.png')
                    saved_drift_slip = slip
                if first_toggle_contacts and not (output / 'sequence-toggle.png').exists():
                    picture.save(output / 'sequence-toggle.png')
                next_frame += 1/30
        np.savez(output / 'sequence-state.npz', qpos=data.qpos, qvel=data.qvel)
        displacement = np.linalg.norm(data.xpos[pins, :2] - initial[:, :2], axis=1)
        result = {'simulation_seconds': float(data.time), 'wall_seconds': time.perf_counter()-wall,
                  'claw_closure': metadata['claw']['closure_springs'],
                  'start_m_rad': [-.15, -.90, 0.], 'final_position_m': data.body('robot').xpos.tolist(),
                  'max_sideways_speed_m_s': max_slip, 'drift_sideways_speed_m_s': max_drift_slip, 'max_toggle_rotation_rad': max_toggle.tolist(),
                  'pin_displacements_m': {model.body(i).name: float(v) for i, v in zip(pins, displacement)},
                  'interactions': sorted(contacts), 'warnings': [int(w.number) for w in data.warning],
                  'waypoint_reached': waypoint, 'phases': phases, 'arm_travel_rad': arm_travel, 'lift_rest_crank_range_rad': lift_rest_range,
                  'first_toggle_contacts': first_toggle_contacts, 'absolute_toggle_contact_torque_impulse_Nm_s': toggle_contact_impulses, 'wheel_torque_only': True, 'propulsion': 'Wheel motor torque; lift and claw hold zero targets; toggle arms are passive.',
                  'limitations': metadata['limitations'] + ['Uncalibrated contact and motor parameters; simulation is not hardware validation.']}
        drift_displacement = np.linalg.norm(drift_final[:, :2] - initial[:, :2], axis=1)
        result['drift_pin_displacements_m'] = {model.body(i).name: float(v) for i, v in zip(pins, drift_displacement) if model.body(i).name in drift_contacts}
        result['drift_verified'] = max_drift_slip > .15 and any(v > .10 for v in result['drift_pin_displacements_m'].values())
        passive_impulse = sum(value for name, value in toggle_contact_impulses.items() if name.startswith('robot_toggle_'))
        result['passive_arm_toggle_torque_impulse_Nm_s'] = passive_impulse
        other_contacts = [record for name, record in first_toggle_contacts.items() if not name.startswith('robot_toggle_')]
        before_other = abs(min(other_contacts, key=lambda record: record['time_s'])['angle_rad']) if other_contacts else float(max_toggle[1])
        result['toggle_rotation_before_other_robot_contact_rad'] = before_other
        result['toggle_verified'] = bool(before_other > .3 and passive_impulse > 1e-5)
        result['outcome'] = 'passed' if result['drift_verified'] and result['toggle_verified'] else 'incomplete'
        (output / 'sequence.json').write_text(json.dumps(result, indent=2))
        np.savetxt(output / 'sequence.csv', records, delimiter=',', header='time_s,x_m,y_m,z_m,yaw_rad,sideways_speed_m_s,waypoint', comments='')
        if renderer:
            Image.fromarray(frame(renderer, data, camera)).save(output / 'sequence-final.png')
        return result
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()


def verify_sequence(output):
    """Run the deterministic physical scenario and assert observable outcomes."""
    result = run_sequence(output, video=False)
    assert result['drift_verified'], 'No simultaneous lateral slide and contacted-pin displacement'
    assert result['toggle_verified'], 'Wall toggle did not rotate through physical contact'
    assert not any(result['warnings']), 'MuJoCo emitted warnings'
    assert any('robot_toggle_' in name for name in result['first_toggle_contacts']), 'No passive-arm toggle contact'
    return result
