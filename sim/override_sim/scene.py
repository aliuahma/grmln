from __future__ import annotations

from pathlib import Path
import json
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from .field import Layout, build_scene


def load(layout=Layout.MATCH, *, timestep=.002, robot_start=None):
    spec = build_scene(layout, timestep=timestep, robot_start=robot_start)
    model = mujoco.MjModel.from_xml_string(spec.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return spec, model, data


def check_state(model, data):
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError('Nonfinite simulation state')
    warnings = [int(w.number) for w in data.warning]
    if any(warnings):
        raise RuntimeError(f'MuJoCo warnings: {warnings}')


def frame(renderer, data, camera='oblique', collisions=False):
    options = mujoco.MjvOption()
    options.geomgroup[3] = collisions
    renderer.update_scene(data, camera=camera, scene_option=options)
    return renderer.render()


def render(layout, output, *, seconds=6., fps=30, width=1280, height=960):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    spec, model, data = load(layout)
    (output / 'scene.xml').write_text(spec.xml)
    (output / 'scene.json').write_text(json.dumps(spec.metadata, indent=2))
    with mujoco.Renderer(model, height, width) as renderer:
        for camera in ('overhead', 'oblique'):
            Image.fromarray(frame(renderer, data, camera)).save(output / f'{camera}.png')
        Image.fromarray(frame(renderer, data, 'oblique', True)).save(output / 'collisions.png')
        trajectory = []
        with imageio.get_writer(output / 'mechanisms.mp4', fps=fps, codec='libx264', quality=8, macro_block_size=1) as writer:
            for i in range(int(seconds * fps)):
                target_time = (i + 1) / fps
                while data.time < target_time - model.opt.timestep / 2:
                    for j in range(4):
                        data.ctrl[model.actuator(f'loader_{j}_lift').id] = .2185 * (.5 - .5 * np.cos(2 * np.pi * data.time / seconds))
                        joint = model.joint(f'toggle_{j}')
                        data.qfrc_applied[joint.dofadr[0]] = .018 if data.time < seconds * .65 else 0
                    mujoco.mj_step(model, data)
                check_state(model, data)
                trajectory.append({'time': float(data.time), 'toggles': [float(data.joint(f'toggle_{j}').qpos[0]) for j in range(4)],
                                   'loaders': [float(data.joint(f'loader_{j}').qpos[0]) for j in range(4)]})
                writer.append_data(frame(renderer, data))
        Image.fromarray(frame(renderer, data)).save(output / 'settled.png')
    (output / 'mechanisms.json').write_text(json.dumps(trajectory, indent=2))
    return {'layout': Layout(layout).value, 'seconds': float(data.time), 'frames': len(trajectory), 'output': str(output)}


def run_route(layout, route, output, *, video=True, timestep=.002):
    from .route import RouteFollower
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    spec, model, data = load(layout, timestep=timestep, robot_start=route.start)
    for contact in data.contact:
        bodies = [model.body(int(model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
        if any(name.startswith('robot') for name in bodies) and contact.dist < -.003:
            raise ValueError(f'Robot start overlaps field geometry by {-contact.dist:.4f} m; choose a clear start')
    body_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or 'world'
                       for body in model.geom_bodyid)
    robot_geoms = np.array([name.startswith('robot') for name in body_names])
    game_geoms = np.array([name.startswith(('pin_', 'cup_', 'toggle_', 'loader_')) for name in body_names])
    follower = RouteFollower(route)
    records = []
    renderer = mujoco.Renderer(model, 720, 960) if video else None
    writer = imageio.get_writer(output / 'route.mp4', fps=30, codec='libx264', macro_block_size=1) if video else None
    next_frame = 0.; next_control = 0.; wall = time.perf_counter()
    collisions = set()
    try:
        while data.time < route.timeout_s:
            if data.time >= next_control:
                complete = follower.update(model, data)
                next_control += .01
                pose = data.body('robot').xpos
                records.append([float(data.time), *pose.tolist(), float(np.linalg.norm(pose[:2] - route.waypoints[min(follower.index, len(route.waypoints) - 1)]))])
                if complete:
                    break
            mujoco.mj_step(model, data)
            check_state(model, data)
            contact_geoms = data.contact.geom
            first, second = contact_geoms[:, 0], contact_geoms[:, 1]
            interactions = ((robot_geoms[first] & game_geoms[second]) |
                            (robot_geoms[second] & game_geoms[first]))
            for first_geom, second_geom in contact_geoms[interactions]:
                collisions.add(tuple(sorted((body_names[first_geom], body_names[second_geom]))))
            if writer and data.time >= next_frame:
                writer.append_data(frame(renderer, data)); next_frame += 1 / 30
        result = {**follower.report(), 'layout': Layout(layout).value, 'simulation_seconds': float(data.time),
                  'wall_seconds_including_rendering': time.perf_counter() - wall, 'interactions': sorted(collisions),
                  'robot': 'hypothetical differential drive; not calibrated to team robot',
                  'outcome': 'completed' if follower.completed else 'timeout', 'assumptions': spec.metadata['assumptions']}
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        np.savetxt(output / 'trajectory.csv', records, delimiter=',', header='time_s,x_m,y_m,z_m,error_m', comments='')
        if renderer:
            Image.fromarray(frame(renderer, data, 'overhead')).save(output / 'route-final.png')
        return result
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
