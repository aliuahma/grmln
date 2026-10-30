from collections import Counter
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from override_sim.field import ASSETS, Layout, build_scene
from override_sim.route import Route, RouteFollower, load_route
from override_sim.scene import load, check_state, run_route


@pytest.mark.parametrize('layout,pins', [(Layout.MATCH, 33), (Layout.SKILLS, 32)])
def test_university_inventory_and_goal_occupancy(layout, pins):
    spec = build_scene(layout)
    placements = spec.metadata['placements']
    assert Counter(p['kind'] for p in placements) == {'cup': 36, 'pin': pins, 'goal': 9, 'toggle': 4, 'loader': 4}
    positions = np.array([p['position'][:2] for p in placements if p['kind'] == 'pin'])
    for goal in (p for p in placements if p['kind'] == 'goal'):
        count = sum(np.linalg.norm(positions - goal['position'][:2], axis=1) < .01)
        assert count == (1 if layout == Layout.MATCH and 'Goal 0' in goal['name'] else 0)
    colors = Counter(p['name'].split('80x_')[-1] for p in placements if p['kind'] == 'pin')
    assert colors['Red-Neutral'] == colors['Blue-Neutral'] == 8
    assert colors['Red-Blue'] == 4
    assert colors['Neutral-Neutral'] == pins - 20


def test_cad_frame_and_known_landmarks():
    metadata = build_scene().metadata
    goals = [p for p in metadata['placements'] if p['kind'] == 'goal']
    blue2 = next(p for p in goals if p['name'].endswith('Blue 2'))
    assert blue2['position'] == pytest.approx([1.196117068, .598058534, 0], abs=1e-7)
    north = next(p for p in metadata['placements'] if p['kind'] == 'toggle')
    assert north['position'] == pytest.approx([0, 1.814856899, .335340194], abs=1e-7)
    assert north['axis'] == pytest.approx([1, 0, 0], abs=1e-10)
    source = json.loads((ASSETS / 'manifest.json').read_text())
    assert source['statistics']['root_children'] == [184]
    assert source['coordinate_system']['floor_top_y'] == pytest.approx(-.2945892)


def test_mechanism_dofs_and_visuals_do_not_collide():
    _, model, _ = load()
    for i in range(4):
        hinge, slide = model.joint(f'toggle_{i}'), model.joint(f'loader_{i}')
        assert hinge.type == mujoco.mjtJoint.mjJNT_HINGE
        assert not hinge.limited
        assert slide.type == mujoco.mjtJoint.mjJNT_SLIDE
        assert slide.range == pytest.approx([0, .2185])
    display = model.geom_group == 1
    assert np.all(model.geom_contype[display] == 0)
    assert np.all(model.geom_conaffinity[display] == 0)


def test_full_field_settles_without_warnings_or_piece_escape():
    _, model, data = load(Layout.SKILLS)
    mujoco.mj_step(model, data, nstep=5000)
    check_state(model, data)
    piece_ids = [i for i in range(model.nbody) if (model.body(i).name or '').startswith(('cup_', 'pin_'))]
    assert np.max(np.abs(data.xpos[piece_ids, :2])) < 1.80
    assert np.min(data.xpos[piece_ids, 2]) > .01
    for body_id in piece_ids:
        dof = model.jnt_dofadr[model.body_jntadr[body_id]]
        assert np.linalg.norm(data.qvel[dof:dof + 3]) < .01
        assert np.linalg.norm(data.qvel[dof + 3:dof + 6]) < .1


def test_route_rejects_start_inside_goal(tmp_path):
    route = Route((0, 0, 0), ((0, 0),), 1)
    with pytest.raises(ValueError, match='start overlaps'):
        run_route(Layout.MATCH, route, tmp_path, video=False)


def test_example_route_completes_through_wheel_contact():
    route = Route((-1.45, -.9, 0), ((-.9, -.9), (-.9, -.3)), 15, .3)
    _, model, data = load(Layout.SKILLS, robot_start=route.start)
    follower = RouteFollower(route)
    while data.time < route.timeout_s and not follower.completed:
        follower.update(model, data)
        mujoco.mj_step(model, data, nstep=5)
    check_state(model, data)
    assert follower.completed
    assert np.linalg.norm(data.body('robot').xpos[:2] - route.waypoints[-1]) < .07


def test_wall_blocks_driven_robot():
    _, model, data = load(Layout.SKILLS, robot_start=(-1.45, -.9, np.pi))
    for side in ('left', 'right'):
        data.ctrl[model.actuator(f'robot_{side}_drive').id] = 8
    mujoco.mj_step(model, data, nstep=1500)
    check_state(model, data)
    assert -1.70 < data.body('robot').xpos[0] < -1.60


@pytest.mark.parametrize('start,points,time,speed', [((0, 0, 0), (), 1, .3), ((float('nan'), 0, 0), ((0, 0),), 1, .3),
                                                   ((2, 0, 0), ((0, 0),), 1, .3), ((0, 0, 0), ((0, 0),), -1, .3)])
def test_invalid_route_rejected(start, points, time, speed):
    with pytest.raises(ValueError):
        Route(start, points, time, speed)
