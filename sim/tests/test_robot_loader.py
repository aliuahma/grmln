"""Verify real insertion order and reject support-only or impact-trapped states."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from override_sim.field import Layout, build_scene, numbers
from override_sim.robot_loader import GOAL, PIECES, placement_status, qualified_stack_pose, qualified_tire_grip


@pytest.fixture(scope='module')
def isolated_goal_xml():
    root = ET.fromstring(build_scene(Layout.MATCH, timestep=.001).xml)
    world = root.find('worldbody')
    for body in list(world.findall('body')):
        if body.get('name') not in PIECES:
            world.remove(body)
    for geom in list(world.findall('geom')):
        position = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
        if geom.get('name') != 'floor' and np.linalg.norm(position[:2]-GOAL) > .12:
            world.remove(geom)
    root.remove(root.find('actuator'))
    return ET.tostring(root, encoding='unicode')


def settled_pair(xml, pin_height=.09, cup_height=.185, flipped=True):
    root = ET.fromstring(xml)
    for name, height in ((PIECES[1], pin_height), (PIECES[0], cup_height)):
        body = root.find(f"worldbody/body[@name='{name}']")
        body.set('pos', numbers([*GOAL, height]))
        if flipped:
            result = np.zeros(4)
            mujoco.mju_mulQuat(result, np.array([0., 1., 0., 0.]),
                              np.fromstring(body.get('quat'), sep=' '))
            body.set('quat', numbers(result))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=5000)
    base_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == 0 and model.geom_group[i] == 3
                  and np.linalg.norm(data.geom_xpos[i, :2]-GOAL) < .10}
    return model, data, base_geoms


def test_pin_seats_in_goal_and_cup_nests_over_pin(isolated_goal_xml):
    model, data, base_geoms = settled_pair(isolated_goal_xml)
    assert all(placement_status(model, data, base_geoms).values())
    assert data.body(PIECES[1]).xpos[2] == pytest.approx(.0845, abs=.0002)
    assert data.body(PIECES[0]).xpos[2]-data.body(PIECES[1]).xpos[2] == pytest.approx(.0856, abs=.0002)
    assert not np.any(data.warning.number)


def test_old_cup_below_pin_order_is_not_insertion(isolated_goal_xml):
    model, data, base_geoms = settled_pair(isolated_goal_xml, .2773, .18, flipped=False)
    checks = placement_status(model, data, base_geoms)
    assert not checks['pin_contacts_goal']
    assert not checks['cup_nested_over_pin']
    assert not all(checks.values())


def test_impact_trapped_pin_is_rejected_despite_contacts(isolated_goal_xml):
    model, data, base_geoms = settled_pair(isolated_goal_xml, .25, .345)
    checks = placement_status(model, data, base_geoms)
    assert checks['pin_contacts_goal'] and checks['cup_contacts_pin']
    assert not checks['contacts_clean']
    assert not all(checks.values())


def test_offset_and_tilt_cannot_pass_nesting_gate(isolated_goal_xml):
    model, settled, base_geoms = settled_pair(isolated_goal_xml)
    offset = mujoco.MjData(model)
    offset.qpos[:], offset.qvel[:] = settled.qpos, settled.qvel
    for name in PIECES:
        address = int(model.jnt_qposadr[int(model.body(name).jntadr[0])])
        offset.qpos[address] += .02
    mujoco.mj_forward(model, offset)
    assert not placement_status(model, offset, base_geoms)['pin_seated_in_goal']

    cup_offset = mujoco.MjData(model)
    cup_offset.qpos[:], cup_offset.qvel[:] = settled.qpos, settled.qvel
    cup_address = int(model.jnt_qposadr[int(model.body(PIECES[0]).jntadr[0])])
    cup_offset.qpos[cup_address] += .02
    mujoco.mj_forward(model, cup_offset)
    checks = placement_status(model, cup_offset, base_geoms)
    assert checks['pin_seated_in_goal']
    assert not checks['cup_nested_over_pin']

    tilted = mujoco.MjData(model)
    tilted.qpos[:], tilted.qvel[:] = settled.qpos, settled.qvel
    rotation = np.array([np.cos(np.deg2rad(5)), np.sin(np.deg2rad(5)), 0., 0.])
    for name in PIECES:
        address = int(model.jnt_qposadr[int(model.body(name).jntadr[0])])
        result = np.zeros(4)
        mujoco.mju_mulQuat(result, rotation, tilted.qpos[address+3:address+7])
        tilted.qpos[address+3:address+7] = result
    mujoco.mj_forward(model, tilted)
    assert not placement_status(model, tilted, base_geoms)['upright']


def test_capture_gate_rejects_arm_suspension_and_requires_loaded_tires_on_both_objects():
    assert not qualified_tire_grip([[0., 0.], [0., 0.]])
    assert not qualified_tire_grip([[.736, 1.025], [0., 0.]])
    assert not qualified_tire_grip([[.736, 0.], [.283, 0.]])
    assert qualified_tire_grip([[.986, 0.], [0., .969]])
    assert qualified_tire_grip([[.736, 1.025], [.283, 0.]])


def test_capture_pose_requires_upright_coaxial_correct_order():
    positions = np.array([[0., 0., .168085], [0., 0., .0825]])
    axes = np.array([[0., 0., 1.], [0., 0., 1.]])
    assert qualified_stack_pose(positions, axes)
    assert not qualified_stack_pose(positions[::-1], axes)
    offset_cup = positions.copy()
    offset_cup[0, 0] = .02
    assert not qualified_stack_pose(offset_cup, axes)
    angle = np.radians(13.5)
    tilted_axes = np.tile([np.sin(angle), 0., np.cos(angle)], (2, 1))
    tilted_positions = np.array([positions[1]+.085585*tilted_axes[0], positions[1]])
    assert not qualified_stack_pose(tilted_positions, tilted_axes)
