import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from override_sim.field import FieldBuilder, Layout
from override_sim.scene import check_state


def piece_pair(moving, offset=(0, 0), yaw_degrees=0):
    builder = FieldBuilder(Layout.MATCH, .002)
    builder.game_piece('1/23', 'pin')
    builder.game_piece('1/24', 'cup')
    pin = builder.world.find("body[@name='pin_1_23']")
    cup = builder.world.find("body[@name='cup_1_24']")
    fixed = cup if moving == 'pin' else pin
    fixed.remove(fixed.find('freejoint'))
    body = pin if moving == 'pin' else cup
    position = np.fromstring(body.get('pos'), sep=' ')
    if moving == 'pin':
        position[2] += .04
    else:
        position[2] = np.fromstring(pin.get('pos'), sep=' ')[2] + .085 + .04
    position[:2] += offset
    body.set('pos', ' '.join(map(str, position)))
    angle = np.deg2rad(yaw_degrees) / 2
    rotation = np.array([np.cos(angle), 0, 0, np.sin(angle)])
    original = np.fromstring(body.get('quat'), sep=' ')
    quaternion = np.empty(4)
    mujoco.mju_mulQuat(quaternion, rotation, original)
    body.set('quat', ' '.join(map(str, quaternion)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(builder.root).decode())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize('moving', ['pin', 'cup'])
def test_pin_in_cup_and_cup_over_pin_settle_at_cad_seating_depth(moving):
    model, data = piece_pair(moving)
    body = data.body(f'{moving}_1_{23 if moving == "pin" else 24}')
    initial_height = float(body.xpos[2])
    mujoco.mj_step(model, data, nstep=1500)
    check_state(model, data)
    pin, cup = data.body('pin_1_23'), data.body('cup_1_24')
    assert initial_height - body.xpos[2] > .038
    assert abs(abs(pin.xpos[2] - cup.xpos[2]) - .085) < .0015
    assert np.linalg.norm(pin.xpos[:2] - cup.xpos[:2]) < .001
    contacts = [contact for contact in data.contact
                if set(model.geom_bodyid[contact.geom]) == {pin.id, cup.id}]
    assert contacts
    assert min(contact.dist for contact in contacts) > -.0001
    assert all(model.geom_contype[contact.geom1] != 4 and model.geom_contype[contact.geom2] != 4
               for contact in contacts)


@pytest.mark.parametrize('goal_id', ['1/2', '1/6', '1/10'])
def test_goal_pin_drop_reaches_cad_seating_depth(goal_id):
    builder = FieldBuilder(Layout.MATCH, .002)
    builder.goal(goal_id)
    goal = builder.placements[-1]
    builder.game_piece('1/27', 'pin')
    pin = builder.world.find("body[@name='pin_1_27']")
    x, y, stand = goal['position']
    cad_seated_height = stand + .0845
    pin.set('pos', f'{x} {y} {cad_seated_height + .03}')
    model = mujoco.MjModel.from_xml_string(ET.tostring(builder.root).decode())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=2000)
    check_state(model, data)
    body = data.body('pin_1_27')
    assert body.xpos[2] == pytest.approx(cad_seated_height, abs=.0015)
    assert np.linalg.norm(body.xpos[:2] - [x, y]) < .001
    contacts = [contact for contact in data.contact if body.id in model.geom_bodyid[contact.geom]]
    assert contacts
    assert min(contact.dist for contact in contacts) > -.0001


@pytest.mark.parametrize('moving', ['pin', 'cup'])
@pytest.mark.parametrize('offset,yaw_degrees', [
    ((.002, 0), 7.5),
    ((-.002, 0), -7.5),
    ((0, .002), 22.5),
    ((0, -.002), -22.5),
])
def test_offset_and_half_sector_yaw_drop_does_not_catch_on_contact_wedges(moving, offset, yaw_degrees):
    model, data = piece_pair(moving, offset, yaw_degrees)
    initial_height = float(data.body(f'{moving}_1_{23 if moving == "pin" else 24}').xpos[2])
    mujoco.mj_step(model, data, nstep=2000)
    check_state(model, data)
    pin, cup = data.body('pin_1_23'), data.body('cup_1_24')
    body = pin if moving == 'pin' else cup
    assert initial_height - body.xpos[2] > .038
    assert abs(abs(pin.xpos[2] - cup.xpos[2]) - .085) < .0015
    assert np.linalg.norm(pin.xpos[:2] - cup.xpos[:2]) < .001
    contacts = [contact for contact in data.contact
                if set(model.geom_bodyid[contact.geom]) == {pin.id, cup.id}]
    assert contacts
    assert min(contact.dist for contact in contacts) > -.0001
