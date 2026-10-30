from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import pytest

from override_sim import scene
from override_sim.field import Layout
from override_sim.route import Route, add_robot


@pytest.mark.parametrize('piece', ['pin_0', 'cup_0', 'toggle_0', 'loader_0'])
@pytest.mark.parametrize('piece_first', [False, True])
@pytest.mark.parametrize('brief', [False, True])
def test_route_records_real_contacts_like_reference(tmp_path, monkeypatch, piece, piece_first, brief):
    root = ET.fromstring('<mujoco><option timestep=".002"/><worldbody>'
                         '<geom type="plane" size="2 2 .1"/></worldbody><actuator/></mujoco>')
    worldbody, actuator = root.find('worldbody'), root.find('actuator')
    obstacle = ET.Element('body', name=piece, pos='.42 0 .09')
    ET.SubElement(obstacle, 'geom', type='sphere', size='.06')
    if piece_first:
        worldbody.append(obstacle)
    route = Route((0, 0, 0), ((1, 0),), 2, .3)
    add_robot(worldbody, actuator, route.start)
    if not piece_first:
        worldbody.append(obstacle)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    monkeypatch.setattr(scene, 'load', lambda *args, **kwargs: (SimpleNamespace(metadata={'assumptions': []}), model, data))
    expected = set()
    contact_steps = []
    checked_steps = []
    original_check = scene.check_state

    def observe(model, data):
        original_check(model, data)
        checked_steps.append(float(data.time))
        for contact in data.contact:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or 'world'
                     for g in (contact.geom1, contact.geom2)]
            if any(name.startswith('robot') for name in names) and any(
                    name.startswith(('pin_', 'cup_', 'toggle_', 'loader_')) for name in names):
                expected.add(tuple(sorted(names)))
                contact_steps.append(data.time)
                if brief:
                    obstacle_geoms = model.geom_bodyid == model.body(piece).id
                    model.geom_contype[obstacle_geoms] = 0
                    model.geom_conaffinity[obstacle_geoms] = 0

    monkeypatch.setattr(scene, 'check_state', observe)
    result = scene.run_route(Layout.MATCH, route, tmp_path, video=False)
    assert expected
    if brief:
        assert len(contact_steps) == 1
    else:
        assert len(contact_steps) > 1
    assert len(checked_steps) == round(result['simulation_seconds'] / model.opt.timestep)
    assert result['interactions'] == sorted(expected)
    assert any(piece in pair for pair in result['interactions'])
    assert result['outcome'] == 'timeout'
