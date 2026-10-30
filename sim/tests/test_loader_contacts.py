"""Probe the real loader openings and adjacent material with physical contacts."""
import xml.etree.ElementTree as ET

import mujoco
import pytest

from override_sim.field import Layout, build_scene
from override_sim.loader_contacts import replace_loader_contacts


@pytest.fixture(scope='module')
def sleeve():
    root = ET.fromstring(build_scene(Layout.MATCH).xml)
    root.remove(root.find('actuator'))
    world = root.find('worldbody')
    for child in list(world):
        world.remove(child)
    ET.SubElement(world, 'body', name='loader_0')
    replace_loader_contacts(root)
    probe = ET.SubElement(world, 'body', name='probe')
    ET.SubElement(probe, 'freejoint')
    ET.SubElement(probe, 'geom', type='sphere', size='.0005', mass='.01', contype='8', conaffinity='3')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    return model


@pytest.mark.parametrize('position, touches_material', [
    ([.049, .02, -.02], False),  # Lower-front corner cut away in the CAD.
    ([.052, .2, .03], True),    # Side wall above the cutout.
    ([0., .2, .049], False),    # Open rear of the U section.
    ([0., .2, -.0445], True),   # Solid front wall.
    ([.0518, .08325, .02956], False),  # Small side-wall mounting hole.
])
def test_sleeve_preserves_openings_and_material(sleeve, position, touches_material):
    data = mujoco.MjData(sleeve)
    data.qpos[:3] = position
    mujoco.mj_forward(sleeve, data)
    assert bool(data.ncon) == touches_material


def test_sleeve_can_complete_human_operated_lift_with_guides_present():
    root = ET.fromstring(build_scene(Layout.MATCH).xml)
    replace_loader_contacts(root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    for index in range(4):
        data.ctrl[model.actuator(f'loader_{index}_lift').id] = .2185
    for _ in range(1000):
        mujoco.mj_step(model, data)
    for index in range(4):
        assert data.joint(f'loader_{index}').qpos[0] == pytest.approx(.2185, abs=.004)
    assert not any(warning.number for warning in data.warning)
