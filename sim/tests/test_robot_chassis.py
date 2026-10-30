"""Functional void and material probes for the CAD chassis replacement."""

import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from override_sim.robot import build_robot_scene


@pytest.mark.parametrize(
    "position, expected",
    [
        ((0.150, 0, 0.030), False),
        ((0.150, 0.025, 0.030), False),
        ((0.0815, 0, 0.030), True),
        ((0.10515, 0.06909, 0.02958), True),
        ((0, 0, 0.010), False),
    ],
)
def test_front_goal_opening_preserves_source_material(position, expected):
    xml, _ = build_robot_scene(fixed=True)
    root = ET.fromstring(xml)
    for geom in root.findall(".//geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        if geom.get("name", "").startswith("robot_chassis_panel_"):
            geom.set("contype", "8")
            geom.set("conaffinity", "2")
    body = ET.SubElement(
        root.find("worldbody"), "body", name="probe", pos=" ".join(map(str, position))
    )
    ET.SubElement(body, "freejoint")
    ET.SubElement(
        body,
        "geom",
        type="sphere",
        size=".001",
        mass=".001",
        contype="2",
        conaffinity="8",
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert bool(data.ncon) is expected


def test_chassis_contacts_do_not_add_inferred_mass():
    xml, _ = build_robot_scene(fixed=True)
    root = ET.fromstring(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    chassis = root.find("worldbody/body[@name='robot']")
    for geom in list(chassis.findall("geom")):
        if geom.get("name", "").startswith("robot_chassis_panel_"):
            chassis.remove(geom)
    without = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    np.testing.assert_array_equal(model.body_mass, without.body_mass)
    np.testing.assert_array_equal(model.body_inertia, without.body_inertia)
    np.testing.assert_array_equal(model.body_ipos, without.body_ipos)
    np.testing.assert_array_equal(model.body_iquat, without.body_iquat)
