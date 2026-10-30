"""Source-section contacts preserving the chassis front goal opening."""

import json
import xml.etree.ElementTree as ET
import numpy as np
from .field import numbers


def add_chassis_contacts(builder):
    from .robot import CAD_ROTATION, ROBOT_ASSETS

    profile = json.loads((ROBOT_ASSETS / "chassis_contacts.json").read_text())
    chassis, pivot = builder.bodies["robot"]
    for index, panel in enumerate(profile["panels"]):
        name = f"robot_chassis_panel_{index}"
        vertices = (np.asarray(panel["vertices"]) - pivot) @ CAD_ROTATION.T
        ET.SubElement(
            builder.asset, "mesh", name=name, vertex=numbers(vertices.ravel())
        )
        ET.SubElement(
            chassis,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            group="3",
            rgba=".9 .3 .1 .3",
            contype="8",
            conaffinity="3",
            mass="0",
        )
    return {key: value for key, value in profile.items() if key != "panels"}
