import json
import math

import mujoco
import numpy as np
import pytest

from override_sim.field import CAD_TO_FIELD, ASSETS, Layout
from override_sim.scene import check_state, load


@pytest.mark.parametrize('layout,total_pins', [(Layout.MATCH, 33), (Layout.SKILLS, 32)])
def test_radial_pin_setup_keeps_cad_tilt_after_gravity_settles(layout, total_pins):
    spec, model, data = load(layout)
    pins = [p for p in spec.metadata['placements'] if p['kind'] == 'pin']
    loose = [p for p in pins if p['name'].endswith(('Red-Neutral', 'Blue-Neutral'))]
    cups = [p for p in spec.metadata['placements'] if p['kind'] == 'cup']
    source = json.loads((ASSETS / 'manifest.json').read_text())
    assert len(pins) == total_pins
    assert len(cups) == 36
    assert len(loose) == 16
    clusters = {}
    initial = {}
    for pin in loose:
        name = 'pin_' + pin['source'].replace('/', '_')
        body = model.body(name)
        assert model.joint(int(body.jntadr[0])).type == mujoco.mjtJoint.mjJNT_FREE
        cad_pose = CAD_TO_FIELD @ np.array(source['nodes'][pin['source']]['world_transform'])
        assert data.body(name).xpos == pytest.approx(cad_pose[:3, 3], abs=1e-9)
        assert data.body(name).xmat.reshape(3, 3) == pytest.approx(cad_pose[:3, :3], abs=1e-9)
        cup = min(cups, key=lambda p: np.linalg.norm(np.array(p['position'][:2]) - cad_pose[:2, 3]))
        clusters[cup['source']] = clusters.get(cup['source'], 0) + 1
        outward = cad_pose[:2, 3] - np.array(cup['position'][:2])
        outward /= np.linalg.norm(outward)
        axis = cad_pose[:3, 1]
        assert np.dot(axis[:2], outward) > .96
        assert math.degrees(math.asin(axis[2])) == pytest.approx(14.973, abs=.001)
        initial[name] = (cad_pose[:3, 3], axis)
        for contact in data.contact:
            if body.id in model.geom_bodyid[contact.geom]:
                other_bodies = [model.body(int(model.geom_bodyid[geom])).name for geom in contact.geom]
                # The unreduced STEP assembly overlaps adjacent cup tips by up to 0.317 mm.
                tolerance = .00035 if any(name.startswith('cup_') for name in other_bodies) else 1e-5
                assert contact.dist > -tolerance
    assert sorted(clusters.values()) == [4, 4, 4, 4]

    mujoco.mj_step(model, data, nstep=5000)
    check_state(model, data)
    for name, (position, axis) in initial.items():
        body = data.body(name)
        assert body.xpos == pytest.approx(position, abs=.0002)
        assert body.xmat.reshape(3, 3)[:, 1] == pytest.approx(axis, abs=.001)
        contacts = [c for c in data.contact if model.body(name).id in model.geom_bodyid[c.geom]]
        assert contacts
        assert min(c.dist for c in contacts) > -1e-5

    pin = model.body(next(iter(initial)))
    dof = model.jnt_dofadr[pin.jntadr[0]]
    position = data.body(pin.id).xpos.copy()
    data.qvel[dof:dof + 3] = [0, 0, .5]
    mujoco.mj_step(model, data, nstep=25)
    assert data.body(pin.id).xpos[2] > position[2] + .005
