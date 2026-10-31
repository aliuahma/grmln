"""Display simplification must leave motor-driven physical behavior unchanged."""
from pathlib import Path
import json

import pytest
import mujoco
import numpy as np

from override_sim.robot import build_robot_scene
from override_sim.robot_control import RobotControl


def test_simple_visuals_preserve_physics_and_motion():
    assets = Path(__file__).parents[1] / 'override_sim/assets/robot'
    definitions = json.loads((assets / 'simple_visuals.json').read_text())['definitions']
    if any(not (assets / definition['source_mesh']).is_file() for definition in definitions.values()):
        pytest.skip('Optional original CAD visuals are absent; regenerate from STEP for CAD parity')
    models, controls, states = [], [], []
    for mode in ('cad', 'simple'):
        xml, metadata = build_robot_scene(visual_mode=mode)
        model = mujoco.MjModel.from_xml_string(xml)
        models.append(model)
        controls.append(RobotControl(model, metadata))
        states.append(mujoco.MjData(model))
    cad, simple = models
    for attribute in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'jnt_pos', 'jnt_axis',
                      'jnt_range', 'dof_armature', 'eq_data', 'eq_solref', 'eq_solimp', 'actuator_trnid'):
        np.testing.assert_array_equal(getattr(cad, attribute), getattr(simple, attribute))
    collisions = [(m.geom_contype != 0) | (m.geom_conaffinity != 0) for m in models]
    for attribute in ('geom_type', 'geom_pos', 'geom_quat', 'geom_size', 'geom_friction', 'geom_contype', 'geom_conaffinity'):
        np.testing.assert_array_equal(getattr(cad, attribute)[collisions[0]], getattr(simple, attribute)[collisions[1]])
    assert simple.nmeshface < cad.nmeshface / 2
    for step in range(2000):
        for model, control, data in zip(models, controls, states):
            control.update(data, forward=.15 if step > 500 else 0., lift=.3 if step > 1000 else 0., intake=-.2)
            mujoco.mj_step(model, data)
        np.testing.assert_allclose(states[0].qpos, states[1].qpos, atol=1e-12, rtol=0.)
    assert not any(w.number for data in states for w in data.warning)


def test_packaged_simple_visuals_run_without_originals():
    xml, metadata = build_robot_scene()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    control = RobotControl(model, metadata)
    for step in range(2000):
        control.update(data, forward=.15 if step > 500 else 0., lift=.3 if step > 1000 else 0., intake=-.2)
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert not any(w.number for w in data.warning)


def test_missing_cad_mesh_explains_restoration(tmp_path):
    from override_sim.robot_visuals import RobotVisuals

    with pytest.raises(FileNotFoundError, match='extract_robot.py'):
        RobotVisuals(tmp_path, mode='cad').mesh_path('missing', 'meshes/missing.obj')
