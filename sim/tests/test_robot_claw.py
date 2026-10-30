"""Powered claw motion must reach the rollers through the passive jaw carriers."""
import mujoco
import numpy as np
import pytest

from override_sim.robot_control import RobotControl
from override_sim.robot_verification import _commands, _model, _pose


def transmission_errors(data):
    positions = {f'{side}_{part}': float(data.joint(f'robot_claw_{side}_{part}').qpos[0])
                 for side in ('left', 'right') for part in ('shaft', 'jaw', 'roller')}
    return [positions[f'{side}_roller'] + positions[f'{side}_jaw'] - positions[f'{side}_shaft']
            for side in ('left', 'right')] + [positions['left_shaft'] + positions['right_shaft']]


@pytest.mark.parametrize('scenario,seconds', [('claw_cycle', 6.), ('pickup_attempt', 8.)])
def test_powered_claw_preserves_transmission_and_fixed_mount(scenario, seconds):
    pieces = scenario == 'pickup_attempt'
    model, data, metadata = _model(not pieces, pieces)
    if pieces:
        _pose(model, data, 'pin_1_15', [.27, 0., .083])
        _pose(model, data, 'cup_1_16', [.48, 0., .083])
        mujoco.mj_forward(model, data)
    control = RobotControl(model, metadata)
    maxima_rad = 0.
    rollers, heights = [], []
    relative = data.body('robot_lift_end').xmat.reshape(3, 3).T @ data.body('robot_claw_mount').xmat.reshape(3, 3)
    assert model.body('robot_claw_mount').jntnum[0] == 0
    np.testing.assert_allclose(relative, np.eye(3), atol=1e-12, rtol=0.)
    assert metadata['part_bodies']['1/463'] == 'robot_lift_end'
    assert model.geom('robot_visual_1_463').bodyid[0] == model.body('robot_lift_end').id
    for _ in range(round(seconds / model.opt.timestep)):
        control.update(data, **_commands(scenario, data.time))
        mujoco.mj_step(model, data)
        maxima_rad = max(maxima_rad, *map(abs, transmission_errors(data)))
        rollers.append([float(data.joint(f'robot_claw_{side}_roller').qpos[0]) for side in ('left', 'right')])
        rotation = data.body('robot_lift_end').xmat.reshape(3, 3).T @ data.body('robot_claw_mount').xmat.reshape(3, 3)
        np.testing.assert_allclose(rotation, relative, atol=1e-12, rtol=0.)
        heights.append(float(data.body('robot_lift_end').xpos[2]))
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    assert not any(warning.number for warning in data.warning)
    assert maxima_rad < .004
    assert np.all(np.ptp(rollers, axis=0) > 10.)
    assert np.ptp(heights) > .1
    for side in ('left', 'right'):
        jaw = model.joint(f'robot_claw_{side}_jaw')
        roller = model.joint(f'robot_claw_{side}_roller')
        assert model.dof_armature[jaw.dofadr[0]] == 0.
        assert model.dof_armature[roller.dofadr[0]] == 0.
        assert not np.any(model.actuator_trnid[:, 0] == jaw.id)


def test_intake_hold_recovers_position_after_external_load():
    from override_sim.robot import build_robot_scene
    xml, metadata = build_robot_scene(fixed=True)
    final_angles = {}
    for mode in ('brake', 'hold'):
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        control = RobotControl(model, metadata, intake_brake=mode)
        joint = model.joint('robot_claw_left_shaft')
        for step in range(2000):
            data.qfrc_applied[joint.dofadr[0]] = -.2 if 500 < step < 1000 else 0.
            control.update(data)
            mujoco.mj_step(model, data)
            assert abs(data.actuator_force[control.intake_actuator]) <= control.intake.stall
        final_angles[mode] = abs(float(data.joint('robot_claw_left_shaft').qpos[0]))
        assert not any(warning.number for warning in data.warning)
    assert final_angles['hold'] < .06
    assert final_angles['brake'] > .2


@pytest.mark.parametrize('timestep', [.001, .0002])
def test_passive_cad_pawls_freewheel_intake_and_open_jaws_on_outtake(timestep):
    results = {}
    for command in (-.1, .5):
        model, data, metadata = _model(True, False)
        model.opt.timestep = timestep
        control = RobotControl(model, metadata, intake_brake='hold')
        contacts = 0
        locked_angles = []
        for _ in range(round(2. / model.opt.timestep)):
            control.update(data, intake=command if data.time < 1 - timestep / 2 else 0.)
            mujoco.mj_step(model, data)
            contacts += sum('robot_ratchet' in (model.geom(c.geom1).name or '')
                            and 'robot_ratchet' in (model.geom(c.geom2).name or '') for c in data.contact)
            if data.time > 1.3:
                locked_angles.append([float(data.joint(f'robot_claw_{side}_shaft').qpos[0]
                                            - data.joint(f'robot_claw_{side}_jaw').qpos[0])
                                      for side in ['left', 'right']])
        assert np.all(np.ptp(locked_angles, axis=0) < 2 * np.pi / 24 / 10)
        assert contacts > 0
        assert not any(w.number for w in data.warning)
        results[command] = [float(data.joint('robot_claw_left_shaft').qpos[0]),
                            float(data.joint('robot_claw_left_jaw').qpos[0]),
                            float(data.joint('robot_claw_right_jaw').qpos[0])]
        for side, node in [('left', '112'), ('right', '110')]:
            name = f'robot_claw_{side}_pawl'
            assert metadata['part_bodies'][f'1/468/{node}'] == name
            assert not np.any(model.actuator_trnid[:, 0] == model.joint(name).id)
            assert model.dof_armature[model.joint(name).dofadr[0]] == 0
    assert results[-.1][0] < -1.8
    assert -.14 < results[-.1][1] < -.1
    assert .1 < results[-.1][2] < .14
    assert results[.5][1] > .3
    assert results[.5][2] < -.2


def test_proposed_closure_centers_jaws_with_comparable_closing_torque():
    from override_sim.robot import build_robot_scene
    from override_sim.robot_claw import ClawClosure
    for closure in [ClawClosure(), ClawClosure(0, 0)]:
        xml, metadata = build_robot_scene(fixed=True, claw_closure=closure)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for common in [-.2, 0., .2]:
            for side, opening_sign in [('left', 1), ('right', -1)]:
                data.joint(f'robot_claw_{side}_jaw').qpos[0] = opening_sign * .28 + common
            mujoco.mj_forward(model, data)
            torques = [float(data.qfrc_spring[model.joint(f'robot_claw_{side}_jaw').dofadr[0]])
                       for side in ['left', 'right']]
            assert sum(torques) == pytest.approx(-2 * closure.stiffness_nm_per_rad * common)
            if closure.stiffness_nm_per_rad:
                assert torques[0] < 0 < torques[1]
            else:
                assert torques == [0., 0.]
        assert metadata['claw']['closure_springs']['calibrated'] is False
        assert metadata['claw']['closure_springs']['source_cad_hardware'] is False
