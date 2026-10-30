"""Contact-driven translation and passive arm limits in the assembled robot."""
import math

import mujoco
import numpy as np
import pytest

from override_sim.robot import build_robot_scene
from override_sim.robot_control import Motor, RobotControl


@pytest.mark.parametrize('rpm,watts', [(100, 11), (200, 11), (600, 11), (200, 5.5)])
def test_motor_respects_power_and_torque(rpm, watts):
    motor = Motor(rpm, watts)
    for velocity in np.linspace(-2 * motor.speed, 2 * motor.speed, 101):
        for command in (-3 * motor.speed, 0, 3 * motor.speed):
            torque = motor.torque(command, velocity, count=2, reduction=7)
            assert abs(torque) <= 2 * motor.stall * 7 + 1e-10
            assert abs(torque * velocity) <= 2 * watts + 1e-10
    assert motor.torque(motor.speed, motor.speed) == 0


def test_half_motor_has_fixed_cartridge():
    with pytest.raises(ValueError):
        Motor(600, 5.5)


def test_drive_starts_from_rest_and_passive_arms_obey_lower_stop():
    xml, metadata = build_robot_scene()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    control = RobotControl(model, metadata)
    mujoco.mj_forward(model, data)
    assert np.all(data.qvel == 0)
    origin = data.body('robot').xpos.copy()
    lowest_arm_angle = 0.
    for _ in range(math.ceil(3. / model.opt.timestep)):
        control.update(data, forward=.3 if data.time > 1 else 0.)
        mujoco.mj_step(model, data)
        for arm in metadata['toggle_arms']['arms']:
            lowest_arm_angle = min(lowest_arm_angle, float(data.joint(arm['name']).qpos[0]))
    assert data.body('robot').xpos[0] > origin[0] + .4
    assert abs(data.body('robot').xpos[1] - origin[1]) < .15
    assert lowest_arm_angle > -.01
    assert not metadata['toggle_arms']['actuators']
    assert all(not warning.number for warning in data.warning)
    assert np.isfinite(data.qpos).all()


def test_low_speed_cartridges_configure_their_stall_torque_bounds():
    xml, metadata = build_robot_scene(fixed=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    control = RobotControl(model, metadata, drive_rpm=100, intake_rpm=100)
    control.update(data, forward=1., intake=1.)
    mujoco.mj_forward(model, data)
    assert 0 < data.actuator_force[control.intake_actuator] <= 2.1
    assert control.intake.torque(100., 0.) == pytest.approx(2.1)
    for actuator, _, _ in control.wheels:
        assert 0 < data.actuator_force[actuator] <= 2 * 2.1 * 32 / 24
    assert sum(metadata['body_masses_kg'].values()) == pytest.approx(model.body_mass.sum())
    for wheel in metadata['wheels']:
        for index in range(18):
            joint = model.joint(f'{wheel["joint"]}_roller_{index}')
            assert model.dof_armature[joint.dofadr[0]] == 0


def test_stationary_lift_settles_without_motor_limit_cycle():
    xml, metadata = build_robot_scene()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    control = RobotControl(model, metadata)
    heights, torques = [], []
    for _ in range(round(2. / model.opt.timestep)):
        control.update(data)
        mujoco.mj_step(model, data)
        if data.time > 1.:
            heights.append(data.body('robot_lift_end').xpos[2])
            torques.append(data.actuator_force[control.lift_actuator])
    assert np.ptp(heights) < .0001
    assert np.ptp(torques) < .1
    assert not any(warning.number for warning in data.warning)
