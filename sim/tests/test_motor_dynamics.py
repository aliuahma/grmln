"""The constrained intake must track motor speed and hold at the field timestep."""
import mujoco
import numpy as np
import pytest

from override_sim.robot_control import RobotControl
from override_sim.robot_verification import _model


def unloaded_transmission():
    model, data, metadata = _model(True, False)
    # Pawl stops intentionally stall one direction; isolate the motor/transmission.
    ratchet_geoms = (model.geom_contype == 64) | (model.geom_contype == 128)
    model.geom_contype[ratchet_geoms] = 0
    model.geom_conaffinity[ratchet_geoms] = 0
    for side in ['left', 'right']:
        model.jnt_stiffness[model.joint(f'robot_claw_{side}_jaw').id] = 0.
    return model, data, metadata


@pytest.mark.parametrize('rpm', [100, 200, 600])
@pytest.mark.parametrize('command', [-.1, .1])
def test_intake_speed_and_hold_converge_across_timesteps(rpm, command):
    results = []
    for timestep in ((.0002, .0001) if rpm == 100 else (.001, .0002)):
        model, data, metadata = unloaded_transmission()
        model.opt.timestep = timestep
        control = RobotControl(model, metadata, intake_rpm=rpm, intake_brake='hold')
        assert model.opt.timestep == timestep
        joint = control.intake_joint
        samples = []
        moving_angle = None
        for _ in range(round(2 / timestep)):
            moving = data.time < 1 - timestep / 2
            previous_velocity = float(data.qvel[joint.dofadr[0]])
            control.update(data, intake=command if moving else 0.)
            mujoco.mj_step(model, data)
            torque = float(data.actuator_force[control.intake_actuator])
            velocity = float(data.qvel[joint.dofadr[0]])
            assert abs(torque) <= control.intake.stall + 1e-10
            assert abs(torque * previous_velocity) <= control.intake.watts + 1e-10
            assert torque * (previous_velocity + velocity) / 2 <= control.intake.watts + 1e-6
            if moving:
                moving_angle = float(data.qpos[joint.qposadr[0]])
            if data.time > 1.8:
                samples.append(velocity)
        damping = sum(model.dof_damping[model.joint(f'robot_claw_{side}_{part}').dofadr[0]]
                      for side in ['left', 'right'] for part in ['shaft', 'roller'])
        slope = control.intake.stall / control.intake.speed
        target_angle = command * control.intake.speed * slope / (slope + damping)
        assert moving_angle == pytest.approx(target_angle, rel=.04, abs=.04)
        assert abs(np.mean(samples)) < .002
        assert np.ptp(samples) < .002
        assert abs(float(data.qpos[joint.qposadr[0]]) - moving_angle) < .01
        assert not any(w.number for w in data.warning)
        results.append(float(data.qpos[joint.qposadr[0]]))
    assert results[0] == pytest.approx(results[1], abs=.004)


@pytest.mark.parametrize('rpm', [100, 200, 600])
def test_full_intake_start_reversal_and_hold_obey_motor_envelope(rpm):
    results = []
    for timestep in ((.0002, .0001) if rpm == 100 else (.001, .0002)):
        model, data, metadata = unloaded_transmission()
        model.opt.timestep = timestep
        control = RobotControl(model, metadata, intake_rpm=rpm, intake_brake='hold')
        joint = control.intake_joint
        total_travel = 0.
        hold_position = None
        final_velocities = []
        for _ in range(round(2 / timestep)):
            time = data.time
            command = -1. if time < .5 - timestep / 2 else (1. if time < 1 - timestep / 2 else 0.)
            before = float(data.qpos[joint.qposadr[0]])
            previous_velocity = float(data.qvel[joint.dofadr[0]])
            control.update(data, intake=command)
            mujoco.mj_step(model, data)
            position = float(data.qpos[joint.qposadr[0]])
            velocity = float(data.qvel[joint.dofadr[0]])
            torque = float(data.actuator_force[control.intake_actuator])
            total_travel += abs(position - before)
            assert abs(torque) <= control.intake.stall + 1e-10
            assert torque * (previous_velocity + velocity) / 2 <= control.intake.watts + 1e-6
            assert abs(velocity) <= control.intake.speed + .01
            if command == 0 and hold_position is None:
                hold_position = before
            if data.time > 1.95:
                final_velocities.append(velocity)
        assert abs(position - hold_position) < .01
        assert abs(final_velocities[-1]) < .01
        assert np.ptp(final_velocities) < .003
        assert not any(w.number for w in data.warning)
        results.append((position, total_travel))
    assert abs(results[0][0] - results[1][0]) < .02 * results[1][1]


def test_loaded_hold_torque_matches_momentum_and_respects_stall():
    from override_sim.robot_control import Motor
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001" gravity="0 0 0" integrator="discrete"/>
      <worldbody><body><joint name="shaft" type="hinge"/><inertial pos="0 0 0" mass="1" diaginertia=".0001 .0001 .0001"/>
      </body></worldbody><actuator><motor joint="shaft" forcelimited="true" forcerange="-1.05 1.05"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    control = RobotControl.__new__(RobotControl)
    control.model = model
    motor = Motor(200)
    for _ in range(10):
        velocity = float(data.qvel[0])
        control._motor(data, 0, motor, 0., velocity)
        data.qfrc_applied[0] = 2.
        mujoco.mj_step(model, data)
        torque_from_momentum = .0001 * (data.qvel[0] - velocity) / model.opt.timestep - 2.
        assert torque_from_momentum == pytest.approx(data.actuator_force[0], abs=1e-10)
        assert abs(torque_from_momentum) <= motor.stall + 1e-10


@pytest.mark.parametrize('inertia', [.000001, .0001, .01])
@pytest.mark.parametrize('rpm', [100, 200, 600])
def test_motor_supplied_step_energy_respects_power(inertia, rpm):
    from override_sim.robot_control import Motor
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep=".001" gravity="0 0 0"/>
      <worldbody><body><joint name="shaft" type="hinge"/><inertial pos="0 0 0" mass="1" diaginertia="{inertia} {inertia} {inertia}"/>
      </body></worldbody><actuator><motor joint="shaft" forcelimited="true" forcerange="-3 3"/></actuator></mujoco>''')
    control = RobotControl.__new__(RobotControl)
    control.model = model
    motor = Motor(rpm)
    for velocity in (-motor.speed / 2, 0., motor.speed / 2):
        for target in (-motor.speed, 0., motor.speed):
            data = mujoco.MjData(model)
            data.qvel[0] = velocity
            mujoco.mj_forward(model, data)
            control._motor(data, 0, motor, target, velocity)
            mujoco.mj_step(model, data)
            supplied = .5 * inertia * (data.qvel[0] ** 2 - velocity ** 2)
            assert supplied <= motor.watts * model.opt.timestep + 1e-12
            assert abs(data.actuator_force[0]) <= motor.stall + 1e-12
