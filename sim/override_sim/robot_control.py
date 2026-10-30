"""Explicit, torque-bounded V5 DC motor approximations and shaft position hold."""
from dataclasses import dataclass
import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class Motor:
    rpm: int
    watts: float = 11.

    def __post_init__(self):
        if self.watts not in (5.5, 11.) or self.rpm not in (100, 200, 600):
            raise ValueError('V5 motor requires 11 W at100/200/600 RPM or5.5 W at200 RPM')
        if self.watts == 5.5 and self.rpm != 200:
            raise ValueError('5.5 W motor has fixed200 RPM gearing')

    @property
    def speed(self):
        return self.rpm * 2 * math.pi / 60

    @property
    def stall(self):
        return .5 if self.watts == 5.5 else 2.1 * 100 / self.rpm

    def command(self, target_speed, actual_speed, count=1, reduction=1.):
        limit = count * self.stall * reduction
        speed_limit = self.speed / reduction
        target_speed = np.clip(target_speed, -speed_limit, speed_limit)
        gain = limit / speed_limit
        demand = gain * (target_speed - actual_speed)
        available = limit * max(0., 1 - abs(actual_speed) / speed_limit)
        if demand * actual_speed < 0:
            available = limit
        power_limit = count * self.watts / max(abs(actual_speed), .1)
        return float(target_speed), gain, min(available, power_limit)

    def torque(self, target_speed, actual_speed, count=1, reduction=1.):
        target, gain, limit = self.command(target_speed, actual_speed, count, reduction)
        return float(np.clip(gain * (target - actual_speed), -limit, limit))


class RobotControl:
    def __init__(self, model, metadata, drive_rpm=600, lift_rpm=100, intake_rpm=200, intake_brake='brake'):
        if intake_brake not in ('coast', 'brake', 'hold'):
            raise ValueError('Intake brake mode must be coast, brake or hold')
        self.model = model
        self.metadata = metadata
        self.drive = Motor(drive_rpm)
        self.lift = Motor(lift_rpm)
        self.intake = Motor(intake_rpm)
        self.intake_brake = intake_brake
        self.intake_hold_position = None
        self.wheels = [(model.actuator(w['actuator']).id, model.joint(w['joint']).dofadr[0], w['side'])
                       for w in metadata['wheels']]
        self.lift_actuator = model.actuator('robot_lift_drive').id
        self.lift_joint = model.joint('robot_lift_crank')
        self.intake_actuator = model.actuator('robot_intake_drive').id
        self.intake_joint = model.joint('robot_claw_left_shaft')
        if intake_rpm == 100:
            # The low-speed cartridge needs this resolution for the light shaft transmission.
            model.opt.timestep = min(model.opt.timestep, .0002)
        for actuator in [*(a for a, _, _ in self.wheels), self.lift_actuator, self.intake_actuator]:
            model.actuator_biastype[actuator] = mujoco.mjtBias.mjBIAS_NONE
            model.actuator_gainprm[actuator, 0] = 1.
            model.actuator_ctrllimited[actuator] = False
            model.actuator_forcelimited[actuator] = True

    def _motor(self, data, actuator, motor, target, velocity, count=1, reduction=1.):
        if not np.any(data.M):
            mujoco.mj_forward(self.model, data)
        if not hasattr(self, '_inertia_buffers'):
            self._inertia_buffers = (np.zeros((1, self.model.nv)), np.zeros((1, self.model.nv)))
        inverse, impulse = self._inertia_buffers
        impulse.fill(0.)
        joint = self.model.actuator_trnid[actuator, 0]
        dof = self.model.jnt_dofadr[joint]
        impulse[0, dof] = 1.
        mujoco.mj_solveM(self.model, data, inverse, impulse)
        inverse_inertia = inverse[0, dof]
        if not hasattr(self, '_constraint_jacobian'):
            self._project_mass(data)
        jacobian = self._constraint_jacobian
        if len(jacobian):
            response = jacobian @ inverse[0]
            inverse_inertia -= response @ self._constraint_mass @ response
        target, gain, limit = motor.command(target, velocity, count, reduction)
        # Bound work supplied over one explicit step using the physical inverse mass.
        # Project the rigid mechanism equalities out of the physical inverse mass.
        # Other constraint reactions can reduce acceleration; no inertia is added.
        power = count * motor.watts
        energy_limit = 2 * power / (abs(velocity) + math.sqrt(
            velocity * velocity + 2 * self.model.opt.timestep * inverse_inertia * power))
        limit = min(limit, energy_limit)
        self.model.actuator_forcerange[actuator] = [-limit, limit]
        data.ctrl[actuator] = float(np.clip(gain * (target - velocity), -limit, limit))

    def _project_mass(self, data):
        rows = []
        for row in np.flatnonzero(data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY):
            if mujoco.mj_isSparse(self.model):
                start = data.efc_J_rowadr[row]
                count = data.efc_J_rownnz[row]
                values = np.zeros(self.model.nv)
                values[data.efc_J_colind[start:start + count]] = data.efc_J[start:start + count]
            else:
                values = data.efc_J.reshape(data.nefc, self.model.nv)[row].copy()
            rows.append(values)
        self._constraint_jacobian = np.asarray(rows).reshape(-1, self.model.nv)
        if rows:
            response = np.zeros_like(self._constraint_jacobian)
            mujoco.mj_solveM(self.model, data, response, self._constraint_jacobian)
            self._constraint_mass = np.linalg.pinv(self._constraint_jacobian @ response.T, rcond=1e-10)

    def update(self, data, *, forward=0., turn=0., lift=0., intake=0.):
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)
        mujoco.mj_crb(self.model, data)
        mujoco.mj_factorM(self.model, data)
        mujoco.mj_makeConstraint(self.model, data)
        self._project_mass(data)
        for actuator, dof, side in self.wheels:
            velocity = (forward + (1 if side == 'right' else -1) * turn * .3127375 / 2) / .034925
            self._motor(data, actuator, self.drive, velocity, data.qvel[dof], count=2, reduction=32 / 24)
        q = data.qpos[self.lift_joint.qposadr[0]]
        dq = data.qvel[self.lift_joint.dofadr[0]]
        self._motor(data, self.lift_actuator, self.lift, 8 * (lift - q), dq, count=2, reduction=7)
        intake_speed = float(np.clip(intake, -1, 1)) * self.intake.speed
        if intake != 0:
            self.intake_hold_position = None
        elif self.intake_brake == 'hold':
            position = float(data.qpos[self.intake_joint.qposadr[0]])
            if self.intake_hold_position is None:
                self.intake_hold_position = position
            intake_speed = 80 * (self.intake_hold_position - position)
        self._motor(data, self.intake_actuator, self.intake, intake_speed,
                    data.qvel[self.intake_joint.dofadr[0]])
        if intake == 0 and self.intake_brake == 'coast':
            self.model.actuator_forcerange[self.intake_actuator] = [0., 0.]
