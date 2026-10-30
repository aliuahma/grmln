"""Verify the imported linkage by its constrained motion and moving CAD parts."""
import math

import mujoco
import numpy as np
import pytest

from override_sim.robot import CAD_ROTATION, build_robot_scene
from override_sim.robot_control import RobotControl


@pytest.fixture(scope='module')
def robot_lift():
    xml, metadata = build_robot_scene(fixed=True)
    model = mujoco.MjModel.from_xml_string(xml)
    return model, metadata


def equality_rows(model, data, kind):
    equality = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    ids = data.efc_id[equality]
    selected = np.array([
        model.equality(int(index)).name.startswith('robot_lift_') and model.eq_type[index] == kind
        for index in ids
    ])
    return np.flatnonzero(equality)[selected]


def test_lift_has_one_degree_of_freedom_and_closed_reference_geometry(robot_lift):
    model, metadata = robot_lift
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert np.allclose(CAD_ROTATION.T @ CAD_ROTATION, np.eye(3))
    assert np.linalg.det(CAD_ROTATION) == pytest.approx(1.)
    connect_rows = equality_rows(model, data, mujoco.mjtEq.mjEQ_CONNECT)
    gear_rows = equality_rows(model, data, mujoco.mjtEq.mjEQ_JOINT)
    assert len(connect_rows) == 15
    assert len(gear_rows) == 1
    assert np.max(np.abs(data.efc_pos[connect_rows])) < 1e-8
    assert np.max(np.abs(data.efc_pos[gear_rows])) < 1e-10
    rows = np.r_[connect_rows, gear_rows]
    if mujoco.mj_isSparse(model):
        jacobian = np.zeros((len(rows), model.nv))
        for output_row, row in enumerate(rows):
            start = data.efc_J_rowadr[row]
            end = start + data.efc_J_rownnz[row]
            jacobian[output_row, data.efc_J_colind[start:end]] = data.efc_J[start:end]
    else:
        jacobian = data.efc_J.reshape(data.nefc, model.nv)[rows]
    dofs = [model.joint(body['name']).dofadr[0] for body in metadata['lift']['bodies']]
    assert len(dofs) - np.linalg.matrix_rank(jacobian[:, dofs], tol=1e-8) == 1


@pytest.fixture(scope='module')
def raised_lift(robot_lift):
    model, metadata = robot_lift
    data = mujoco.MjData(model)
    control = RobotControl(model, metadata)
    mujoco.mj_forward(model, data)
    initial_height = data.body('robot_lift_end').xpos[2]
    initial_aligner_rotation = data.geom('robot_visual_1_471').xmat.copy().reshape(3, 3)
    max_point_error_m = 0.
    max_gear_error_rad = 0.
    for _ in range(math.ceil(8. / model.opt.timestep)):
        target = 1.9 * np.clip((data.time - .5) / 4., 0., 1.)
        control.update(data, lift=target)
        mujoco.mj_step(model, data)
        point_rows = equality_rows(model, data, mujoco.mjtEq.mjEQ_CONNECT)
        gear_rows = equality_rows(model, data, mujoco.mjtEq.mjEQ_JOINT)
        max_point_error_m = max(max_point_error_m, float(np.max(np.abs(data.efc_pos[point_rows]))))
        max_gear_error_rad = max(max_gear_error_rad, float(np.max(np.abs(data.efc_pos[gear_rows]))))
    mujoco.mj_forward(model, data)
    return data, initial_height, initial_aligner_rotation, max_point_error_m, max_gear_error_rad


def test_crank_raises_full_robot_without_breaking_linkage(robot_lift, raised_lift):
    model, _ = robot_lift
    data, initial_height, _, point_error_m, gear_error_rad = raised_lift
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    assert not any(warning.number for warning in data.warning)
    assert model.opt.gravity[2] < -9.
    assert point_error_m < .0002
    assert gear_error_rad < .003
    assert data.body('robot_lift_end').xpos[2] > initial_height + .85
    assert data.joint('robot_lift_crank').qpos[0] == pytest.approx(1.9, abs=.06)
    assert np.allclose(data.body('robot_lift_end').xmat.reshape(3, 3), np.eye(3), atol=.01)
    q = data.joint('robot_lift_lower_c').qpos[0]
    c, s = math.cos(q), math.sin(q)
    lower = np.array([.1154434332 - .358775, -.1044425698 - .05715])
    upper = np.array([.114304878 - .356743433, .0584867311 + .1044425698])
    expected_height = .001 + .358775 + (c * lower[0] - s * lower[1]) + .2413 + (c * upper[0] + s * upper[1])
    assert data.body('robot_lift_end').xpos[2] == pytest.approx(expected_height, abs=.002)


def test_aligner_rotates_with_upper_follower_and_mount_hardware_moves(robot_lift, raised_lift):
    model, metadata = robot_lift
    data, _, initial_rotation, _, _ = raised_lift
    owner = model.body('robot_lift_upper_second').id
    for name in ('robot_visual_1_471', 'robot_visual_1_438_410',
                 'robot_visual_1_438_396', 'robot_visual_1_438_397'):
        assert model.geom(name).bodyid[0] == owner
    relative = data.geom('robot_visual_1_471').xmat.reshape(3, 3) @ initial_rotation.T
    rotation = math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    assert rotation > 1.7
    assert rotation == pytest.approx(data.joint('robot_lift_lower_c').qpos[0], abs=.01)
    assert all(owner != 'robot' for node, owner in metadata['part_bodies'].items() if node.startswith('1/438/'))
    for index in list(range(440, 463)) + list(range(464, 468)):
        assert metadata['part_bodies'][f'1/{index}'] == 'robot_lift_end'
    for index in (401, 403, 404, 405, 406, 407):
        assert metadata['part_bodies'][f'1/{index}'] in ('robot_lift_crank', 'robot_lift_rod')
    for index in (408, 409):
        assert metadata['part_bodies'][f'1/{index}'] == 'robot_lift_lower_c'
