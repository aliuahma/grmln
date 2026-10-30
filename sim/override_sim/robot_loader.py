"""Contact-only loader retrieval trial, with a gate before carrying toward a goal."""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .field import Layout, numbers
from .robot import build_robot_scene
from .robot_control import RobotControl
from .scene import check_state, frame


PIECES = ('cup_1_16', 'pin_1_15')
LOADER = np.array([-1.7342, -1.4926])
GOAL = np.array([-1.196117068, -.598058534])
PLACE_YAW = math.pi


def loader_scene(visual_mode='simple'):
    """Load the flipped assembly, pin below cup, before time zero."""
    kwargs = {'visual_mode': visual_mode}
    xml, metadata = build_robot_scene(Layout.MATCH, start=(-1.05, LOADER[1], math.pi), **kwargs)
    root = ET.fromstring(xml)
    for name, height in zip(PIECES, (.555, .46)):
        body = root.find(f"worldbody/body[@name='{name}']")
        body.set('pos', numbers([*LOADER, height]))
        flipped = np.zeros(4)
        mujoco.mju_mulQuat(flipped, np.array([0., 1., 0., 0.]),
                          np.fromstring(body.get('quat'), sep=' '))
        body.set('quat', numbers(flipped))
    return ET.tostring(root, encoding='unicode'), metadata


def placement_status(model, data, base_geoms):
    """Strict insertion: pin in the goal, cup over pin, clean loaded contacts."""
    cup, pin = [data.body(name).xpos.copy() for name in PIECES]
    cup_axis, pin_axis = [data.body(name).xmat.reshape(3, 3)[:, 1] for name in PIECES]
    normal = {'pin_goal': 0., 'cup_pin': 0., 'cup_goal': 0., 'floor': 0.}
    depth = {'pin_goal': 0., 'cup_pin': 0.}
    robot_contact = False
    wrench = np.zeros(6)
    for index, contact in enumerate(data.contact):
        geoms = (int(contact.geom1), int(contact.geom2))
        names = [model.body(int(model.geom_bodyid[g])).name for g in geoms]
        if not any(name in PIECES for name in names):
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        force = max(0., float(wrench[0]))
        if any(name.startswith('robot') for name in names) and force > .01:
            robot_contact = True
        relation = None
        if PIECES[1] in names and any(g in base_geoms for g in geoms):
            relation = 'pin_goal'
        elif all(name in names for name in PIECES):
            relation = 'cup_pin'
        elif PIECES[0] in names and any(g in base_geoms for g in geoms):
            relation = 'cup_goal'
        elif any(model.geom(g).name == 'floor' for g in geoms):
            relation = 'floor'
        if relation is not None:
            normal[relation] += force
            if relation in depth and force > .01:
                depth[relation] = max(depth[relation], -float(contact.dist))
    stationary = True
    for name in PIECES:
        body = model.body(name)
        dof = int(model.jnt_dofadr[int(body.jntadr[0])])
        stationary &= (np.linalg.norm(data.qvel[dof:dof+3]) < .02
                       and np.linalg.norm(data.qvel[dof+3:dof+6]) < .2)
    # CAD low-goal mouth and nominal pin flange seat. Each complete pin is 165 mm long.
    mouth_z, pin_half_length, cup_half_length = .08253265, .0825, .08225
    pin_lower_tip = pin-pin_half_length*pin_axis
    pin_upper_tip = pin+pin_half_length*pin_axis
    cup_lower_mouth = cup-cup_half_length*cup_axis
    pin_depth_in_cup = float((pin_upper_tip-cup_lower_mouth) @ cup_axis)
    axis_spacing = float((cup-pin) @ pin_axis)
    radial_offset = np.linalg.norm((cup-pin)-axis_spacing*pin_axis)
    pin_seated = (np.linalg.norm(pin[:2]-GOAL) < .005 and abs(pin[2]-.0845) < .001
                  and pin_lower_tip[2] < mouth_z-.075 and pin_lower_tip[2] > .0005)
    cup_nested = (.083 < cup[2]-pin[2] < .089 and .083 < axis_spacing < .089
                  and radial_offset < .005
                  and .075 < pin_depth_in_cup < .085
                  and float(cup_axis @ pin_axis) > math.cos(math.radians(3)))
    upright = min(cup_axis[2], pin_axis[2]) > math.cos(math.radians(5))
    # Gravity-supported seating has ~1.7 N goal load and micron-scale penetration.
    # Opposed lip trapping can look seated while generating >100 kN: reject it.
    clean = (normal['pin_goal'] < 10. and normal['cup_pin'] < 10.
             and max(depth.values()) < .0001 and normal['cup_goal'] < .01
             and normal['floor'] < .01)
    robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
    robot_clear = not robot_contact and (robot_id < 0 or
        np.linalg.norm(data.body(robot_id).xpos[:2]-GOAL) > .38)
    return {'pin_contacts_goal': normal['pin_goal'] > .01,
            'cup_contacts_pin': normal['cup_pin'] > .01,
            'contacts_clean': bool(clean), 'pin_seated_in_goal': bool(pin_seated),
            'cup_nested_over_pin': bool(cup_nested), 'upright': bool(upright),
            'stationary': bool(stationary), 'robot_clear': bool(robot_clear)}


def tire_grip_forces(model, data):
    """Normal tire loads in N, rows cup/pin and columns left/right."""
    forces = np.zeros((2, 2))
    wrench = np.zeros(6)
    for index, contact in enumerate(data.contact):
        for tire, piece in ((int(contact.geom1), int(contact.geom2)),
                            (int(contact.geom2), int(contact.geom1))):
            if not model.geom(tire).name.startswith('robot_intake_tire_'):
                continue
            piece_name = model.body(int(model.geom_bodyid[piece])).name
            if piece_name not in PIECES:
                continue
            side_name = model.body(int(model.geom_bodyid[tire])).name
            side = 0 if 'left' in side_name else 1
            mujoco.mj_contactForce(model, data, index, wrench)
            forces[PIECES.index(piece_name), side] += max(0., wrench[0])
    return forces


def qualified_tire_grip(forces):
    """Require tire load on each piece and opposing jaw sides.

    Shared-seam contact may put the cup and pin on different sides. This only
    classifies the contacts; pose and sustained retention remain separate gates.
    """
    forces = np.asarray(forces)
    return bool(np.all(np.sum(forces, axis=0) > .01)
                and np.all(np.sum(forces, axis=1) > .01))


def qualified_stack_pose(positions, axes):
    """Correct-order pair, upright and coaxial independently of tire loading."""
    cup, pin = np.asarray(positions)
    cup_axis, pin_axis = np.asarray(axes)
    offset = cup-pin
    spacing = float(offset @ pin_axis)
    return bool(min(cup_axis[2], pin_axis[2]) > math.cos(math.radians(5))
                and float(cup_axis @ pin_axis) > math.cos(math.radians(3))
                and .083 < spacing < .089
                and np.linalg.norm(offset-spacing*pin_axis) < .005)


def run_loader_transfer(output, video=True, *, visual_mode='simple', intake=-.05, seconds=120.):
    """Attempt retrieval; carry/place only after both objects pass retention checks.

    Only motor torque and the existing human-operated loader actuator change after
    initialization. Objects retain their free joints, geometry, masses and contacts.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    xml, metadata = loader_scene(visual_mode)
    (output / 'loader.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    control = RobotControl(model, metadata, intake_brake='hold')
    renderer = mujoco.Renderer(model, 720, 960) if video else None
    writer = imageio.get_writer(output / 'loader.mp4', fps=30, codec='libx264', macro_block_size=1) if video else None
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.45, 285, -28
    phase, entered = 'load pin and cup', 0.
    phases = [{'time_s': 0., 'phase': phase}]
    records, collisions = [], set()
    next_control = next_frame = next_progress = 0.
    forward = turn = lift = roller = 0.
    captured = False
    capture_tire_forces = np.zeros((2, 2))
    max_heights = np.zeros(2)
    loader_travel = 0.
    outcome = 'incomplete'
    failure = None
    capture_origin = None
    loader_command = 0.
    raised_retreat_distance = 0.
    carry_offset = np.array([.19, 0., 0.])
    waypoints = [np.array(p) for p in [(-.90, -1.45), (-.90, -.9), (GOAL[0]+.35, GOAL[1])]]
    waypoint = 0
    placement_samples = []
    final_placement = {}
    support_since = None
    release_clear_since = None
    release_origin = None
    carry_loss_since = None
    supported_lift = None
    descent_target = .7
    base_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == 0 and model.geom_group[i] == 3
                  and np.linalg.norm(data.geom_xpos[i, :2]-GOAL) < .10}
    geom_bodies = tuple(model.body(int(body)).name for body in model.geom_bodyid)
    robot_geoms = np.array([name.startswith('robot') for name in geom_bodies])
    tracked_geoms = np.array([name.startswith('loader_') or name in PIECES for name in geom_bodies])
    turning = False
    translating = False
    previous_carry_turn = 0.
    pickup_lift = .094
    minimum_pocket_depth = .015
    grip_since = extraction_origin = extraction_loss_since = None
    extraction_forward = None
    pickup_row_height = None
    draw_depth_reached = False
    withdrawn_since = None

    def measured_grip():
        robot = data.body('robot')
        rotation = robot.xmat.reshape(3, 3)
        pieces = np.array([data.body(name).xpos.copy() for name in PIECES])
        axes = [data.body(name).xmat.reshape(3, 3)[:, 1] for name in PIECES]
        # A mean could conceal a piece still ahead of one asymmetric column.
        column = min((rotation.T @ (data.geom(f'robot_intake_tire_{number}').xpos-robot.xpos))[0]
                     for number in (172, 163))
        # Positive depth means the complete pair is past the rollers, inside the claw.
        depths = column-(pieces-robot.xpos) @ rotation[:, 0]
        loaded = qualified_tire_grip(tire_grip_forces(model, data))
        return pieces, depths, loaded and qualified_stack_pose(pieces, axes)

    def loader_contacts_clear():
        for contact in data.contact:
            names = [geom_bodies[int(g)] for g in (contact.geom1, contact.geom2)]
            loader_contact = (any(name.startswith('loader_') for name in names)
                              or any(model.geom(int(g)).name.startswith('loader_guide_')
                                     for g in (contact.geom1, contact.geom2)))
            if loader_contact and any(name in PIECES for name in names):
                # Unloaded touching can become loaded on the next lift step.
                return False
        return True

    def heading_control(error, limit):
        nonlocal turning
        if abs(error) < .025:
            turning = False
        elif abs(error) > .05:
            turning = True
        return math.copysign(max(1.2, min(limit, 5*abs(error))), error) if turning else 0.

    def stack_commands(position, rotation):
        nonlocal turning, translating
        pin = data.body(PIECES[1]).xpos
        error = rotation[:2, :2].T @ (GOAL-pin[:2])
        offset = rotation[:2, :2].T @ (pin[:2]-position[:2])
        desired_turn = .8*error[1]/max(.08, offset[0])
        desired_forward = .8*error[0]+desired_turn*offset[1]
        if abs(desired_turn) < .025:
            turning = False
        elif abs(desired_turn) > .05:
            turning = True
        if abs(desired_forward) < .003:
            translating = False
        elif abs(desired_forward) > .006:
            translating = True
        turn_target = math.copysign(max(1.2, min(2., abs(desired_turn))), desired_turn) if turning else 0.
        forward_target = math.copysign(max(.08, min(.12, abs(desired_forward))), desired_forward) if translating else 0.
        return forward_target, turn_target

    try:
        while data.time < seconds:
            if data.time >= next_control:
                max_heights = np.maximum(max_heights, [data.body(name).xpos[2] for name in PIECES])
                loader_travel = max(loader_travel, float(data.joint('loader_0').qpos[0]))
                position = data.body('robot').xpos.copy()
                rotation = data.body('robot').xmat.reshape(3, 3)
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                age = data.time - entered
                if data.time >= next_progress:
                    progress = {'time_s': float(data.time), 'phase': phase, 'waypoint': waypoint,
                                'robot_position_m': position.tolist(),
                                'piece_positions_m': {name: data.body(name).xpos.tolist() for name in PIECES},
                                'claw_mount_height_m': float(data.body('robot_claw_mount').xpos[2]),
                                'lift_crank_rad': float(data.joint('robot_lift_crank').qpos[0])}
                    if phase in ('draw stack into rollers', 'extract stack from loader'):
                        _, depths, _ = measured_grip()
                        progress['inside_claw_depth_m'] = dict(zip(PIECES, depths.tolist()))
                        progress['tire_normal_forces_N'] = tire_grip_forces(model, data).tolist()
                    (output / 'loader-progress.json').write_text(json.dumps(progress, indent=2))
                    next_progress += .5
                forward = turn = roller = 0.
                new_phase = phase
                if phase == 'load pin and cup':
                    pieces = np.array([data.body(name).xpos.copy() for name in PIECES])
                    loading_complete = bool(np.all(pieces[:, 2] < .22)
                                            and np.all(np.linalg.norm(pieces[:, :2]-LOADER, axis=1) < .045))
                    if age > 2. and loading_complete:
                        new_phase = 'raise loader sleeve'
                    elif age > 6.:
                        failure = 'The freely dropped pin/cup pair did not settle inside the lowered loader.'
                        new_phase = 'loading failed'
                elif phase == 'raise loader sleeve':
                    loader_command, lift = .2185, pickup_lift
                    if age > 1. and data.joint('loader_0').qpos[0] > .20:
                        new_phase = 'power open jaws'
                elif phase == 'power open jaws':
                    roller = .5
                    jaw_angles = [abs(data.joint(f'robot_claw_{side}_jaw').qpos[0]) for side in ('left', 'right')]
                    if min(jaw_angles) > .30 or age > .35:
                        new_phase = 'approach loader'
                elif phase == 'approach loader':
                    desired = math.pi-float(np.clip(4*(LOADER[1]-position[1]), -.20, .20))
                    error = math.atan2(math.sin(desired-yaw), math.cos(desired-yaw))
                    turn = float(np.clip(3*error, -1., 1.))
                    forward = .10
                    mouth = (data.body('robot_claw_left_roller').xpos+data.body('robot_claw_right_roller').xpos)/2
                    if np.linalg.norm(mouth[:2]-data.body(PIECES[0]).xpos[:2]) < .035:
                        pickup_row_height = float(np.mean([data.geom(f'robot_intake_tire_{n}').xpos[2]
                                                           for n in (172, 163)]))
                        new_phase = 'draw stack into rollers'
                    elif age > 9.:
                        failure = 'The open roller mouth did not reach the loaded stack.'
                        new_phase = 'retrieval failed'
                elif phase == 'draw stack into rollers':
                    _, depths, loaded = measured_grip()
                    draw_depth_reached |= bool(np.all(depths > minimum_pocket_depth))
                    seated = loaded and np.all(depths > minimum_pocket_depth)
                    # Continuing intake while pitch settles pulls the pair past the capture depth.
                    roller = 0. if draw_depth_reached else intake
                    grip_since = data.time if seated and grip_since is None else grip_since if seated else None
                    if grip_since is not None and data.time-grip_since > .2:
                        extraction_origin = position[:2].copy()
                        extraction_forward = rotation[:2, 0].copy()
                        new_phase = 'extract stack from loader'
                    elif age > 6.:
                        failure = 'Both upright objects did not seat inside the roller columns with loaded tire contacts.'
                        new_phase = 'retrieval failed'
                elif phase == 'extract stack from loader':
                    lift = pickup_lift
                    forward = -min(.08, .04*age)
                    pieces, depths, loaded = measured_grip()
                    loaded &= bool(np.all(depths > minimum_pocket_depth))
                    extraction_loss_since = (None if loaded else
                        data.time if extraction_loss_since is None else extraction_loss_since)
                    withdrawn = ((position[:2]-extraction_origin) @ -extraction_forward > .12
                                 and np.all((pieces[:, :2]-LOADER) @ -extraction_forward > .12)
                                 and loader_contacts_clear())
                    if withdrawn and loaded:
                        forward = 0.
                        withdrawn_since = data.time if withdrawn_since is None else withdrawn_since
                    else:
                        withdrawn_since = None
                    if extraction_loss_since is not None and data.time-extraction_loss_since > .2:
                        failure = 'The pair lost its upright loaded tire grip during back-out; lift was not started.'
                        new_phase = 'retrieval failed'
                    elif (withdrawn_since is not None and data.time-withdrawn_since > .2
                          and np.linalg.norm(data.body('robot').cvel[3:5]) < .02):
                        new_phase = 'attempt lift'
                    elif age > 6.:
                        failure = 'Both pieces did not clear the loader during bounded straight back-out.'
                        new_phase = 'retrieval failed'
                elif phase == 'attempt lift':
                    lift = min(.7, pickup_lift+.18*age)
                    if age > 3.5:
                        capture_origin = position[:2].copy()
                        new_phase = 'retreat and test retention'
                elif phase == 'retreat and test retention':
                    lift, forward = .7, -.12
                    raised_retreat_distance = float(np.linalg.norm(position[:2]-capture_origin))
                    if raised_retreat_distance > .22 or age > 4.:
                        new_phase = 'hold raised stack'
                elif phase == 'hold raised stack':
                    lift = .7
                    if age > 2.5:
                        mount = data.body('robot_claw_mount').xpos
                        pieces = np.array([data.body(name).xpos.copy() for name in PIECES])
                        capture_tire_forces = tire_grip_forces(model, data)
                        _, raised_depths, _ = measured_grip()
                        captured = bool(qualified_tire_grip(capture_tire_forces)
                                        and np.all(raised_depths > minimum_pocket_depth)
                                        and qualified_stack_pose(pieces, [data.body(name).xmat.reshape(3, 3)[:, 1]
                                                                         for name in PIECES])
                                        and raised_retreat_distance > .22
                                        and np.all(np.linalg.norm(pieces[:, :2]-mount[:2], axis=1) < .17)
                                        and np.all(pieces[:, 2] > .22))
                        new_phase = 'turn toward base' if captured else 'retrieval failed'
                        if captured:
                            carry_offset = rotation.T @ (pieces[0]-position)
                        else:
                            failure = 'Both objects did not retain a loaded tire grip after the raised retreat and hold.'
                elif phase in ('turn toward base', 'drive toward base'):
                    lift = .7
                    delta = waypoints[waypoint]-position[:2]
                    desired = math.atan2(delta[1], delta[0])
                    error = math.atan2(math.sin(desired-yaw), math.cos(desired-yaw))
                    turn = heading_control(error, 1.3)
                    if phase == 'turn toward base' and abs(error) < .08:
                        new_phase = 'drive toward base'
                    if phase == 'drive toward base':
                        forward = .08 if abs(error) < .15 else 0.
                        if np.linalg.norm(delta) < .10:
                            waypoint += 1
                            if waypoint == len(waypoints):
                                new_phase = 'align claw with base'
                elif phase == 'align claw with base':
                    lift = .7
                    error = math.atan2(math.sin(PLACE_YAW-yaw), math.cos(PLACE_YAW-yaw))
                    turn = heading_control(error, 2.)
                    if abs(error) < .035:
                        new_phase = 'approach base'
                elif phase == 'approach base':
                    lift = .7
                    forward, turn = stack_commands(position, rotation)
                    if np.linalg.norm(data.body(PIECES[1]).xpos[:2]-GOAL) < .005:
                        new_phase = 'lower onto base'
                elif phase == 'lower onto base':
                    if supported_lift is None and np.linalg.norm(data.body(PIECES[1]).xpos[:2]-GOAL) < .005:
                        descent_target = max(0., descent_target-.18*.01)
                    lift = descent_target
                    forward, turn = stack_commands(position, rotation)
                    support = placement_status(model, data, base_geoms)
                    if all(support[key] for key in ('pin_contacts_goal', 'cup_contacts_pin', 'contacts_clean',
                                                        'pin_seated_in_goal', 'cup_nested_over_pin', 'upright')):
                        if supported_lift is None:
                            supported_lift = float(data.joint('robot_lift_crank').qpos[0])
                            descent_target = lift = supported_lift
                        if support_since is None:
                            support_since = data.time
                        if data.time-support_since > .2:
                            new_phase = 'release'
                    else:
                        support_since = None
                    if age > 10.:
                        failure = 'The descending cup did not make sustained contact with the target base.'
                        new_phase = 'placement failed'
                elif phase == 'release':
                    lift = max(.05, supported_lift-.15*age)
                    jaw_angles = [abs(float(data.joint(f'robot_claw_{side}_jaw').qpos[0]))
                                  for side in ('left', 'right')]
                    roller = .5 if age < 1.3 else 0.
                    if age > 1.8 and min(jaw_angles) > .40:
                        new_phase = 'lift clear of base'
                    elif age > 3.:
                        failure = 'The powered jaws did not open while the supported stack unloaded.'
                        new_phase = 'placement failed'
                elif phase == 'lift clear of base':
                    lift = .12
                    if release_origin is None:
                        release_origin = position[:2].copy()
                    support = placement_status(model, data, base_geoms)
                    pairs = data.contact.geom
                    robot_pair = robot_geoms[pairs[:, 0]] | robot_geoms[pairs[:, 1]]
                    goal_contact = bool(np.any(robot_pair & np.isin(pairs, list(base_geoms)).any(axis=1)))
                    object_contact = bool(np.any(robot_pair &
                        (np.array([geom_bodies[g] in PIECES for g in pairs[:, 0]])
                         | np.array([geom_bodies[g] in PIECES for g in pairs[:, 1]]))))
                    supported = all(support[key] for key in ('pin_contacts_goal', 'cup_contacts_pin',
                                    'contacts_clean', 'pin_seated_in_goal', 'cup_nested_over_pin', 'upright'))
                    backoff = np.linalg.norm(position[:2]-release_origin)
                    if not goal_contact and supported and object_contact and backoff < .025:
                        forward = -.08
                    ready = not goal_contact and not object_contact and supported
                    release_clear_since = (data.time if release_clear_since is None else release_clear_since) if ready else None
                    if release_clear_since is not None and data.time-release_clear_since > .2:
                        new_phase = 'clear base'
                    elif age > 4.:
                        failure = 'The claw did not disengage within the bounded 25 mm withdrawal.'
                        new_phase = 'placement failed'
                elif phase == 'clear base':
                    forward = -.12
                    if age > .5 and placement_status(model, data, base_geoms)['robot_clear']:
                        new_phase = 'settle placement'
                    elif age > 6.:
                        failure = 'The robot did not physically clear the released stack.'
                        new_phase = 'placement failed'
                elif phase == 'settle placement':
                    final_placement = placement_status(model, data, base_geoms)
                    if age > 1.:
                        placement_samples.append(final_placement)
                    if age > 2.:
                        placed = bool(placement_samples and all(all(sample.values()) for sample in placement_samples[-50:]))
                        outcome = 'passed' if placed else 'incomplete'
                        if not placed:
                            failure = 'Released stack did not remain upright, supported by the base and stationary after robot clearance.'
                        new_phase = 'finished'
                if phase in ('turn toward base', 'drive toward base', 'align claw with base', 'approach base'):
                    pieces = np.array([data.body(name).xpos.copy() for name in PIECES])
                    mount = data.body('robot_claw_mount').xpos
                    _, carry_depths, _ = measured_grip()
                    retained = bool(np.all(pieces[:, 2] > .22)
                                    and np.all(carry_depths > minimum_pocket_depth)
                                    and np.all(np.linalg.norm(pieces[:, :2]-mount[:2], axis=1) < .18)
                                    and qualified_stack_pose(pieces, [data.body(name).xmat.reshape(3, 3)[:, 1]
                                                                     for name in PIECES]))
                    carry_loss_since = None if retained else (data.time if carry_loss_since is None else carry_loss_since)
                    if carry_loss_since is not None and data.time-carry_loss_since > .2:
                        failure = 'A carried game piece left the raised claw before placement.'
                        new_phase = 'carry failed'
                        forward = turn = roller = 0.
                if phase in ('turn toward base', 'drive toward base'):
                    if turn * previous_carry_turn <= 0. or abs(turn) > abs(previous_carry_turn):
                        previous = abs(previous_carry_turn) if turn * previous_carry_turn > 0. else 0.
                        turn = math.copysign(min(abs(turn), previous+.65*.01), turn)
                    previous_carry_turn = turn
                else:
                    previous_carry_turn = 0.
                if new_phase in ('loading failed', 'retrieval failed',
                                 'placement failed', 'carry failed'):
                    forward = turn = roller = 0.
                if new_phase != phase:
                    turning = translating = False
                    phase, entered = new_phase, data.time
                    phases.append({'time_s': float(data.time), 'phase': phase})
                    np.savez(output / f'loader-phase-{len(phases):02d}.npz', qpos=data.qpos, qvel=data.qvel)
                records.append([float(data.time), *position, yaw, *data.body(PIECES[0]).xpos,
                                *data.body(PIECES[1]).xpos, float(data.joint('robot_lift_crank').qpos[0])])
                next_control += .01
            control.update(data, forward=forward, turn=turn, lift=lift, intake=roller)
            data.ctrl[model.actuator('loader_0_lift').id] = loader_command
            mujoco.mj_step(model, data)
            check_state(model, data)
            contact_geoms = data.contact.geom
            first, second = contact_geoms[:, 0], contact_geoms[:, 1]
            tracked = ((robot_geoms[first] & tracked_geoms[second])
                       | (robot_geoms[second] & tracked_geoms[first]))
            for pair in contact_geoms[tracked]:
                collisions.add(tuple(model.geom(g).name or geom_bodies[g] for g in pair))
            if writer and data.time >= next_frame:
                if phase in ('lower onto base', 'release', 'lift clear of base', 'clear base', 'settle placement', 'finished'):
                    camera.lookat[:] = [*GOAL, .24]
                    camera.distance = 1.25
                elif captured:
                    camera.lookat[:] = data.body('robot').xpos + [0., 0., .25]
                    camera.distance = 1.7
                else:
                    camera.lookat[:] = [-1.48, LOADER[1], .35 if phase == 'load pin and cup' else .22]
                picture = Image.fromarray(frame(renderer, data, camera))
                draw = ImageDraw.Draw(picture)
                draw.rectangle((10, 10, 950, 96), fill=(15, 20, 30))
                draw.text((22, 18), f'{data.time:05.2f}s | {phase}', fill='white', font=ImageFont.load_default(size=23))
                draw.text((22, 48), 'Contact physics | motor torque + human loader input', fill='#d0d6e0', font=ImageFont.load_default(size=16))
                draw.text((22, 72), 'Proposed centering springs | uncalibrated hardware assumption',
                          fill='#d0d6e0', font=ImageFont.load_default(size=14))
                writer.append_data(np.asarray(picture))
                if not (output / f'loader-{len(phases):02d}.png').exists():
                    picture.save(output / f'loader-{len(phases):02d}.png')
                next_frame += 1/30
            if phase in ('loading failed', 'retrieval failed', 'placement failed', 'carry failed', 'finished') and data.time-entered > 1.:
                break
        final_placement = placement_status(model, data, base_geoms)
        if outcome == 'passed':
            if not all(final_placement.values()):
                outcome = 'incomplete'
                failure = 'The stack did not remain settled through the final clearance view.'
        if failure is None and outcome != 'passed':
            failure = f'Timed out during {phase}.'
        result = {'outcome': outcome, 'failure': failure, 'simulation_seconds': float(data.time),
                  'placement_checks': final_placement, 'placement_stability_samples': len(placement_samples),
                  'carry_capture_offset_robot_m': carry_offset.tolist(),
                  'capture_tire_normal_forces_N': {name: dict(zip(('left', 'right'), capture_tire_forces[i].tolist())) for i, name in enumerate(PIECES)},
                  'claw_closure': metadata['claw']['closure_springs'],
                  'controller_sensing': 'Exact simulated robot and object poses; not a hardware perception model.',
                  'max_piece_heights_m': dict(zip(PIECES, max_heights.tolist())),
                  'loader_lift_travel_m': loader_travel, 'raised_retreat_distance_m': raised_retreat_distance, 'retention_verified': captured, 'placement_verified': outcome == 'passed', 'phases': phases,
                  'piece_positions_m': {name: data.body(name).xpos.tolist() for name in PIECES},
                  'robot_position_m': data.body('robot').xpos.tolist(),
                  'interactions': sorted(collisions), 'warnings': [int(w.number) for w in data.warning],
                  'initial_stack_center_m': LOADER.tolist(), 'initial_cup_pin_origin_heights_m': [.555, .46], 'target_base_center_m': GOAL.tolist(),
                  'intake_command': intake, 'post_initialization_object_state_writes': 0,
                  'pickup_control': {'lift_target_rad': pickup_lift,
                                     'measured_lowest_tire_height_m': pickup_row_height,
                                     'minimum_inside_claw_depth_m': minimum_pocket_depth,
                                     'depth_reference': 'Minimum depth past BOTH physical lowest roller columns.',
                                     'back_out_before_lift': True,
                                     'clearance_from_loader_m': .12},
                  'external_object_forces': False, 'object_welds': False,
                  'limitations': list(dict.fromkeys(metadata['limitations'] + metadata['claw']['limitations']
                                + ['Contact and motor parameters are uncalibrated.']))}
        (output / 'loader.json').write_text(json.dumps(result, indent=2))
        np.savetxt(output / 'loader.csv', records, delimiter=',', comments='',
                   header='time_s,robot_x,robot_y,robot_z,yaw,cup_x,cup_y,cup_z,pin_x,pin_y,pin_z,lift_rad')
        np.savez(output / 'loader-state.npz', qpos=data.qpos, qvel=data.qvel)
        return result
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
