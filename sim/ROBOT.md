# GRMLN KISS V2 robot

The robot is a MuJoCo MJCF model assembled from the supplied STEP geometry. It has a free chassis, four driven omni wheels, a closed-loop lift, a roller claw, and two passive wall-toggle arms. This is a geometry-grounded simulation with estimated physical parameters. It does not run PROS or LemLib.

## Run the verification

```sh
uv sync --project sim --extra test
uv run --project sim override-sim robot-verify --output sim/out/robot
uv run --project sim override-sim robot-verify --lift-rpm 200 --no-video --output sim/out/robot-200
uv run --project sim override-sim robot-sequence --output sim/out/robot/sequence
uv run --project sim override-sim robot-loader --output sim/out/robot/loader
uv run --project sim pytest -q
uv run --project sim python sim/tools/benchmark_robot.py
uv run --project sim python sim/tools/benchmark_robot_visuals.py
```

`robot-verify` exercises the lift, claw and drivetrain, then attempts contact pickup. Its report distinguishes mechanism checks from unverified placement. `robot-sequence` starts at rest on the university match field and drives through pins toward a wall toggle. Its report records actual contact, pin displacement, lateral slip and toggle rotation. An incomplete sequence returns a nonzero exit status. Neither demonstration changes robot poses after initialization or welds game pieces to the claw.

`robot-loader` attempts loader retrieval and placement on an empty base. The intended pickup order is admit both objects past the roller centers into the claw, close around them, back straight out at the pickup height, then lift after clearing the loader. Aligning the object centers with the roller centers does not establish the requested capture depth. It uses exact simulated poses for navigation, bounded motor torque, motor HOLD, passive contacts, and a force-limited loader actuator representing human input. It does not emulate vision or odometry. Actual object retention and loader clearance must be measured before transport. `loader.json` records phase transitions, placement checks, warnings and failure reasons. A failed trial remains a failed trial even if part of its video looks successful.

The current controller requires a provisional 15 mm depth behind both roller columns, an upright nested pair and loaded tire contacts before withdrawal. It stops the intake when both pieces reach that depth, then checks settling. Lifting requires signed withdrawal of the robot and both pieces, no sleeve or guide contacts, and a stopped clearance dwell. The depth target is a verification threshold, not a qualified pickup recipe. Deeper floor experiments can retain both pieces through back-out and lifting when the lower rollers engage beneath the pin shoulder, but the retained pair still leans about 11 degrees. That result does not establish a straight grasp or loader transfer.

For an empty base, the pin must enter the goal and the cup must nest over that pin, as specified by [SC2](https://manuals.vex.com/v5/override/v2.1/game/). The fixed claw requires that assembly order at pickup. Earlier diagnostic runs used a cup below the pin and checked support on the base; those results do not establish valid insertion. The corrected transfer remains under verification.

Placement checks now require the pin at the CAD seating depth, the cup nested above it, loaded support contacts, upright axes, low velocity, and robot clearance. They reject deep penetrations and excessive opposing contact forces. A high-drop diagnostic currently traps the pin below the goal lip despite looking seated; gentle gravity seating passes. This remains a contact-model limitation, not evidence of a real mechanical jam.

Videos and generated XML stay under ignored `sim/out/`. XML files reference packaged mesh assets by absolute path and therefore belong to the generating checkout. Regenerate them after moving the checkout. Control schedules live in the corresponding Python modules; XML alone does not replay those commands.

Navigation commands update every 10 ms, while motor feedback updates every 1 ms physics step. Holding motor feedback for 10 ms caused an artificial lift oscillation. The motor controller applies bounded torque with a conservative linear DC speed-torque slope. Earlier high-gain velocity feedback caused artificial chatter in the low-inertia intake shafts. An implicit velocity servo suppressed that chatter but could exceed the physical torque limit under sudden load, so it is not used. Lift closure constraints represent rigid joints with high impedance; their compliance is numerical and does not model measured bar flex or gear backlash.

## Geometry and linkage

Source file `GRMLN KISS Bot V2 Main Assm.stp` has SHA-256 `6346c299b035f8f21e0f56306827781bd0aa595cf404b091d51844b45b55fcfd`. The exporter retains its assembly hierarchy, transforms, geometric mass properties and cylindrical surfaces. The runtime transform is `(forward, left, up) = (CAD_Z, CAD_X, CAD_Y)`, a proper rotation. Dimensions use meters.

The lift has twelve hinges, five point closures and one gear relation. Their constraint Jacobian leaves one degree of freedom at the reference pose. The motor drives a crank through a 12:84 reduction. A connecting rod drives the lower lift, so motor angle and lift angle are not interchangeable. The two 84-tooth gear stages couple the upper and lower linkage. The end frame stays approximately level while rising. The notched “Darn You Dinkleberg” aligner follows the upper follower link and rotates with it, as in the supplied renders.

`assets/robot/lift_graph.json` records pivots, body parents, CAD ownership and closures. The operational lower-link range is -0.03 to 2.05 rad. This is an assumed useful branch range below the linkage singularity, not a measured mechanical hardstop. Point closure errors are measured in meters; gear and transmission errors are measured separately in radians.

Each wheel retains eighteen passive roller hinges, including their measured staggered rows and tangent axes. Wheel torque passes through the CAD-derived 24:32 external gearing. Each wheel represents two V5 motors. The claw uses two passive jaw hinges, opposing drive shafts and 1:1 chain relations to the rollers. The claw mount is fixed to the lift end at its CAD rest orientation and has no tilt actuator. The original pneumatic CAD part remains fixed to the lift end with its CAD-derived mass; hardware removal is unspecified. Passive pawls use CAD-derived tooth contacts. Two mount-anchored torsion springs provide passive jaw closure and centering. They are proposed hardware additions, not components confirmed in the STEP.

The two front toggle arms have no actuator or centering spring. Gravity lowers them against the user-confirmed pink stops. Their end wheels spin independently. Only the lower stop is established. The large numerical upper joint limit is not a claimed physical stop.

The pawl contact profiles come from the source STEP boundary surfaces, tessellated at 5 micrometers. Collision geometry uses the verified flat pawl layer and preserves the gear teeth. The source assembly itself contains intersecting pawl and gear teeth at its exported rest angles, so the simulation starts the passive pawls deflected clear of those teeth. Elastic pawl return and travel limits remain estimates. Intake lets the pawls ratchet; reverse rotation engages them to open the jaws. Motor HOLD acts on the intake shafts; it does not directly constrain the passive jaw angles.

The earlier direct-band assumption between CAD attachment nodes allowed both jaws to shift sideways together. The proposed paired springs restore the jaws toward the center. `ClawClosure` defaults to 0.04 Nm/rad stiffness and 0.085 Nm preload per jaw, giving closing torque comparable to the former band assumption over 0 to 0.4 rad opening. At 0.65 rad, each spring supplies 0.111 Nm. These are uncalibrated design assumptions. Use `build_robot_scene(claw_closure=ClawClosure(0, 0))` to examine the mechanism without added jaw-centering springs; assumed pawl-return springs remain. Consistent grasping with the proposed springs remains under verification.

## Motors and physical assumptions

The defaults are 600 RPM drivetrain cartridges, 100 RPM lift cartridges, and a 200 RPM intake cartridge. `RobotControl` accepts these cartridge choices independently; the lift verification CLI also accepts 200 RPM. CAD identifies motor housings and external gears, but cannot establish the installed internal cartridges. All identified motors currently use the 11 W model. The reusable `Motor` model also supports the fixed 200 RPM, 5.5 W motor.

The model bounds motor torque using published VEX operating points. A per-step work estimate uses the physical mass matrix and treats transmission equalities as rigid. Actual simulated equalities have finite compliance, so this estimate is not an exact general power guarantee. A loaded single-hinge regression measures applied torque through momentum balance. The 100 RPM intake configuration limits the physics step to 0.2 ms; the default 200 RPM intake uses 1 ms. Its conservative linear speed-torque envelope is an approximation, not VEX's regulated performance curve. Sources are [V5 motor specifications](https://kb.vex.com/hc/en-us/articles/360060929971-Understanding-V5-Smart-Motors), [11 W motor performance](https://kb.vex.com/hc/en-us/articles/360044325872-Understanding-V5-Smart-Motor-11W-Performance), and [5.5 W motor performance](https://kb.vex.com/hc/en-us/articles/10002101702932-Understanding-V5-Smart-Motor-5-5W-Performance).

`RobotControl(..., intake_brake="hold")` captures the intake shaft angle when the intake command returns to zero and applies bounded restoring motor torque. `brake` resists velocity without returning to a stored angle; `coast` applies no intake motor torque. This follows the active position-holding intent of [PROS motor brake modes](https://pros.cs.purdue.edu/v5/api/c/motors.html). HOLD is not an infinite-strength joint lock.

Material densities, printed infill, friction, damping and elastic band forces are estimates. Geometric inertia uses those estimated masses. Robot self-contact is disabled except for isolated gear-to-pawl contacts because the simplified bar proxies overlap at real joints. Field and game-piece contact remains enabled. Therefore these runs cannot validate robot self-clearance or precise hardware drift, grip strength, lift timing or scoring reliability.

The aligner uses convex prisms that preserve its notch. Small mounting holes are filled. Lift bars use capsules; chassis rails use boxes; intake tires use cylinders. The chassis baseplate, crossbar and front goal guides use CAD-section convex panels that preserve the 89.7 mm front opening. A former solid chassis box filled that opening and stopped the robot 54.8 mm before the real crossbar. Tests probe both the opening and surrounding material and verify unchanged compiled mass and inertia. Tires are rigid contact shapes without a calibrated deformation model. Original meshes remain visual-only, and fasteners contribute to mass without separate collision geometry. These choices keep contact work focused on functional surfaces.

The active claw tire instances reference CAD definition `0:1:1:148`, named `Urethane - Custom 2in Flexwheel`, with a 2-inch diameter and 1/2-inch width. Use 40A hardness as the user-specified assumption for these custom urethane wheels. Their force-deflection curve remains unknown. The separate `2_000_40A_500` definition is not the active claw tire and must not be used to assign its material. The current tire friction coefficient of 1.2 is an assumption. Native soft-contact experiments remain diagnostic; they do not establish a calibrated deformation model.

## Regenerate assets

```sh
uv sync --project sim --extra cad --extra test
uv run --project sim python sim/tools/extract_robot.py '/Users/ali/Downloads/GRMLN KISS Bot V2 Main Assm.stp'
uv run --project sim --with shapely python sim/tools/extract_aligner.py
uv run --project sim --extra cad python sim/tools/generate_robot_visuals.py
uv run --project sim --extra cad --with shapely==2.2.0 python sim/tools/extract_ratchet_profiles.py '/Users/ali/Downloads/GRMLN KISS Bot V2 Main Assm.stp'
```

The original STEP and user reference renders stay outside git. Packaged meshes, assembly metadata, roller profiles and the authored lift graph are runtime inputs. CAD extraction does not infer joint constraints automatically; review the authored graph when changing the source robot.

Regenerate chassis contact panels from the packaged source meshes with:

```sh
uv run --project sim --extra cad --with shapely==2.2.0 --with mapbox-earcut==2.1.0 --with networkx==3.7 --with trimesh==5.2.0 python sim/tools/extract_robot_chassis.py
```

`chassis_contacts.json` records source mesh hashes, section levels and approximation limits. The extractor simplifies section boundaries by 0.3 mm and fills small mounting holes. Its panels are horizontal extrusions, not exact curved BREP surfaces. Preserve the front-opening and material probes in `test_robot_chassis.py` when changing this asset.

The four claw arms use extruded profiles from the original STEP, preserving all five plate openings. Their former capsule contacts added material outside the arm boundary, including a failed-release contact point 2.23 mm outside the exact solid. `test_robot_jaw_contacts.py` probes that empty region and real material. Regenerate the profiles with `uv run --project sim --locked --extra cad --with shapely==2.2.0 python sim/tools/extract_robot_jaws.py '/Users/ali/Downloads/GRMLN KISS Bot V2 Main Assm.stp'`. The measured profile boundary deviation is at most 22.8 micrometers; joint travel and elastic preload remain estimates.

## Runtime measurement

On the development Mac with MuJoCo 3.17.0, five simulated seconds of the complete match field and articulated robot took a median 3.873 seconds over five runs, range 3.837–3.907 seconds. This is about 1.29 times real time without rendering. The benchmark includes motor feedback every physics step and excludes construction, video encoding and per-contact verification diagnostics. Every run completed all 5,000 steps with identical final robot position and no MuJoCo warnings. Other desktop applications and a video render were active during measurement.

A separate profile attributed about 93% of runtime to `mj_step`, rather than Python control. The model has 1,685 geoms and 529 generalized velocities. This is a historical workload measurement, not a claimed speedup or a guarantee for the current model. Retaining the full passive roller count and CAD pin support surfaces is deliberate; reducing them without contact regressions would undermine the behavior being tested.

That measurement predates removal of the claw tilt mechanism, the detailed insertion contacts, the pawl contacts, and the revised motor controller. The fixed mount removes the pitch hinge, cylinder hinge and slide, cylinder closure, and pitch actuator. Remeasure before attributing a speedup to that change.

## Simplified display meshes

Robot scenes default to `visual_mode='simple'`. The `robot-verify`, `robot-sequence`, and `robot-loader` commands accept `--visual-mode cad` to restore the original display meshes. Both modes use the same masses, inertias, joints, actuators, and collision geometry.

The simple meshes fill channel mounting holes while retaining open U and L cross-sections, smooth gear faces, simplify motor housings, and replace dense omni-wheel visuals with hubs and shaped roller barrels. The aligner's notch and the physical wheel rollers remain unchanged. Instanced robot display triangles decrease from 4,599,529 to 458,353. `assets/robot/simple_visuals.json` records each replacement and its source bounds. Sampled surface deviations are diagnostics, not formal maximum-error bounds.

Five interleaved robot-only runs on the development Mac rendered twenty warmed 960 by 720 frames in a median 0.232 seconds with simple meshes, compared with 1.157 seconds for CAD meshes. The ranges were 0.224 to 0.255 seconds and 1.152 to 1.211 seconds respectively. That is about 5.0 times the rendering throughput for this camera workload. A separate profile identified rendering as the dominant cost. The full-field transfer recording ran concurrently. These timings exclude physics stepping and video encoding, so they do not establish a physics speedup.

Scene construction medians were 0.081 seconds for simple meshes and 0.111 seconds for CAD meshes. The benchmark writes both images and all measurements to `sim/out/visual-benchmark`. The visual parity test compares compiled physical properties and motor-driven trajectories between modes.
