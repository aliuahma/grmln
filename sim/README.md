# Override university simulation

Run route experiments on separate VEX U match and skills fields in MuJoCo. Geometry and placements come from the supplied Override STEP assembly. The presets follow the university figures in manual v2.1.

This is a field and contact sandbox with an example differential-drive robot. It does not execute PROS firmware or the LemLib controller. Masses, inertia, friction, damping, and drive response need physical calibration before route timing predicts your robot.

The separate [GRMLN KISS V2 robot](ROBOT.md) uses the supplied robot CAD, articulated lift and claw, omni rollers, and passive wall-toggle arms. Its mechanism and driving videos have dedicated `robot-verify` and `robot-sequence` commands. Existing `run` routes still use the example robot.

## Run with uv

From the repository root:

```sh
uv sync --project sim --extra test
uv run --project sim override-sim render --layout university-match --output sim/out/university-match
uv run --project sim override-sim render --layout university-skills --output sim/out/university-skills
uv run --project sim override-sim verify --output sim/out/verification
uv run --project sim override-sim run --layout university-skills --route sim/routes/example.json --output sim/out/route
uv run --project sim pytest -q
uv run --project sim override-sim benchmark --layout university-skills --output sim/out/benchmark
```

`sim/uv.lock` pins the Python environment. Ordinary runs need no CAD tools or original STEP file. Rendering requires an OpenGL context. This Mac supports rendering with the commands above.

For the interactive viewer on macOS, use MuJoCo's main-thread launcher:

```sh
uv run --project sim mjpython -m override_sim view --layout university-match
```

On Linux, use `uv run --project sim override-sim view`. Headless Linux rendering may require `MUJOCO_GL=egl` and a working EGL driver.

`render` writes overhead, oblique, collision-overlay, and settled PNGs, plus a six-second mechanism video and trajectory. `verify` writes individual experiment videos and measurements. `run` writes a route video, trajectory CSV, and outcome JSON. Use `--no-video` for faster route iterations. A route timeout exits with status 1; invalid input or simulation failure exits with status 2.

## Inspect the physics suite

The verification suite uses the same bodies and collision geometry as the complete fields. Each experiment prunes the rest of the field. It changes initial poses or velocities once, then uses gravity, contact, or applied forces.

- A pin falls into an upright cup and settles.
- A cup descends over a pin and settles.
- An off-center pin hits the rim and deflects.
- Pins and cups roll and tumble with initial velocity.
- A toggle receives a torque pulse, rotates beyond one turn, and coasts.
- A loader rises and lowers while its guide stays fixed.
- A pin drops through the raised loader's open lower passage.

Inspect `sim/out/verification/report.json` for checks and measurements. Each scenario has an MP4, PNG, trajectory JSON, and XML with an `experiment_initial` keyframe. Reset MuJoCo to keyframe 0 to reconstruct the starting state. The scripted torque and loader-control schedule remain in `verification.py`.

These tests establish simulated behavior and detect filled openings, blocked motion, unstable contacts, and missing limits. They do not establish measured physical realism. The aperture test starts a pin inside the raised chute; it does not demonstrate loading a cup through the entire top opening.

## Field evidence and coordinates

The official sources are [university rule modifications](https://manuals.vex.com/v5/override/v2.1/vex-u/), [university match figure](https://content.vexrobotics.com/docs/2026-2027/override/online-manual/assets/image/VEX-U-Top.png), [university skills figure](https://content.vexrobotics.com/docs/2026-2027/override/online-manual/assets/image/VEXUSkills-Top.png), and [Appendix A](https://manuals.vex.com/v5/override/v2.1/appendix-a/).

Both presets contain 36 cups, 16 horizontal alliance/yellow pins, four upright red/blue pins, and 12 yellow/yellow pins in cups. Match adds one yellow/yellow pin in the center goal. Skills leaves all nine goals empty. The four short neutral goals are empty in both university presets. Off-field match loads and robot preloads are not spawned into the field.

Coordinates use meters, with the origin at the floor center. Positive x points right and positive y points up in the university overhead figure. Positive z points upward. Yaw is counterclockwise from positive x. A LemLib compass heading converts as `yaw = pi/2 - compass_radians`.

The source STEP uses millimeters, with Y vertical. The exporter converts to meters. Runtime applies `(x, y, z) = (CAD_X, -CAD_Z, CAD_Y + 0.2945892)` to place the measured tile surface at z=0. The measured tile pitch is about 0.5980584 m. Do not replace these placements with a nominal 24-inch grid.

## Collision and mechanism approximations

Fixed perimeter hardware becomes four wall colliders and one floor. Tile seams and tape are visual geometry. Fasteners do not receive separate bodies or collisions. Render meshes have collision masks disabled and contribute no mass.

Pins, cups, and goal openings use hollow convex partitions of the original STEP surfaces, tessellated at 0.025 mm linear tolerance. Display mesh reduction does not determine insertion clearances. Axial sections preserve flange relief, cup shoulders, and the inner throat instead of bridging them with one convex hull. Separate filled cup proxies contact only the floor. The pin partitions preserve the approximately 15-degree starting tilt of loose radial pins. Tests drop pins into cups and cups over pins, checking settled center spacing against the 85 mm CAD seating offset. Current nested-pin offsets differ by less than 1 mm; tessellation and convex approximation still limit precision. Goal insertion tests cover floor, short, and tall supports. Goals retain hollow openings and solid supports. Small concavities, narrow slots, and exact rim details are omitted from contact geometry. Use the collision-overlay image to inspect this difference.

Four toggles use continuous hinges about the CAD bar axes. Their collision shape retains a triangular cross-section. Initial upper faces are yellow. Damping and breakaway friction are estimates; there is no calibrated three-position detent. The four loaders have vertical slide joints with 0.2185 m travel, derived from the 365 mm and 583.5 mm heights in the official loader drawing. Their sleeves use CAD-derived wall panels that preserve the lower diagonal cutouts, side openings, and open rear face. Fixed guides retain contact with the robot and game pieces. Sleeve-to-guide contact is excluded because the slide joint already constrains that bearing motion. A force-limited position actuator represents a person lifting a loader. Initial sleeve bottoms are normalized to the floor because the STEP has different red and blue starting heights.

The floor contact time constant is 0.004 seconds at a 0.002-second simulation step. Articulated robot scenes use a 0.001-second step. Physical masses, contact coefficients, controller response, and material compliance are assumptions in `field.py` and `route.py`. No GPS-strip texture, official scoring, pin-half separation, sensor noise, or odometry model is implemented.

## Define a route

Copy `sim/routes/example.json`. The schema requires `schema_version: 1`, `frame: "field_center_m_rad"`, a three-number `start`, two-number `waypoints`, `timeout_s`, and optional `speed_mps`. Coordinates must be finite and inside +/-1.7 m. Initial robot contacts penetrating more than 3 mm are rejected before route execution.

The example robot has two driven wheels, low-friction caster contacts, a free chassis, and bounded motor torque. A truth-pose controller steers toward waypoints within 65 mm. It can collide, stall, tip, or time out. It is not calibrated to either university robot size or drivetrain. The example route illustrates the interface, not a competition strategy.

## Regenerate the CAD assets

The original STEP stays outside git. The compact derived visual meshes, complete instance transforms, radial profiles, and source SHA-256 are under `override_sim/assets/cad/`.

```sh
uv sync --project sim --extra cad --extra test
uv run --project sim python sim/tools/extract_cad.py '/Users/ali/Downloads/276-9250-000 (2026-04-26).STEP'
```

The extractor tessellates reusable definitions rather than every occurrence. It omits fastener visuals, records local and accumulated transforms, and verifies the common tile-top height. Visual decimation uses a fixed sampling seed and bounded sampled deviation. That sampled bound is not a formal maximum surface-error guarantee. `profiles.json` records radial cross-sections for inspection; the smallest radius can include fastening ribs and does not prove a circular opening.

The Python scene generator owns field assembly and collision approximations. `route.py` owns the hypothetical robot and route follower. `scene.py` owns stepping and output. `verification.py` owns physical experiments. Keep the PROS library and its build independent of this package.

See [performance checks](PERFORMANCE.md) for measured route-loop improvements and the sleeping configurations rejected by fidelity checks.

Regenerate functional contacts with `uv run --project sim --extra cad python sim/tools/extract_piece_contacts.py '/Users/ali/Downloads/276-9250-000 (2026-04-26).STEP'`. The extractor checks the source hash and records tessellation parameters in `assets/cad/piece_contacts.json`. Keep seating-depth and loose-pin tilt tests together when validating these contacts. Convex partitions must end at functional ledges; a hull spanning a ledge can block insertion even when its vertices lie on the original surface.

The seating audit reduced the nested pin's rise above its CAD pose from 12.36 mm to 0.55 mm. Measured supporting contact points were within 0.30 mm of the unreduced cup surface. These are measurements for the tested resting configurations, not a global collision-error bound. The source assembly also has up to 0.317 mm of overlap between loose radial pins and adjacent cups. Tests retain those source poses and permit their small initial relaxation while separately checking floor penetration.
