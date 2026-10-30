# Development guide

## Purpose and current state
- This is a LemLib fork intended for VEX U development on the VEX V5 brain using PROS and C++. Preserve the upstream MIT license.
- The checkout is the in-progress 0.6.0 rewrite, not the stable 0.5.x API. Prefer current headers and implementations over README feature lists, old tutorials, and copied examples.
- `Makefile` sets `IS_LIBRARY=1`: it builds `bin/LemLib.a` and a demo robot program. `src/main.cpp` is excluded from the library archive.
- The current `src/main.cpp` configures specific motor/sensor ports, starts odometry, prints pose to the LCD, and calls `turnTo` inside `initialize()`. `autonomous()` and `opcontrol()` are empty. Treat it as a hardware demo; do not assume its wiring, tuning, or initialization motion suits our robot.

## Code map
- `include/lemlib/` contains public APIs; `src/lemlib/` contains their implementations. `lemlib.hpp` is the umbrella header, with optional macros and namespace aliases.
- `motions/{turnTo,moveToPoint,moveToPose,follow}` implement turning/swing turns, point motion, pose motion, and path following. Each exposes parameter/settings structs with motor groups, pose callbacks, controllers, and other dependencies.
- `config.hpp` declares external defaults used by settings structs. Callers using those defaults must provide definitions; do not silently hard-code robot configuration into library algorithms.
- `tracking/TrackingWheelOdom` wraps tracking wheels and combines encoder/IMU readings into a pose in a PROS background task. Review object lifetime and shared pose access when changing it.
- `MotionHandler` runs motions asynchronously and supports cancellation/wait helpers. `MotionCancelHelper` handles loop timing, task notifications, and competition-state cancellation; motion loops currently use a 10 ms period.
- `PID`, `Timer`, `ExitCondition`, and `util` provide control, timing, settling, and math helpers. Preserve reset behavior, elapsed-time semantics, timeout handling, and drivetrain stopping/brake restoration when modifying control code.
- `include/{pros,hardware,units,LemLog,lemlog,hot-cold-asset,liblvgl}` and dependency archives in `firmware/` are supplied dependencies. Avoid incidental edits or mass formatting there; use PROS for dependency changes. Preserve exact include-path casing for Linux CI.
- `docs/` uses Sphinx, MyST, Breathe, and Doxygen. Some pages and configuration still reference older chassis APIs; verify examples against this checkout.

## Build and dependencies
- Run commands from the repo root. Use `pros make` for an incremental build; use `pros make clean` followed by `pros make` after dependency/build changes or for a clean verification.
- The verified local setup is PROS CLI 3.5.6, ARM GCC 14.3.1, and GNU make. `common.mk` requests `gnu23` / `gnu++26` and the `arm-none-eabi-` toolchain, not Apple's host compiler.
- On this Mac, `pros` is in `~/.local/bin`; the compiler is in `/opt/homebrew/opt/arm-gcc-bin@14/bin`. Interactive zsh is configured. If a noninteractive shell cannot find the compiler, run `export PATH="/opt/homebrew/opt/arm-gcc-bin@14/bin:$PATH"` first.
- The CLI is installed through `uv tool` with Python 3.12 and `setuptools<81` because it still uses `pkg_resources`. A Python deprecation warning is distinct from a C++ build failure.
- `project.pros` records kernel 4.2.2, hardware 0.5.0, units 0.7.3, lemlog 0.3.0, hot-cold-asset 1.0.2, and liblvgl 9.2.0. Consult it before changing versions; do not automatically upgrade the kernel or libraries.
- `.gitignore` excludes `include/liblvgl/*` and `firmware/liblvgl.a`. Fresh clones can list liblvgl as installed while lacking its files. Restore the recorded version with `pros conductor install liblvgl@9.2.0 --force-apply`; this downloads from the PROS depot and restores LCD support.
- Dependency installation may update machine-specific template locations in `project.pros`; distinguish this metadata change from a version change.
- Successful builds produce `bin/LemLib.a`, `bin/cold.package.bin`, and `bin/hot.package.bin`. Generated binaries, `.d/`, and compile commands are ignored; do not commit them.
- `pros make template` is the packaging target. Before distributing a template, check `TEMPLATE_FILES`: its current chassis/differential globs refer to an older layout and must not be assumed to export the current API correctly.

## Conventions and verification
- Follow `.clang-format` and `.github/CONTRIBUTING.md`: four-space indentation, 120 columns, lowerCamelCase functions/variables, UpperCamelCase types, same-line braces. CI uses clang-format 19 on `src/lemlib` and `include/lemlib`; format only relevant changes.
- Keep public declarations and Doxygen comments aligned with implementation changes. Use the existing `lemlib` namespace, typed units, and hardware abstractions.
- Preserve unit distinctions: lengths/time are quantities; `_cDeg` denotes compass angles while `_stDeg` denotes standard-position angles. `AngleRange` is a magnitude. Review `include/units/Angle.hpp` before changing angle arithmetic or conversions.
- PID inputs are explicitly converted (e.g. meters and standard radians), and PID integration/differentiation use elapsed seconds. Do not import stable-version tuning constants or assume raw PROS motor command scales apply to these wrappers.
- For code changes, build and inspect compilation/link output. CI packages the library through `LemLib/pros-build@v5.1.0`; the separate Python simulator has its own tests and does not validate PROS firmware.
- Add focused regression checks for substantive math/control fixes where feasible. A passing cross-compile does not establish sensor correctness, motion accuracy, task safety, or competition behavior; state what still needs a V5 hardware test.
- Uploading or running the current demo can move the robot during initialization. Keep build verification distinct from hardware execution, and verify the robot configuration and intended behavior before a hardware run.
- For documentation changes requiring rendering, use the dependencies in `docs/requirements.txt`, then run `doxygen` and `make html` from `docs/`; report unavailable tooling or stale doc configuration rather than claiming a successful render.
- Keep unrelated working-tree changes intact. Update this guide when project structure, dependency versions, or verified development commands change.

## University field simulator
- `sim/` is a separate MuJoCo Python package. Use uv for all Python package management. Run `uv sync --project sim --extra test`, then `uv run --project sim pytest -q`. Commit `sim/uv.lock` when dependencies change.
- `sim/README.md` documents rendering, route experiments, physics verification videos, and optional CAD extraction. Runtime needs only packaged assets; original STEP files and generated videos stay outside git.
- Preserve university match/skills distinctions, hollow object contacts, continuous toggle hinges, and loader slide travel. Collision masks deliberately separate cup-floor proxies from hollow cup-object contacts.
- Loose radial pins start tilted in the source CAD. Preserve their asymmetric CAD convex-hull support surfaces; circular frustum proxies incorrectly flatten them under gravity. Check both initial and settled poses.
- Physical coefficients and the example robot are uncalibrated. Do not describe passing simulated checks as hardware validation or LemLib controller parity.
- `sim/ROBOT.md` documents the CAD-derived KISS V2 robot. `robot-verify` exercises its mechanisms; `robot-sequence` runs contact-driven pin and toggle interactions. Existing route commands retain the simpler example robot.
- Preserve the lift's closed loops, aligner follower ownership, and passive lower-stopped toggle arms. Passive omni rollers must not inherit driven-joint armature; transfer their mass and inertia out of the parent wheel aggregate to avoid double counting.
- Motor cartridge changes must update both command and force limits. Cartridge choices, jaw springs and robot self-clearance remain assumptions until calibrated or measured. The claw mount is fixed at its CAD rest orientation by the user's hardware decision; do not restore a tilt actuator.

- Robot display meshes default to the simplified mode. Keep `visual_mode="cad"` available for comparison. Display simplification must preserve collision geometry, masses, inertias, and joints; `test_robot_visuals.py` checks compiled properties and trajectory parity. Regenerate replacements with `uv run --project sim --extra cad python sim/tools/generate_robot_visuals.py`.
- Collision simplification must preserve seating depth and accessible openings, not only outer bounds. A single convex hull can bridge a flange or taper and create false support. Compare CAD sections with contact surfaces and verify settled pin-in-cup and cup-over-pin poses before accepting proxy changes or pickup videos.
- Empty-goal placement requires the pin below the cup and actual nesting. Reject support-only states and excessive opposing contact forces even when the render looks seated. Grasp verification must measure orientation, lateral centering, depth behind both roller columns and loaded contacts as well as retention. The pair must enter past the rollers before closure, then back out at pickup height and clear the loader before lifting. Any proposed hardware additions, including the paired jaw springs, must appear in exported experiment reports as assumptions.
