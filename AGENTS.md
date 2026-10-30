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
- For code changes, build and inspect compilation/link output. CI packages the library through `LemLib/pros-build@v5.1.0`; there is no checked-in automated unit-test suite or simulator harness.
- Add focused regression checks for substantive math/control fixes where feasible. A passing cross-compile does not establish sensor correctness, motion accuracy, task safety, or competition behavior; state what still needs a V5 hardware test.
- Uploading or running the current demo can move the robot during initialization. Keep build verification distinct from hardware execution, and verify the robot configuration and intended behavior before a hardware run.
- For documentation changes requiring rendering, use the dependencies in `docs/requirements.txt`, then run `doxygen` and `make html` from `docs/`; report unavailable tooling or stale doc configuration rather than claiming a successful render.
- Keep unrelated working-tree changes intact. Update this guide when project structure, dependency versions, or verified development commands change.
