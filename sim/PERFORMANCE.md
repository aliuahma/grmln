# Performance checks

The first optimization pass preserves the field geometry, collision masks, timestep, solver settings, and rendered detail. The route runner classifies geoms once and filters contacts with arrays on every physics step. Brief contacts remain observable.

## Route contact reporting

A five-pair interleaved comparison on the development Mac used the same frozen university-skills model and example route for both implementations. Rendering and model compilation were excluded; route control, stepping, contact reporting, and output writing were included. The route advanced 4.692 simulated seconds and completed in every run.

| Implementation | Median wall time | Range |
| --- | --- | --- |
| Original contact reporting | 2.826 s | 2.767–2.872 s |
| Cached classification and array filtering | 1.780 s | 1.774–1.822 s |

This is 37% less wall time for that workload. All ten trajectory files were byte-identical and all non-timing result fields matched. A separate profile found about 1.52 million repeated body-name lookups in the original loop. The optimized loop resolves names once per geom. These measurements isolate contact reporting with sleeping disabled; they are not a combined speedup claim or a prediction for other robots.

## Sleeping was rejected

Sleeping remains disabled. Global tolerance sweeps changed the resting pose of pins by about 9 mm and 16 degrees. Keeping pins awake improved resting equivalence, but paired active-contact comparisons still failed the acceptance bounds of 0.5 mm and 0.5 degrees.

Even with pins always awake and a sleep tolerance of `1e-8`, an off-center rim impact differed by 0.664 mm and 1.181 degrees in the generalized-position trajectory. This comparison used positions derived from `qpos`, avoiding differences caused by stale intermediate body transforms. All eight coarse scenario outcome checks could pass while these trajectory differences remained.

Sleeping is therefore not part of the measured route improvement. Reconsider it only with paired active-contact trajectories and wake-up tests, not resting stability or scenario success alone.

## Geometry and rendering

Removing all 249 visual-only geoms from the match model produced no measurable native stepping improvement in five interleaved runs. The final physical state matched exactly. Visuals remain available for inspection and videos.

A separate diagnostic profile attributed about 72% of native stepping time to collision detection. Remaining contacts include cups resting on the floor, pins nesting in cups, and neighboring pieces touching. No collider was removed merely because it was outside the camera view.

A 60-frame diagnostic at 960×720 found scene updates much cheaper than rendering and image readback. It ran during other investigation work and is not a controlled performance comparison. Rendering quality, resolution, shadows, and mesh detail remain unchanged.

## Recheck on your machine

Use `uv run --project sim pytest -q` for physics and contact-reporting regressions. Use `uv run --project sim override-sim benchmark --layout university-skills --output sim/out/benchmark` for native field stepping. Use `uv run --project sim override-sim run --layout university-skills --route sim/routes/example.json --no-video --output sim/out/route-performance` for complete route-loop timing. Compare matching workloads and settings; rendering and compilation timings must be reported separately.
