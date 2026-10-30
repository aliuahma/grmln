from pathlib import Path
import json
import os
import platform
import statistics
import time

import mujoco
import numpy as np

from .scene import load, check_state


def measure(model, seconds=4., repeats=5):
    results = []
    for _ in range(repeats):
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data, nstep=250)
        start_sim = data.time
        start = time.perf_counter()
        mujoco.mj_step(model, data, nstep=round(seconds / model.opt.timestep))
        elapsed = time.perf_counter() - start
        check_state(model, data)
        advanced = data.time - start_sim
        if abs(advanced - seconds) > model.opt.timestep:
            raise RuntimeError('Timed loop did not advance the requested simulated duration')
        results.append({'wall_seconds': elapsed, 'simulation_seconds': advanced, 'real_time_factor': advanced / elapsed,
                        'contacts': data.ncon, 'max_speed': float(np.max(np.abs(data.qvel))), 'warnings': 0})
    return results


def benchmark(layout, output):
    _, model, _ = load(layout)
    results = measure(model)
    factors = [r['real_time_factor'] for r in results]
    result = {'layout': str(layout), 'mujoco': mujoco.__version__, 'machine': platform.platform(),
              'cpu_count': os.cpu_count(), 'load_average': os.getloadavg(), 'timestep_s': model.opt.timestep,
              'geoms': model.ngeom, 'bodies': model.nbody, 'degrees_of_freedom': model.nv,
              'method': 'Five repeats. 0.5s warmup then 4 simulated seconds. Native mj_step; no rendering or model compilation timed.',
              'median_real_time_factor': statistics.median(factors), 'range_real_time_factor': [min(factors), max(factors)],
              'samples': results, 'scope': 'Resting full field with active game-object contacts, not robot collision stress.'}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / 'benchmark.json').write_text(json.dumps(result, indent=2))
    return result
