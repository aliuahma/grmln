"""Measure five repeated motor-driven runs of the complete articulated match field."""
import argparse
import cProfile
import json
import platform
import statistics
import time
from pathlib import Path

import mujoco
import numpy as np

from override_sim.field import Layout
from override_sim.robot import build_robot_scene
from override_sim.robot_control import RobotControl

xml, metadata = build_robot_scene(Layout.MATCH, start=(-.15, -.9, 0.))
model = mujoco.MjModel.from_xml_string(xml)
control = RobotControl(model, metadata)


def run():
    data = mujoco.MjData(model)
    contacts = 0
    start = time.perf_counter()
    for _ in range(5000):
        control.update(data, forward=.3 if data.time > 1 else 0.)
        mujoco.mj_step(model, data)
        contacts = max(contacts, data.ncon)
    elapsed = time.perf_counter() - start
    assert np.isfinite(data.qpos).all() and not any(w.number for w in data.warning)
    return {'wall_s': elapsed, 'simulated_s': data.time, 'max_contacts': contacts,
            'robot_position_m': data.body('robot').xpos.tolist()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('sim/out/robot-benchmark'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run()
    rows = [run() for _ in range(5)]
    report = {'machine': platform.platform(), 'mujoco': mujoco.__version__, 'steps': 5000,
              'dt_s': model.opt.timestep, 'ngeom': model.ngeom, 'nv': model.nv,
              'scope': 'Full university match field, articulated robot, five simulated seconds, motor feedback every step; excludes scene build, rendering and diagnostics.',
              'runs': rows, 'median_wall_s': statistics.median(r['wall_s'] for r in rows)}
    (args.output / 'runtime.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    cProfile.run('run()', str(args.output / 'runtime.prof'))
