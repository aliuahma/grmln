"""Compare robot display modes with matched physics and interleaved render runs."""
import argparse
import json
from pathlib import Path
import statistics
import time

import mujoco
import numpy as np
from PIL import Image

from override_sim.robot import build_robot_scene


def measure(mode, output):
    started = time.perf_counter()
    xml, metadata = build_robot_scene(visual_mode=mode)
    model = mujoco.MjModel.from_xml_string(xml)
    build_seconds = time.perf_counter() - started
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.03, 0, .29]
    camera.distance, camera.azimuth, camera.elevation = 1.2, 135, -20
    options = mujoco.MjvOption()
    options.geomgroup[3] = False
    with mujoco.Renderer(model, 720, 960) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        renderer.render()
        started = time.perf_counter()
        for index in range(20):
            camera.azimuth = 135 + index
            renderer.update_scene(data, camera=camera, scene_option=options)
            pixels = renderer.render()
        render_seconds = time.perf_counter() - started
        assert pixels.shape == (720, 960, 3) and pixels.std() > 5
        Image.fromarray(pixels).save(output / f'{mode}.png')
    return {'build_s': build_seconds, 'render_20_frames_s': render_seconds,
            'mesh_faces': model.nmeshface, 'body_mass_kg': float(model.body_mass.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('sim/out/visual-benchmark'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in ('cad', 'simple'):
        measure(mode, args.output)
    runs = {'cad': [], 'simple': []}
    for repetition in range(5):
        for mode in (('cad', 'simple') if repetition % 2 == 0 else ('simple', 'cad')):
            runs[mode].append(measure(mode, args.output))
    report = {'mujoco': mujoco.__version__, 'resolution': [960, 720], 'runs': runs,
              'scope': 'Scene construction and warmed rendering separately. No simulation stepping or video encoding.',
              'median': {mode: {metric: statistics.median(run[metric] for run in records)
                                for metric in ('build_s', 'render_20_frames_s')}
                         for mode, records in runs.items()}}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
