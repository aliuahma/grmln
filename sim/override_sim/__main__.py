from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .field import Layout


def main():
    parser = argparse.ArgumentParser(description='University Override MuJoCo field and route experiments')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('render', 'run', 'view', 'benchmark'):
        child = commands.add_parser(command)
        child.add_argument('--layout', choices=[x.value for x in Layout], default=Layout.MATCH.value)
        if command != 'view':
            child.add_argument('--output', type=Path, default=Path('sim/out') / command)
        if command == 'run':
            child.add_argument('--route', type=Path, required=True)
            child.add_argument('--no-video', action='store_true')
        if command == 'render':
            child.add_argument('--seconds', type=float, default=6.)
    verify = commands.add_parser('verify')
    verify.add_argument('--output', type=Path, default=Path('sim/out/verification'))
    robot = commands.add_parser('robot-verify')
    robot.add_argument('--output', type=Path, default=Path('sim/out/robot'))
    robot.add_argument('--lift-rpm', type=int, choices=(100, 200), default=100)
    robot.add_argument('--no-video', action='store_true')
    robot.add_argument('--visual-mode', choices=('simple', 'cad'), default='simple')
    sequence = commands.add_parser('robot-sequence')
    sequence.add_argument('--output', type=Path, default=Path('sim/out/robot/sequence'))
    sequence.add_argument('--no-video', action='store_true')
    sequence.add_argument('--visual-mode', choices=('simple', 'cad'), default='simple')
    loader = commands.add_parser('robot-loader')
    loader.add_argument('--output', type=Path, default=Path('sim/out/robot/loader'))
    loader.add_argument('--no-video', action='store_true')
    loader.add_argument('--visual-mode', choices=('simple', 'cad'), default='simple')
    args = parser.parse_args()
    try:
        if args.command == 'render':
            from .scene import render
            if not 0 < args.seconds <= 120:
                raise ValueError('seconds must be between 0 and 120')
            result = render(args.layout, args.output, seconds=args.seconds)
        elif args.command == 'run':
            from .route import load_route
            from .scene import run_route
            result = run_route(args.layout, load_route(args.route), args.output, video=not args.no_video)
        elif args.command == 'verify':
            from .verification import verify
            result = verify(args.output)
        elif args.command == 'robot-verify':
            from .robot_verification import verify_robot
            result = verify_robot(args.output, video=not args.no_video, lift_rpm=args.lift_rpm, visual_mode=args.visual_mode)
        elif args.command == 'robot-sequence':
            from .robot_sequence import run_sequence
            result = run_sequence(args.output, video=not args.no_video, visual_mode=args.visual_mode)
        elif args.command == 'robot-loader':
            from .robot_loader import run_loader_transfer
            result = run_loader_transfer(args.output, video=not args.no_video, visual_mode=args.visual_mode)
        elif args.command == 'benchmark':
            from .benchmark import benchmark
            result = benchmark(args.layout, args.output)
        else:
            import mujoco
            import mujoco.viewer
            from .scene import load
            _, model, data = load(args.layout)
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    start = time.perf_counter()
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(max(0, model.opt.timestep - (time.perf_counter() - start)))
            return 0
        print(json.dumps(result, indent=2))
        return 1 if result.get('passed') is False or result.get('outcome') in ('timeout', 'incomplete') else 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f'override-sim: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
