"""Routes and a hypothetical contact-driven differential robot, in SI units."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

WHEEL_RADIUS = 0.05
TRACK = 0.28
MAX_WHEEL_SPEED = 16.0


@dataclass(frozen=True)
class Route:
    start: tuple[float, float, float]
    waypoints: tuple[tuple[float, float], ...]
    timeout_s: float
    speed_mps: float = 0.4

    def __post_init__(self):
        def numbers(value, length):
            if not isinstance(value, (list, tuple)) or len(value) != length:
                raise ValueError(f"expected {length} coordinates")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
                raise ValueError("coordinates must be finite numbers")
            return tuple(float(v) for v in value)

        start = numbers(self.start, 3)
        if not isinstance(self.waypoints, (list, tuple)) or not self.waypoints:
            raise ValueError("route requires at least one waypoint")
        points = tuple(numbers(point, 2) for point in self.waypoints)
        if any(abs(v) > 1.7 for point in (start[:2], *points) for v in point):
            raise ValueError("route coordinates must be inside +/-1.7 m")
        for name in ("timeout_s", "speed_mps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.speed_mps > MAX_WHEEL_SPEED * WHEEL_RADIUS:
            raise ValueError("speed_mps exceeds robot wheel speed limit")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "waypoints", points)


def load_route(path: Path) -> Route:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("route must be a JSON object")
    allowed = {"schema_version", "frame", "start", "waypoints", "timeout_s", "speed_mps"}
    if value.keys() - allowed:
        raise ValueError("unknown route fields")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported route schema_version")
    if value.get("frame") != "field_center_m_rad":
        raise ValueError("route frame must be field_center_m_rad")
    try:
        return Route(value["start"], value["waypoints"], value["timeout_s"], value.get("speed_mps", 0.4))
    except KeyError as exc:
        raise ValueError(f"missing route field {exc.args[0]}") from exc


def add_robot(worldbody: ET.Element, actuator: ET.Element, start: tuple[float, float, float]) -> None:
    x, y, yaw = start
    body = ET.SubElement(worldbody, "body", name="robot", pos=f"{x} {y} {WHEEL_RADIUS + .003}",
                         quat=f"{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}")
    ET.SubElement(body, "freejoint", name="robot_free")
    ET.SubElement(body, "geom", name="robot_chassis", type="box", size=".16 .12 .035", pos="0 0 .045",
                  mass="3.5", rgba=".12 .25 .65 1")
    for side, sign in (("left", 1), ("right", -1)):
        wheel = ET.SubElement(body, "body", name=f"robot_{side}_wheel", pos=f"0 {sign * TRACK / 2} 0")
        ET.SubElement(wheel, "joint", name=f"robot_{side}", type="hinge", axis="0 1 0", damping=".01")
        ET.SubElement(wheel, "geom", type="cylinder", size=f"{WHEEL_RADIUS} .015", quat=".70710678 .70710678 0 0",
                      mass=".2", friction="1 .005 .0001", rgba=".08 .08 .08 1")
        ET.SubElement(actuator, "velocity", name=f"robot_{side}_drive", joint=f"robot_{side}", kv=".5",
                      ctrllimited="true", ctrlrange=f"-{MAX_WHEEL_SPEED} {MAX_WHEEL_SPEED}",
                      forcelimited="true", forcerange="-1.5 1.5")
    for sign in (-1, 1):
        ET.SubElement(body, "geom", type="sphere", size=".018", pos=f"{sign * .125} 0 -.032", mass=".05",
                      friction=".001 .00001 .00001", priority="1", rgba=".4 .4 .4 1")


class RouteFollower:
    """Truth-pose waypoint controller. Does not emulate PROS or LemLib."""

    def __init__(self, route: Route):
        self.route = route
        self.index = 0
        self.completed = False
        self.error: str | None = None
        self._start_time: float | None = None

    def update(self, model, data) -> bool:
        if self._start_time is None:
            self._start_time = float(data.time)
        body = model.body("robot").id
        ids = [model.actuator(f"robot_{side}_drive").id for side in ("left", "right")]
        position = data.xpos[body]
        rotation = data.xmat[body]
        if not all(math.isfinite(float(v)) for v in (*position, *rotation)):
            self.error = "nonfinite robot pose"
        if not self.completed and data.time - self._start_time >= self.route.timeout_s:
            self.error = "route timeout"
        left = right = 0.0
        if not self.completed and self.error is None:
            while self.index < len(self.route.waypoints):
                tx, ty = self.route.waypoints[self.index]
                dx, dy = tx - position[0], ty - position[1]
                distance = math.hypot(dx, dy)
                if distance > .065:
                    break
                self.index += 1
            self.completed = self.index == len(self.route.waypoints)
            if not self.completed:
                yaw = math.atan2(rotation[3], rotation[0])
                angle = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
                angular = max(-2.5, min(2.5, 4.0 * angle))
                linear = min(self.route.speed_mps, distance * 1.5) * max(0.0, math.cos(angle))
                left = (linear - angular * TRACK / 2) / WHEEL_RADIUS
                right = (linear + angular * TRACK / 2) / WHEEL_RADIUS
                scale = max(1.0, abs(left) / MAX_WHEEL_SPEED, abs(right) / MAX_WHEEL_SPEED)
                left, right = left / scale, right / scale
        data.ctrl[ids] = (left, right)
        return self.completed

    def report(self) -> dict:
        return {"completed": self.completed, "waypoint_index": self.index, "waypoint_count": len(self.route.waypoints),
                "error": self.error}
