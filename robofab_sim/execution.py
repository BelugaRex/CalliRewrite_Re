"""Deterministic Cartesian playback shared by the Viser UI and evidence exporter.

This is kinematic/parametric simulation, not collision-checked robot control.
"""
from dataclasses import dataclass
import numpy as np
from kanjivg_to_trajectory import Z_SAFE, pressure_to_z


@dataclass
class Sample:
    target: np.ndarray
    pressure: float
    stroke: int  # -1 = approach; zero-based stroke index otherwise
    phase: str


def plan_samples(strokes, start_tip, spacing=0.0008):
    """Explicit lift -> travel at clearance -> lower -> write -> vertical lift."""
    if not strokes or spacing <= 0:
        raise ValueError("Nonempty strokes and positive spacing required")
    samples = []
    prev = np.asarray(start_tip, dtype=float)
    pressure = 0.0

    def segment(end, p, stroke, phase):
        nonlocal prev, pressure
        end = np.asarray(end, dtype=float)
        if not np.all(np.isfinite(end)):
            raise ValueError("Non-finite waypoint")
        n = max(1, int(np.ceil(np.linalg.norm(end-prev) / spacing)))
        for i in range(1, n+1):
            t = i/n
            samples.append(Sample(prev*(1-t)+end*t, pressure*(1-t)+p*t, stroke, phase))
        prev, pressure = end, p

    segment([*prev[:2], max(Z_SAFE, prev[2])], 0, -1, "lift")
    for idx, stroke in enumerate(strokes):
        a = np.asarray(stroke, dtype=float)
        if a.ndim != 2 or a.shape[1] != 4 or not np.all(np.isfinite(a)):
            raise ValueError("Each stroke must contain finite XY Z pressure waypoints")
        contact = a[a[:, 3] > 0]
        if len(contact) < 2:
            raise ValueError("Stroke must contain at least two contact waypoints")
        segment([*contact[0, :2], max(Z_SAFE, prev[2])], 0, idx, "travel")
        segment([*contact[0, :2], Z_SAFE], 0, idx, "lower")
        for j, point in enumerate(contact):
            segment([*point[:2], pressure_to_z(point[3])], point[3], idx,
                    "lower" if j == 0 else "write")
        segment([*contact[-1, :2], Z_SAFE], 0, idx, "lift")
    return samples


class Execution:
    """One sample step = actual IK/FK, brush contact, then paper deposition."""
    def __init__(self, kinematics, brush, paper, strokes, q=None):
        self.kinematics, self.brush, self.paper = kinematics, brush, paper
        self.q = (kinematics.NOMINAL_REST_Q if q is None else q).copy()
        self.last_tip = kinematics.forward_kinematics(self.q)[2]
        self.samples = plan_samples(strokes, self.last_tip)
        self.index = 0
        self.previous_ink = None
        self.trace = []

    @property
    def done(self):
        return self.index == len(self.samples)

    def step(self, bleed=.7):
        if self.done:
            return None
        sample = self.samples[self.index]
        self.q = self.kinematics.solve_ik(sample.target, self.q)
        hand, rotation, tip = self.kinematics.forward_kinematics(self.q)
        residual = float(np.linalg.norm(tip-sample.target))
        if residual > .0005:
            raise RuntimeError(f"IK failed at sample {self.index}: {residual:.6f} m")
        tuft = self.brush.compute_deformation(hand+.180*rotation[:, 2], tip, rotation,
                                             (tip-self.last_tip) * 60)
        # No ink in approach/travel even if a malformed physical pose contacts.
        deposit = sample.phase not in ("travel",) and sample.stroke >= 0 and tuft["is_contact"]
        if deposit:
            xy = tuft["actual_tip"][:2]
            # Join contact samples densely, but never bridge across a pen-up.
            n = 1 if self.previous_ink is None else max(1, int(np.ceil(np.linalg.norm(xy-self.previous_ink)/.0005)))
            for t in np.linspace(1/n, 1, n):
                p = xy if self.previous_ink is None else self.previous_ink*(1-t)+xy*t
                self.paper.deposit_ink(*p, tuft["footprint_radius"], tuft["normal_force"], bleed)
            self.previous_ink = xy.copy()
        else:
            self.previous_ink = None
        self.last_tip = tip.copy()
        row = dict(index=self.index, stroke=sample.stroke, phase=sample.phase,
                   target=sample.target.tolist(), pressure=float(sample.pressure),
                   q=self.q.tolist(), tip=tip.tolist(), residual_m=residual,
                   contact=bool(tuft["is_contact"]), deposited=bool(deposit),
                   force_n=float(tuft["normal_force"]), radius_m=float(tuft["footprint_radius"]),
                   axis_error=float(np.linalg.norm(rotation[:, 2]-[0, 0, -1])))
        self.trace.append(row)
        self.index += 1
        return tuft