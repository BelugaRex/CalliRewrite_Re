"""
RoboFab Calligraphy Simulator  (ISDN5240 Group Project)
=======================================================
Franka Panda arm + deformable brush + Xuan paper in Viser.

Stage 1 (60 pts) — manual waypoints, writes one character.
Stage 2 (80 pts) — pressure-modulated strokes, multiple characters.
Stage 3 (100 pts) — automatic trajectory from KanjiVG for any CJK character.

Usage:
    # Stage 1/2: choose from built-in characters (yong / yi / ren / zhong / shan)
    python calligraphy_sim.py

    # Stage 3: pass any Chinese character directly
    python calligraphy_sim.py --char 福
    python calligraphy_sim.py --char 水

Dependencies: viser, yourdfpy, robot_descriptions, Pillow, numpy
Data:         data/kanjivg.xml  (run data/download_kanjivg.sh to fetch)
"""

from __future__ import annotations
import sys
import argparse
import time
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# Stage 3: KanjiVG automatic trajectory generation
try:
    from kanjivg_to_trajectory import KanjiVGParser
    _KANJIVG_XML = Path(__file__).parent / "data" / "kanjivg.xml"
    _kvg_parser = KanjiVGParser(_KANJIVG_XML) if _KANJIVG_XML.exists() else None
except ImportError:
    _kvg_parser = None

import viser
import viser.transforms as tf

HAS_ROBOT_DESCRIPTIONS = False
try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    from viser.extras import ViserUrdf
    HAS_ROBOT_DESCRIPTIONS = True
except ImportError:
    pass

try:
    from deformable_brush_physics import RealisticDeformableTuft
except ImportError:
    # Standalone fallback if run without external module
    class RealisticDeformableTuft:
        def __init__(self, length=0.045, root_radius=0.007, stiffness=1.2, splay_factor=1.4, paper_z=0.002):
            self.length = length
            self.root_radius = root_radius
            self.stiffness = stiffness
            self.splay_factor = splay_factor
            self.paper_z = paper_z
            self.ink_level = 100.0

        def compute_deformation(self, root_pos, nominal_tip_pos, hand_rot, velocity):
            tool_dir = hand_rot[:, 2]
            penetration = max(0.0, self.paper_z - nominal_tip_pos[2])
            is_contact = penetration > 1e-4

            if is_contact:
                actual_tip = nominal_tip_pos.copy()
                actual_tip[2] = self.paper_z
                speed = float(np.linalg.norm(velocity[:2]))
                if speed > 1e-4:
                    drag_unit = velocity[:2] / speed
                    lag = min(0.018, penetration * 1.25)
                    actual_tip[:2] -= drag_unit * lag
                    drag_dir_3d = np.array([drag_unit[0], drag_unit[1], 0.0])
                else:
                    drag_dir_3d = np.array([0.0, 0.0, 0.0])

                P0 = root_pos
                P1 = root_pos + (self.length / 3.0) * tool_dir
                P2 = actual_tip - (self.length / 3.0) * drag_dir_3d
                P3 = actual_tip
                normal_force = self.stiffness * (penetration ** 1.25) * 650.0
                footprint_r = 0.0018 + self.splay_factor * np.sqrt(penetration) * 0.048
            else:
                actual_tip = nominal_tip_pos.copy()
                P0 = root_pos
                P1 = root_pos + (self.length / 3.0) * tool_dir
                P2 = root_pos + (2.0 * self.length / 3.0) * tool_dir
                P3 = nominal_tip_pos
                normal_force = 0.0
                footprint_r = 0.0

            u_vec = hand_rot[:, 0]
            v_vec = hand_rot[:, 1]
            num_rings = 10
            num_radial = 14
            s_vals = np.linspace(0.0, 0.95, num_rings)
            vertices = []

            for s in s_vals:
                C_s = ((1-s)**3)*P0 + 3*((1-s)**2)*s*P1 + 3*(1-s)*(s**2)*P2 + (s**3)*P3
                if s < 0.20:
                    r = self.root_radius + 0.0008 * (s / 0.20)
                else:
                    r = (self.root_radius + 0.0008) * (1.0 - (s - 0.20) / 0.80)**0.85
                if is_contact:
                    r += self.splay_factor * np.sqrt(penetration) * 0.042 * (s**1.6)

                for i in range(num_radial):
                    theta = 2.0 * np.pi * i / num_radial
                    radial_offset = (np.cos(theta) * u_vec + np.sin(theta) * v_vec) * r
                    pos = C_s + radial_offset
                    pos[2] = max(self.paper_z, pos[2])
                    vertices.append(pos)

            vertices.append(P3)
            vertices = np.array(vertices, dtype=np.float32)

            faces = []
            for r_idx in range(num_rings - 1):
                for i in range(num_radial):
                    next_i = (i + 1) % num_radial
                    v0 = r_idx * num_radial + i
                    v1 = r_idx * num_radial + next_i
                    v2 = (r_idx + 1) * num_radial + next_i
                    v3 = (r_idx + 1) * num_radial + i
                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])

            tip_idx = len(vertices) - 1
            last_ring_start = (num_rings - 1) * num_radial
            for i in range(num_radial):
                next_i = (i + 1) % num_radial
                faces.append([last_ring_start + i, last_ring_start + next_i, tip_idx])
            faces = np.array(faces, dtype=np.int32)

            num_bristles = 18
            num_subdiv = 6
            bristle_segments = []
            t_vals = np.linspace(0.0, 1.0, num_subdiv + 1)
            for b in range(num_bristles):
                b_angle = 2.0 * np.pi * b / num_bristles
                bristle_pts = []
                for s in t_vals:
                    C_s = ((1-s)**3)*P0 + 3*((1-s)**2)*s*P1 + 3*(1-s)*(s**2)*P2 + (s**3)*P3
                    if s < 0.20:
                        r = self.root_radius * 1.02
                    else:
                        r = (self.root_radius * 1.02) * (1.0 - (s - 0.20) / 0.80)**0.85
                    if is_contact:
                        r += self.splay_factor * np.sqrt(penetration) * 0.046 * (s**1.6)
                    b_offset = (np.cos(b_angle) * u_vec + np.sin(b_angle) * v_vec) * r
                    b_pos = C_s + b_offset
                    b_pos[2] = max(self.paper_z, b_pos[2])
                    bristle_pts.append(b_pos)
                for seg in range(num_subdiv):
                    bristle_segments.append([bristle_pts[seg], bristle_pts[seg + 1]])

            bristle_segments = np.array(bristle_segments, dtype=np.float32)
            return {
                "vertices": vertices,
                "faces": faces,
                "bristles": bristle_segments,
                "actual_tip": actual_tip,
                "penetration": penetration,
                "normal_force": normal_force,
                "footprint_radius": footprint_r,
                "is_contact": is_contact,
            }


# ==============================================================================
# 1. Franka Emika Panda Kinematics (Hand & Tool Forward/Inverse Kinematics)
# ==============================================================================
class PandaKinematics:
    """
    Kinematic model and Damped Least-Squares (DLS) IK for Franka Emika Panda.
    Calculates exact positions of both panda_hand (gripper) and the brush tip.
    """
    JOINT_LIMITS = [
        (-2.8973, 2.8973),  # panda_joint1
        (-1.7628, 1.7628),  # panda_joint2
        (-2.8973, 2.8973),  # panda_joint3
        (-3.0718, -0.0698), # panda_joint4
        (-2.8973, 2.8973),  # panda_joint5
        (-0.0175, 3.7525),  # panda_joint6
        (-2.8973, 2.8973),  # panda_joint7
    ]

    NOMINAL_REST_Q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.785])

    def __init__(self, tool_length=0.225):
        # Distance from panda_hand palm to undeformed brush tip:
        # Hand to grasp center: 0.075m, Grasp center to tip: 0.150m -> Total = 0.225m
        self.tool_length = tool_length

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            hand_pos: (3,) position of panda_hand origin in world
            hand_rot: (3,3) rotation matrix of panda_hand
            tip_pos:  (3,) position of nominal undeformed brush tip
        """
        dh_params = [
            (0.0,      0.333, -np.pi/2, 0.0),
            (0.0,      0.0,    np.pi/2, 0.0),
            (0.0825,   0.316,  np.pi/2, 0.0),
            (-0.0825,  0.0,   -np.pi/2, 0.0),
            (0.0,      0.384,  np.pi/2, 0.0),
            (0.088,    0.0,    np.pi/2, 0.0),
            (0.0,      0.107,  0.0,     0.0)
        ]

        T = np.eye(4)
        for i in range(7):
            a, d, alpha, theta_off = dh_params[i]
            theta = q[i] + theta_off
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)

            T_i = np.array([
                [ct, -st * ca,  st * sa, a * ct],
                [st,  ct * ca, -ct * sa, a * st],
                [0.0, sa,       ca,      d],
                [0.0, 0.0,      0.0,     1.0]
            ])
            T = T @ T_i

        hand_pos = T[:3, 3]
        hand_rot = T[:3, :3]
        tool_dir = hand_rot[:, 2]  # Hand Z axis (points forward between gripper fingers)
        tip_pos = hand_pos + self.tool_length * tool_dir
        return hand_pos, hand_rot, tip_pos

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        eps = 1e-5
        J = np.zeros((3, 7))
        _, _, p0 = self.forward_kinematics(q)

        for i in range(7):
            q_plus = q.copy()
            q_plus[i] += eps
            _, _, p_plus = self.forward_kinematics(q_plus)
            J[:, i] = (p_plus - p0) / eps

        return J

    def solve_ik(self, target_tip_pos: np.ndarray, current_q: np.ndarray, max_iters=25, tol=1e-4) -> np.ndarray:
        """
        DLS Inverse Kinematics targeting the nominal brush tip coordinate.
        """
        q = current_q.copy()
        lambda_damp = 0.04
        k_null = 0.15

        for _ in range(max_iters):
            _, _, curr_tip = self.forward_kinematics(q)
            err = target_tip_pos - curr_tip
            if np.linalg.norm(err) < tol:
                break

            J = self.compute_jacobian(q)
            JJT = J @ J.T + (lambda_damp ** 2) * np.eye(3)
            J_dls = J.T @ np.linalg.inv(JJT)

            null_proj = np.eye(7) - J_dls @ J
            delta_q_null = null_proj @ (k_null * (self.NOMINAL_REST_Q - q))

            delta_q = J_dls @ err + delta_q_null
            max_step = 0.20
            step_norm = np.linalg.norm(delta_q)
            if step_norm > max_step:
                delta_q = (delta_q / step_norm) * max_step

            q += delta_q

            for i in range(7):
                low, high = self.JOINT_LIMITS[i]
                q[i] = np.clip(q[i], low, high)

        return q


# ==============================================================================
# 2. Dynamic Xuan Paper Canvas (1024x1024)
# ==============================================================================
class XuanPaperCanvas:
    """
    1024x1024 Xuan Paper (宣纸) raster with 米字格 guide grid and capillary ink bleed.
    Coordinate mapping is calibrated so that when the robot moves from left to right
    across the table, the ink trail is drawn strictly from left to right.
    """
    def __init__(self, resolution=1024, physical_width=0.36, physical_height=0.36, origin_xy=(0.55, 0.0)):
        self.res = resolution
        self.pw = physical_width
        self.ph = physical_height
        self.ox, self.oy = origin_xy
        self.image = Image.new("RGBA", (self.res, self.res), (251, 247, 238, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.draw_grid()

    def draw_grid(self):
        pad = int(self.res * 0.06)
        w, h = self.res, self.res
        grid_color = (194, 53, 49, 75)
        self.draw.rectangle([pad, pad, w - pad, h - pad], outline=grid_color, width=3)
        self.draw.line([(w // 2, pad), (w // 2, h - pad)], fill=grid_color, width=2)
        self.draw.line([(pad, h // 2), (w - pad, h // 2)], fill=grid_color, width=2)
        self.draw.line([(pad, pad), (w - pad, h - pad)], fill=grid_color, width=1)
        self.draw.line([(w - pad, pad), (pad, h - pad)], fill=grid_color, width=1)

    def deposit_ink(self, world_x: float, world_y: float, radius: float, normal_force: float, bleed_factor=0.7):
        """
        Maps 3D world coordinates onto the 2D Xuan paper canvas.
        - world_x (forward/back) -> maps monotonically along the forward axis.
        - world_y (left/right)   -> maps monotonically along the lateral axis (Left -> Right).
        """
        rx = (world_x - (self.ox - self.pw / 2.0)) / self.pw
        # Fixed lateral mapping: as world_y moves from -Y to +Y (left to right), ry increases from 0 to 1!
        ry = (world_y - (self.oy - self.ph / 2.0)) / self.ph

        px = int(np.clip(rx * self.res, 0, self.res - 1))
        py = int(np.clip(ry * self.res, 0, self.res - 1))
        pr = max(2, int((radius / self.pw) * self.res))

        if 0 <= px < self.res and 0 <= py < self.res:
            alpha = min(240, int(140 + normal_force * 35))
            self.draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(16, 16, 16, alpha))
            if bleed_factor > 0.1:
                p_bleed = int(pr * (1.0 + bleed_factor * 0.35))
                self.draw.ellipse([px - p_bleed, py - p_bleed, px + p_bleed, py + p_bleed], outline=(30, 25, 20, 30), width=1)

    def clear(self):
        self.image = Image.new("RGBA", (self.res, self.res), (251, 247, 238, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.draw_grid()


# ==============================================================================
# 3. Calligraphy Trajectories (永字八法 & Classical Characters)
# ==============================================================================
CALLIGRAPHY_TRAJECTORIES = {
    # 永 (Yong - The 8 Principles: Ce, Le, Nu, Di, Ce, Lve, Zhuo, Zhe)
    "yong": [
        # 1. 点 (Ce / Dot)
        [
            (0.550, 0.000, 0.025, 0.0),
            (0.548, 0.003, -0.002, 0.8),
            (0.545, 0.006, -0.005, 1.2),
            (0.543, 0.003, -0.001, 0.5),
            (0.544, 0.001, 0.025, 0.0)
        ],
        # 2. 横 (Le / Horizontal) - Left to Right!
        [
            (0.525, -0.055, 0.025, 0.0),
            (0.524, -0.052, -0.003, 0.8),
            (0.523, -0.020, -0.002, 0.7),
            (0.522, 0.015, -0.002, 0.7),
            (0.521, 0.048, -0.005, 1.1),
            (0.523, 0.044, 0.025, 0.0)
        ],
        # 3. 竖 (Nu / Vertical) - Top to Bottom
        [
            (0.530, 0.000, 0.025, 0.0),
            (0.526, 0.000, -0.004, 1.0),
            (0.495, 0.000, -0.003, 0.8),
            (0.465, -0.001, -0.003, 0.8),
            (0.435, -0.002, -0.005, 1.1)
        ],
        # 4. 钩 (Di / Hook) - Hooks leftward
        [
            (0.435, -0.002, -0.005, 1.1),
            (0.433, -0.006, -0.006, 1.3),
            (0.442, -0.015, -0.002, 0.6),
            (0.455, -0.026, 0.025, 0.0)
        ],
        # 5. 提 (Ce / Rising) - Flicks upward-right
        [
            (0.465, -0.045, 0.025, 0.0),
            (0.463, -0.042, -0.003, 0.9),
            (0.472, -0.030, -0.002, 0.6),
            (0.485, -0.012, -0.001, 0.3),
            (0.490, -0.005, 0.025, 0.0)
        ],
        # 6. 撇 (Lve / Long Slanted) - Sweeps down-left
        [
            (0.510, 0.010, 0.025, 0.0),
            (0.508, 0.012, -0.004, 1.0),
            (0.490, 0.000, -0.003, 0.7),
            (0.470, -0.018, -0.002, 0.5),
            (0.448, -0.038, -0.001, 0.3),
            (0.430, -0.055, 0.025, 0.0)
        ],
        # 7. 短撇 (Zhuo / Peck) - Sharp down-left flick
        [
            (0.485, 0.015, 0.025, 0.0),
            (0.483, 0.018, -0.003, 0.8),
            (0.472, 0.010, -0.001, 0.4),
            (0.462, 0.002, 0.025, 0.0)
        ],
        # 8. 捺 (Zhe / Pressing Down-Right) - Deep press fanning to bottom-right
        [
            (0.468, 0.012, 0.025, 0.0),
            (0.465, 0.015, -0.002, 0.5),
            (0.450, 0.028, -0.004, 0.9),
            (0.435, 0.042, -0.006, 1.3),
            (0.428, 0.058, -0.003, 0.7),
            (0.424, 0.075, 0.025, 0.0)
        ]
    ],
    # 一 (Yi - Horizontal) - Left to Right!
    "yi": [
        [
            (0.550, -0.070, 0.025, 0.0),
            (0.548, -0.065, -0.002, 0.6),
            (0.547, -0.067, -0.005, 1.1),
            (0.546, -0.030, -0.003, 0.7),
            (0.545, 0.010, -0.003, 0.7),
            (0.544, 0.050, -0.003, 0.8),
            (0.543, 0.065, -0.006, 1.3),
            (0.545, 0.060, 0.025, 0.0)
        ]
    ]
}


# ==============================================================================
# 4. Main Viser Simulation Studio
# ==============================================================================
def main(extra_char: str | None = None):
    # Stage 3: inject KanjiVG trajectory for --char argument
    if extra_char and _kvg_parser is not None:
        try:
            traj = _kvg_parser.get_trajectory(extra_char)
            CALLIGRAPHY_TRAJECTORIES[extra_char] = traj
            print(f"[KanjiVG] Loaded '{extra_char}' → {len(traj)} strokes")
        except Exception as e:
            print(f"[KanjiVG] Failed to load '{extra_char}': {e}")
            extra_char = None

    print("=" * 70)
    print("  RoboFab Calligraphy Simulator  (ISDN5240)")
    print("=" * 70)

    # 1. Start Viser Server
    server = viser.ViserServer()
    server.initial_camera.position = (1.4, -0.9, 1.1)
    server.initial_camera.look_at = (0.55, 0.0, 0.2)
    server.initial_camera.up = (0.0, 0.0, 1.0)

    # 2. Add Studio Environment (Calligraphy Table, Xuan Paper, Scholar's Items)
    server.scene.add_grid("/ground", width=3.0, height=3.0, position=(0.0, 0.0, -0.4))

    # Rosewood Calligraphy Table (Top at Z = 0.0)
    server.scene.add_box(
        "/studio/table",
        color=(45, 30, 22),
        dimensions=(0.60, 0.80, 0.04),
        position=(0.55, 0.0, -0.02),
    )

    # Xuan Paper Canvas (Surface at Z = 0.002)
    paper = XuanPaperCanvas(resolution=1024, physical_width=0.36, physical_height=0.36, origin_xy=(0.55, 0.0))
    paper_handle = server.scene.add_image(
        "/studio/xuan_paper",
        image=np.array(paper.image.convert("RGB")),
        render_width=paper.pw,
        render_height=paper.ph,
        position=(paper.ox, paper.oy, 0.002),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    # Brass Paperweights
    server.scene.add_box(
        "/studio/paperweight_left",
        color=(185, 145, 60),
        dimensions=(0.025, 0.30, 0.012),
        position=(0.35, 0.0, 0.008),
    )
    server.scene.add_box(
        "/studio/paperweight_right",
        color=(185, 145, 60),
        dimensions=(0.025, 0.30, 0.012),
        position=(0.75, 0.0, 0.008),
    )

    # 3. Load Franka Emika Panda Model
    kinematics = PandaKinematics(tool_length=0.225)
    brush_tuft = RealisticDeformableTuft(length=0.045, root_radius=0.007, stiffness=1.2, splay_factor=1.4)
    current_q = kinematics.NOMINAL_REST_Q.copy()

    viser_urdf = None

    def update_viser_urdf_cfg(q_7dof: np.ndarray):
        """
        Safely maps 7-DOF arm configuration to the URDF actuated joint array.
        Sets Franka gripper fingers to 0.006m, actively grasping the 11mm brush handle!
        """
        if viser_urdf is None:
            return
        actuated_limits = viser_urdf.get_actuated_joint_limits()
        num_actuated = len(actuated_limits)
        full_cfg = np.zeros(num_actuated)
        # 7 arm revolute joints
        for idx in range(min(7, num_actuated)):
            full_cfg[idx] = q_7dof[idx]
        # Franka gripper finger joints (clamp onto bamboo handle)
        if num_actuated > 7:
            for idx in range(7, num_actuated):
                full_cfg[idx] = 0.006
        viser_urdf.update_cfg(full_cfg)

    if HAS_ROBOT_DESCRIPTIONS:
        try:
            print("[INFO] Loading official Franka Emika Panda description via robot_descriptions...")
            urdf = load_robot_description("panda_description", load_meshes=True, build_scene_graph=True)
            viser_urdf = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/franka_panda")
            update_viser_urdf_cfg(current_q)
            print(f"[INFO] Franka Panda URDF and meshes loaded successfully ({len(viser_urdf.get_actuated_joint_limits())} actuated joints)!")
        except Exception as e:
            print(f"[WARN] Could not load meshes via robot_descriptions: {e}. Using geometric representation.")

    # 4. Create Initial Brush Visual Scene Elements
    hand_p, hand_r, nominal_tip_p = kinematics.forward_kinematics(current_q)
    tool_dir = hand_r[:, 2]
    q_wxyz = tf.SO3.from_matrix(hand_r).wxyz

    # Bamboo handle clamped between Franka gripper finger pads
    handle_init_pos = hand_p + 0.085 * tool_dir
    brush_handle = server.scene.add_cylinder(
        "/brush/handle",
        radius=0.0055,
        height=0.140,
        color=(155, 110, 60),
        position=handle_init_pos,
        wxyz=q_wxyz,
    )

    # Black Ox Horn Ferrule (斗)
    ferrule_init_pos = hand_p + 0.1675 * tool_dir
    brush_ferrule = server.scene.add_cylinder(
        "/brush/ferrule",
        radius=0.0072,
        height=0.025,
        color=(25, 25, 25),
        position=ferrule_init_pos,
        wxyz=q_wxyz,
    )

    # Compute initial tuft deformation
    tuft_root_p = hand_p + 0.180 * tool_dir
    init_tuft_data = brush_tuft.compute_deformation(
        tuft_root_p, nominal_tip_p, hand_r, velocity=np.zeros(3)
    )

    # Lofted 3D Ink-Sheened Core Mesh
    brush_tuft_mesh = server.scene.add_mesh_simple(
        "/brush/tuft_core",
        vertices=init_tuft_data["vertices"],
        faces=init_tuft_data["faces"],
        color=(20, 20, 20),
        flat_shading=False,
    )

    # Exterior Fine Bristle Hair Fibers
    brush_bristles = server.scene.add_line_segments(
        "/brush/bristles",
        points=init_tuft_data["bristles"],
        colors=(38, 34, 30),
        thickness=0.0006,
    )

    # 5. Viser Interactive GUI Controls
    _default_char = extra_char if extra_char and extra_char in CALLIGRAPHY_TRAJECTORIES else "yong"
    with server.gui.add_folder("Calligraphy Trajectory Execution"):
        char_selector = server.gui.add_dropdown(
            "Character",
            options=list(CALLIGRAPHY_TRAJECTORIES.keys()),
            initial_value=_default_char
        )
        btn_play = server.gui.add_button("Execute Character (Write)")
        btn_step = server.gui.add_button("Step Stroke")
        btn_reset = server.gui.add_button("Reset Arm Pose")
        slider_speed = server.gui.add_slider("Writing Speed", min=0.2, max=3.0, step=0.1, initial_value=1.0)
        chk_loop = server.gui.add_checkbox("Loop Playback", False)

    with server.gui.add_folder("Manual Cartesian IK Jogging"):
        slider_ik_x = server.gui.add_slider("Tip X (Reach)", min=0.35, max=0.75, step=0.002, initial_value=0.55)
        slider_ik_y = server.gui.add_slider("Tip Y (Lateral)", min=-0.18, max=0.18, step=0.002, initial_value=0.0)
        slider_ik_z = server.gui.add_slider("Tip Z (Height/Press)", min=-0.010, max=0.080, step=0.001, initial_value=0.025)
        btn_touch_paper = server.gui.add_button("Touch Paper (Z=0)")
        btn_lift_safe = server.gui.add_button("Lift Brush (Z=35mm)")

    with server.gui.add_folder("Joint Angles (7-DOF Franka Panda)"):
        joint_sliders = []
        for i, (low, high) in enumerate(PandaKinematics.JOINT_LIMITS):
            js = server.gui.add_slider(
                f"panda_joint{i+1}",
                min=low,
                max=high,
                step=0.01,
                initial_value=float(current_q[i])
            )
            joint_sliders.append(js)

    with server.gui.add_folder("Tuft & Xuan Paper Physics"):
        slider_stiffness = server.gui.add_slider("Bristle Stiffness", min=0.5, max=2.5, step=0.1, initial_value=1.2)
        slider_splay = server.gui.add_slider("Splay Factor (铺毫)", min=0.8, max=2.5, step=0.1, initial_value=1.4)
        slider_bleed = server.gui.add_slider("Paper Absorbency", min=0.0, max=2.0, step=0.1, initial_value=0.7)
        btn_clear = server.gui.add_button("Wash Paper (Clear)")
        btn_dip = server.gui.add_button("Dip Ink (100%)")

    # GUI Callbacks
    @btn_clear.on_click
    def _(_):
        paper.clear()
        paper_handle.image = np.array(paper.image.convert("RGB"))

    @btn_dip.on_click
    def _(_):
        brush_tuft.ink_level = 100.0

    @btn_touch_paper.on_click
    def _(_):
        slider_ik_z.value = 0.001

    @btn_lift_safe.on_click
    def _(_):
        slider_ik_z.value = 0.035

    # Trajectory execution state
    play_state = {
        "running": False,
        "char": "yong",
        "stroke_idx": 0,
        "point_idx": 0,
        "t": 0.0,
        "last_tip": np.array([0.55, 0.0, 0.025]),
    }

    @btn_play.on_click
    def _(_):
        play_state["running"] = not play_state["running"]
        btn_play.label = "Pause Writing" if play_state["running"] else "Resume Writing"

    @btn_step.on_click
    def _(_):
        play_state["running"] = False
        btn_play.label = "Resume Writing"
        char_strokes = CALLIGRAPHY_TRAJECTORIES[char_selector.value]
        play_state["stroke_idx"] = (play_state["stroke_idx"] + 1) % len(char_strokes)
        play_state["point_idx"] = 0
        play_state["t"] = 0.0

    @btn_reset.on_click
    def _(_):
        nonlocal current_q
        play_state["running"] = False
        play_state["stroke_idx"] = 0
        play_state["point_idx"] = 0
        play_state["t"] = 0.0
        btn_play.label = "Execute Character (Write)"
        current_q = kinematics.NOMINAL_REST_Q.copy()
        update_viser_urdf_cfg(current_q)
        for i, js in enumerate(joint_sliders):
            js.value = float(current_q[i])
        _, _, tip_p = kinematics.forward_kinematics(current_q)
        slider_ik_x.value = round(float(tip_p[0]), 3)
        slider_ik_y.value = round(float(tip_p[1]), 3)
        slider_ik_z.value = round(float(tip_p[2]), 3)

    # Interactive slider callbacks
    def handle_cartesian_slider_change(_):
        if not play_state["running"]:
            target_pos = np.array([slider_ik_x.value, slider_ik_y.value, slider_ik_z.value])
            nonlocal current_q
            current_q = kinematics.solve_ik(target_pos, current_q)
            update_viser_urdf_cfg(current_q)
            for i, js in enumerate(joint_sliders):
                js.value = float(current_q[i])

    slider_ik_x.on_update(handle_cartesian_slider_change)
    slider_ik_y.on_update(handle_cartesian_slider_change)
    slider_ik_z.on_update(handle_cartesian_slider_change)

    def make_joint_slider_cb(idx):
        def _(_):
            if not play_state["running"]:
                current_q[idx] = joint_sliders[idx].value
                update_viser_urdf_cfg(current_q)
                _, _, tip_p = kinematics.forward_kinematics(current_q)
                slider_ik_x.value = round(float(tip_p[0]), 3)
                slider_ik_y.value = round(float(tip_p[1]), 3)
                slider_ik_z.value = round(float(tip_p[2]), 3)
        return _

    for i, js in enumerate(joint_sliders):
        js.on_update(make_joint_slider_cb(i))

    # 6. Main Simulation Tick Loop
    print("\\n[READY] Franka Panda Calligraphy Simulator is running!")
    print("👉 Connect to Viser at the URL displayed above (e.g. http://localhost:8080 or http://localhost:8081)\\n")

    last_time = time.time()
    while True:
        now = time.time()
        dt = min(0.1, now - last_time)
        last_time = now

        # Update physical parameters from sliders
        brush_tuft.stiffness = slider_stiffness.value
        brush_tuft.splay_factor = slider_splay.value
        bleed = slider_bleed.value

        # A. Trajectory Player
        if play_state["running"]:
            char_strokes = CALLIGRAPHY_TRAJECTORIES[char_selector.value]
            if play_state["stroke_idx"] < len(char_strokes):
                stroke = char_strokes[play_state["stroke_idx"]]
                p_idx = play_state["point_idx"]

                if p_idx + 1 < len(stroke):
                    p0 = stroke[p_idx]
                    p1 = stroke[p_idx + 1]

                    play_state["t"] += dt * slider_speed.value * 2.2
                    alpha = min(1.0, play_state["t"])
                    smooth_t = alpha * alpha * (3.0 - 2.0 * alpha)

                    target_x = p0[0] + (p1[0] - p0[0]) * smooth_t
                    target_y = p0[1] + (p1[1] - p0[1]) * smooth_t
                    target_z = p0[2] + (p1[2] - p0[2]) * smooth_t

                    target_pos = np.array([target_x, target_y, target_z])
                    current_q = kinematics.solve_ik(target_pos, current_q)

                    if play_state["t"] >= 1.0:
                        play_state["t"] = 0.0
                        play_state["point_idx"] += 1
                else:
                    play_state["stroke_idx"] += 1
                    play_state["point_idx"] = 0
                    play_state["t"] = 0.0
            else:
                if chk_loop.value:
                    play_state["stroke_idx"] = 0
                    play_state["point_idx"] = 0
                    play_state["t"] = 0.0
                    paper.clear()
                    paper_handle.image = np.array(paper.image.convert("RGB"))
                else:
                    play_state["running"] = False
                    btn_play.label = "Replay Character"

            for i, js in enumerate(joint_sliders):
                js.value = float(current_q[i])
            slider_ik_x.value = round(float(target_pos[0]), 3)
            slider_ik_y.value = round(float(target_pos[1]), 3)
            slider_ik_z.value = round(float(target_pos[2]), 3)

        # B. Synchronize URDF Robot Mesh Articulation
        update_viser_urdf_cfg(current_q)

        # C. Retrieve Precise Gripper Pose & Update Rigid Brush Parts
        hand_p, hand_r, nominal_tip_p = kinematics.forward_kinematics(current_q)
        tool_dir = hand_r[:, 2]
        q_wxyz = tf.SO3.from_matrix(hand_r).wxyz

        # Update handle & ferrule cylinders (held directly between gripper finger pads)
        brush_handle.position = hand_p + 0.085 * tool_dir
        brush_handle.wxyz = q_wxyz
        brush_ferrule.position = hand_p + 0.1675 * tool_dir
        brush_ferrule.wxyz = q_wxyz

        # D. Realistic Deformable Tuft Mechanics
        tuft_root_p = hand_p + 0.180 * tool_dir
        velocity = (nominal_tip_p - play_state["last_tip"]) / max(1e-4, dt)
        play_state["last_tip"] = nominal_tip_p.copy()

        tuft_data = brush_tuft.compute_deformation(tuft_root_p, nominal_tip_p, hand_r, velocity)

        # Update 3D tuft core mesh and fine bristle fibers
        server.scene.add_mesh_simple(
            "/brush/tuft_core",
            vertices=tuft_data["vertices"],
            faces=tuft_data["faces"],
            color=(20, 20, 20),
            flat_shading=False,
        )
        server.scene.add_line_segments(
            "/brush/bristles",
            points=tuft_data["bristles"],
            colors=(38, 34, 30),
            thickness=0.0006,
        )

        # E. Ink Deposition onto Xuan Paper
        if tuft_data["is_contact"]:
            actual_tip = tuft_data["actual_tip"]
            fn = tuft_data["normal_force"]
            fr = tuft_data["footprint_radius"]
            paper.deposit_ink(actual_tip[0], actual_tip[1], fr, fn, bleed_factor=bleed)
            paper_handle.image = np.array(paper.image.convert("RGB"))

        time.sleep(0.015)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RoboFab Calligraphy Simulator")
    ap.add_argument("--char", default=None,
                    help="Stage 3: any CJK character to write automatically, e.g. --char 福")
    args = ap.parse_args()
    main(extra_char=args.char)