"""
Deformable Chinese Calligraphy Brush Hair Physics & Geometry Engine
===================================================================
This module provides continuous non-linear elastoplastic modeling and dynamic 3D
mesh generation for a traditional Chinese calligraphy brush tuft (毛笔/毛毫).

Physics Principles:
1. Continuous Cubic Bézier Spine Curve under normal contact and planar friction lag.
2. Hard non-penetration constraint against the Xuan paper surface (Z >= Z_paper).
3. Non-linear radial splaying (铺毫 / Pūháo) expanding the footprint under compression.
4. Dual visual representation: smooth 3D lofted core mesh + exterior fine bristle fibers.
"""

from __future__ import annotations
import numpy as np


class RealisticDeformableTuft:
    """
    Parametric geometry and physics generator for a deformable Chinese calligraphy brush tuft.
    """
    def __init__(
        self,
        length: float = 0.045,        # 45 mm tuft length
        root_radius: float = 0.007,   # 7 mm ferrule opening
        stiffness: float = 1.2,       # Elasticity modulus
        splay_factor: float = 1.4,    # Splaying expansion coefficient
        paper_z: float = 0.002        # Paper surface altitude (m)
    ):
        self.length = length
        self.root_radius = root_radius
        self.stiffness = stiffness
        self.splay_factor = splay_factor
        self.paper_z = paper_z
        self.ink_level = 100.0

    def compute_deformation(
        self,
        root_pos: np.ndarray,
        nominal_tip_pos: np.ndarray,
        hand_rot: np.ndarray,
        velocity: np.ndarray
    ) -> dict:
        """
        Calculates the deforming 3D mesh vertices and individual bristle line segments.
        Constrains hair vertices strictly above the paper plane so the brush never penetrates.

        Args:
            root_pos: (3,) World coordinates of the ferrule rim where hairs emerge.
            nominal_tip_pos: (3,) Unconstrained nominal tip position if tuft were rigid.
            hand_rot: (3, 3) Rotation matrix of the gripper / brush handle.
            velocity: (3,) Linear velocity vector of the brush tip.

        Returns:
            dict containing:
                - vertices: (V, 3) float32 array of lofted 3D mesh vertices
                - faces: (F, 3) int32 array of triangular faces
                - bristles: (N, 2, 3) float32 array of individual hair fiber segments
                - actual_tip: (3,) contact tip position on paper
                - penetration: float, compression depth delta (m)
                - normal_force: float, contact force Fn (N)
                - footprint_radius: float, splayed contact footprint radius (m)
                - is_contact: bool
        """
        tool_dir = hand_rot[:, 2]  # Unit vector along the brush shaft
        penetration = max(0.0, self.paper_z - nominal_tip_pos[2])
        is_contact = penetration > 1e-4

        if is_contact:
            # Constrain tip to paper surface plane
            actual_tip = nominal_tip_pos.copy()
            actual_tip[2] = self.paper_z

            # Directional drag lag opposite to horizontal velocity
            speed = float(np.linalg.norm(velocity[:2]))
            if speed > 1e-4:
                drag_unit = velocity[:2] / speed
                lag = min(0.018, penetration * 1.25)
                actual_tip[:2] -= drag_unit * lag
                drag_dir_3d = np.array([drag_unit[0], drag_unit[1], 0.0])
            else:
                drag_dir_3d = np.array([0.0, 0.0, 0.0])

            # Cubic Bézier spine control points: P0=root, P1=tangent at root, P2=tangent at contact, P3=tip
            P0 = root_pos
            P1 = root_pos + (self.length / 3.0) * tool_dir
            P2 = actual_tip - (self.length / 3.0) * drag_dir_3d
            P3 = actual_tip

            # Normal contact force follows Hertzian elasticity (N)
            normal_force = self.stiffness * (penetration ** 1.25) * 650.0
            # Splayed footprint radius on paper (铺毫)
            footprint_r = 0.0018 + self.splay_factor * np.sqrt(penetration) * 0.048
        else:
            actual_tip = nominal_tip_pos.copy()
            P0 = root_pos
            P1 = root_pos + (self.length / 3.0) * tool_dir
            P2 = root_pos + (2.0 * self.length / 3.0) * tool_dir
            P3 = nominal_tip_pos
            normal_force = 0.0
            footprint_r = 0.0

        # Build orthonormal basis perpendicular to tool_dir
        u_vec = hand_rot[:, 0]
        v_vec = hand_rot[:, 1]

        # A. Lofted 3D Mesh Vertices (Ink-Saturated Core Body)
        num_rings = 10
        num_radial = 14
        s_vals = np.linspace(0.0, 0.95, num_rings)
        vertices = []

        for s in s_vals:
            # Cubic Bézier curve point along spine
            C_s = ((1-s)**3)*P0 + 3*((1-s)**2)*s*P1 + 3*(1-s)*(s**2)*P2 + (s**3)*P3

            # Natural Chinese brush fullness profile (wide base, full belly at 20%, sharp tip)
            if s < 0.20:
                r = self.root_radius + 0.0008 * (s / 0.20)
            else:
                r = (self.root_radius + 0.0008) * (1.0 - (s - 0.20) / 0.80)**0.85

            # Lateral splaying under contact pressure
            if is_contact:
                r += self.splay_factor * np.sqrt(penetration) * 0.042 * (s**1.6)

            for i in range(num_radial):
                theta = 2.0 * np.pi * i / num_radial
                radial_offset = (np.cos(theta) * u_vec + np.sin(theta) * v_vec) * r
                pos = C_s + radial_offset
                pos[2] = max(self.paper_z, pos[2])  # strictly above paper
                vertices.append(pos)

        # Apex tip vertex
        vertices.append(P3)
        vertices = np.array(vertices, dtype=np.float32)

        # Build faces
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

        # B. Individual Fine Bristle Fibers (Line Segments: N, 2, 3)
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