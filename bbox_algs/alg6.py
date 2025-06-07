from __future__ import annotations

"""
Default Bounding Box Algorithm — extended with edge-correction and distance parameter
=============================================================

This module extends the original implementation by passing the distance between
camera and target to the edge-correction routines for further adjustments.
"""

import math
from typing import List, Tuple

from awpy.vector import Vector3

__all__ = [
    "ypr_to_vec",
    "DefaultBBoxAlg",
]

# ────────────────── Player model & optics constants ────────────────────
STAND_Z: float = 72.0          # eye height of a standing player (UU)
CROUCH_Z: float = 54.0         # eye height when crouched (UU)
HITBOX_HALF_W: float = 16      # half-width of the 32×32 UU head/torso box
HITBOX_H: float = 73           # total hit-box height (UU)

# ───────────────────── Edge-correction coefficients ────────────────────
EDGE_CORR_H: float = 3.45    # pixel-shift per degree horizontally
EDGE_CORR_V: float = 1.77       # pixel-shift per degree vertically

# ───────────────────────── Helper functions ────────────────────────────

def ypr_to_vec(yaw: float, pitch: float) -> Vector3:
    """Convert *yaw* & *pitch* (in **degrees**) to a forward *unit* vector."""
    y_rad = math.radians(yaw)
    p_rad = math.radians(pitch)
    return Vector3(
        math.cos(p_rad) * math.cos(y_rad),
        math.cos(p_rad) * math.sin(y_rad),
        -math.sin(p_rad),
    )

# ────────────────────────── Core algorithm ─────────────────────────────

class DefaultBBoxAlg:
    """Corner-sampling bounding-box projection **with edge-correction**.

    Now also passes distance between camera and target to the adjust routines.
    """

    def __init__(
        self,
        fov_deg: float,
        screen_w: int,
        screen_h: int,
        *,
        edge_corr_h: float | None = None,
        edge_corr_v: float | None = None,
    ) -> None:
        self.v_fov_deg = fov_deg
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.aspect: float = screen_w / screen_h
        self.tan_half_vfov: float = math.tan(math.radians(fov_deg) / 2)
        self.h_fov_deg: float = math.degrees(
            2 * math.atan(self.aspect * self.tan_half_vfov)
        )

        # Model half-width (constant).
        self.hit_w: float = HITBOX_HALF_W

        # Edge-correction slopes
        self.k_h: float = EDGE_CORR_H if edge_corr_h is None else edge_corr_h
        self.k_v: float = EDGE_CORR_V if edge_corr_v is None else edge_corr_v

    def calcbb(
        self,
        from_tuple: Tuple[float, float, float, float, float, bool],
        to_tuple:   Tuple[float, float, float, float, float, bool],
    ) -> Tuple[int, int, int, int] | None:
        """Project *to_tuple* into *from_tuple*’s screen space with distance-aware edge correction."""
        # Unpack camera & target state
        px, py, pz, yaw, pitch, from_crouch = from_tuple
        tx, ty, tz, _, _, tgt_crouch = to_tuple

        # Build camera basis
        eye_z = pz + (CROUCH_Z if from_crouch else STAND_Z)
        pov_eye = Vector3(px, py, eye_z)

        fwd = ypr_to_vec(yaw, pitch).normalize()
        right = fwd.cross(Vector3(0, 0, 1)).normalize()
        up = right.cross(fwd)

        # Target height based on stance
        tgt_height = CROUCH_Z if tgt_crouch else HITBOX_H

        # Sample 8 corners
        pts_px: List[Tuple[float, float]] = []
        for dx in (-self.hit_w, self.hit_w):
            for dy in (-self.hit_w, self.hit_w):
                for dz in (0.0, tgt_height):
                    world_pt = Vector3(tx + dx, ty + dy, tz + dz)
                    x_c, y_c, z_c = self._world_to_cam(world_pt, pov_eye, right, up, fwd)
                    x_ndc, y_ndc = self._cam_to_ndc(x_c, y_c, z_c)
                    if x_ndc is None or y_ndc is None:
                        continue
                    px_x, px_y = self._ndc_to_px(x_ndc, y_ndc)
                    pts_px.append((px_x, px_y))

        if not pts_px:
            return None

        # Tight AABB
        xs, ys = zip(*pts_px)
        xmin = max(0.0, min(xs))
        ymin = max(0.0, min(ys))
        xmax = min(self.screen_w, max(xs))
        ymax = min(self.screen_h, max(ys))

        # Compute distance between camera and target center
        center_world = Vector3(tx, ty, tz + tgt_height / 2)
        diff = center_world - pov_eye
        dist = math.sqrt(diff.dot(diff))

        # Apply edge-correction with distance
        xmin, ymin, xmax, ymax = self._apply_edge_correction(
            xmin, ymin, xmax, ymax, dist
        )

        # Clamp & return ints
        return (
            int(max(0, min(self.screen_w, xmin))),
            int(max(0, min(self.screen_h, ymin))),
            int(max(0, min(self.screen_w, xmax))),
            int(max(0, min(self.screen_h, ymax))),
        )

    def _world_to_cam(
        self,
        pt: Vector3,
        eye: Vector3,
        r: Vector3,
        u: Vector3,
        f: Vector3,
    ) -> Tuple[float, float, float]:
        d = pt - eye
        return d.dot(r), d.dot(u), d.dot(f)

    def _cam_to_ndc(
        self,
        x_c: float,
        y_c: float,
        z_c: float,
    ) -> Tuple[float | None, float | None]:
        if z_c <= 0:
            return None, None
        x_ndc = (x_c / z_c) * (1 / self.aspect) / self.tan_half_vfov
        y_ndc = (y_c / z_c) / self.tan_half_vfov
        return x_ndc, y_ndc

    def _ndc_to_px(self, x_ndc: float, y_ndc: float) -> Tuple[float, float]:
        x_px = (x_ndc + 1) * self.screen_w / 2
        y_px = (1 - y_ndc) * self.screen_h / 2
        return x_px, y_px

    def _apply_edge_correction(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        distance: float,
    ) -> Tuple[float, float, float, float]:
        # Centre in pixel space
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        # NDC
        x_ndc = (2 * cx / self.screen_w) - 1.0
        y_ndc = 1.0 - (2 * cy / self.screen_h)

        # Angles
        theta_h = x_ndc * (self.h_fov_deg / 2)
        theta_v = y_ndc * (self.v_fov_deg / 2)

        # Apply with distance
        xmin = self.adjust_x(theta_h, xmin, distance)
        xmax = self.adjust_x(theta_h, xmax, distance)
        ymin = self.adjust_y(theta_v, ymin, distance)
        ymax = self.adjust_y(theta_v, ymax, distance)

        return xmin, ymin, xmax, ymax

    def adjust_x(self, theta_h: float, x_px: float, distance: float) -> float:
        """Horizontal edge-correction with distance parameter."""
        # Default: ignore distance
        return x_px + self.k_h * theta_h 

    def adjust_y(self, theta_v: float, y_px: float, distance: float) -> float:
        """Vertical edge-correction with distance parameter."""
        # Default: ignore distance
        scaled_distance = distance/200
        return y_px - self.k_v * theta_v * scaled_distance
