"""
Default Bounding Box Algorithm — extracted from tick_bbxdbg.py
==============================================================

This module provides a reference implementation of a simple, corner-sampling
bounding-box projection for Counter-Strike 2 player models.

It mirrors the original implementation that was embedded in *tick_bbxdbg.py*
but is now packaged as a standalone, importable module so that the main
CLI stays focused on orchestration and user-interaction logic.
"""
from __future__ import annotations

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
HITBOX_HALF_W: float = 16.0    # half-width of the 32×32 UU head-&-torso box
HITBOX_H: float = 73.0         # total hitbox height (UU)

# ───────────────────────── Helper functions ────────────────────────────

def ypr_to_vec(yaw: float, pitch: float) -> Vector3:
    """Convert yaw & pitch (in *degrees*) to a forward unit vector."""
    y_rad = math.radians(yaw)
    p_rad = math.radians(pitch)
    return Vector3(
        math.cos(p_rad) * math.cos(y_rad),  # x
        math.cos(p_rad) * math.sin(y_rad),  # y
        -math.sin(p_rad),                   # z (negative because CS2 uses left-handed)
    )

# ────────────────────────── Core algorithm ─────────────────────────────

class DefaultBBoxAlg:
    """A naïve, corner-sampling bounding-box projection algorithm.

    * Exactly the same maths that used to live in *tick_bbxdbg.py*.
    * Samples the eight corners of a 32 × HITBOX_H UU box around the target’s feet.
    * Projects them into Normalised Device Coordinates (NDC) and then pixels.
    * Computes the tightest axis-aligned rectangle containing the projected
      points (clamped to the viewport).  Returns *None* when **all** points are
      behind the camera frustum.

    Parameters
    ----------
    fov_deg : float
        Vertical field-of-view in **degrees** (⌀).
    screen_w, screen_h : int
        Dimensions of the rendered viewport in **pixels**.
    """

    def __init__(self, fov_deg: float, screen_w: int, screen_h: int) -> None:
        self.v_fov_deg = fov_deg
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.aspect: float = screen_w / screen_h
        self.tan_half_vfov: float = math.tan(math.radians(fov_deg) / 2)

        # Model half-width (constant).
        self.hit_w: float = HITBOX_HALF_W

    # ───────────────────────── Public API ──────────────────────────

    def calcbb(
        self,
        from_tuple: Tuple[float, float, float, float, float, bool],
        to_tuple:   Tuple[float, float, float, float, float, bool],
    ) -> Tuple[int, int, int, int] | None:
        """Project *to_tuple* into the screen space of *from_tuple*’s camera.

        Now adjusts target’s top Z for crouching vs standing.
        """
        px, py, pz, yaw, pitch, from_crouch = from_tuple
        tx, ty, tz, _, _, tgt_crouch = to_tuple

        # Build the POV camera basis (right-handed, CS2 convention)
        eye_z = pz + (CROUCH_Z if from_crouch else STAND_Z)
        pov_eye = Vector3(px, py, eye_z)

        fwd = ypr_to_vec(yaw, pitch).normalize()
        right = fwd.cross(Vector3(0, 0, 1)).normalize() #old
        up = right.cross(fwd)

        # Choose target’s hitbox height: lower if crouching
        tgt_height = CROUCH_Z if tgt_crouch else HITBOX_H

        # Sample the 8 corners of the bounding box surrounding the target
        pts_px: List[Tuple[float, float]] = []
        for dx in (-self.hit_w, self.hit_w):
            for dy in (-self.hit_w, self.hit_w):
                for dz in (0.0, tgt_height):
                    world_pt = Vector3(tx + dx, ty + dy, tz + dz)
                    x_c, y_c, z_c = self._world_to_cam(world_pt, pov_eye, right, up, fwd)
                    x_ndc, y_ndc = self._cam_to_ndc(x_c, y_c, z_c)
                    if x_ndc is None or y_ndc is None:
                        # Corner behind the camera – skip
                        continue
                    px_x, px_y = self._ndc_to_px(x_ndc, y_ndc)
                    pts_px.append((px_x, px_y))

        if not pts_px:
            # All points were behind the camera; nothing to draw.
            return None

        xs, ys = zip(*pts_px)
        xmin = max(0, min(xs))
        ymin = max(0, min(ys))
        xmax = min(self.screen_w, max(xs))
        ymax = min(self.screen_h, max(ys))
        return int(xmin), int(ymin), int(xmax), int(ymax)

    # ───────────────────────── Internal helpers ─────────────────────

    def _world_to_cam(
        self,
        pt: Vector3,
        eye: Vector3,
        r: Vector3,
        u: Vector3,
        f: Vector3,
    ) -> Tuple[float, float, float]:
        """Transform *pt* from world space to the POV’s camera space."""
        d = pt - eye
        return d.dot(r), d.dot(u), d.dot(f)

    def _cam_to_ndc(self, x_c: float, y_c: float, z_c: float) -> Tuple[float | None, float | None]:
        """Perspective-divide camera-space coordinates → NDC."""
        if z_c <= 0:  # behind the near plane
            return None, None
        x_ndc = (x_c / z_c) * (1 / self.aspect) / self.tan_half_vfov
        y_ndc = (y_c / z_c) / self.tan_half_vfov
        return x_ndc, y_ndc

    def _ndc_to_px(self, x_ndc: float, y_ndc: float) -> Tuple[float, float]:
        """Map NDC in [-1, 1]² to actual pixel coordinates."""
        x_px = (x_ndc + 1) * self.screen_w / 2
        y_px = (1 - y_ndc) * self.screen_h / 2
        return x_px, y_px
