from __future__ import annotations

"""
Default Bounding Box Algorithm — extended with edge‑correction
=============================================================

This module is an updated version of the reference implementation that ships
with *tick_bbxdbg.py*.  In addition to the original corner‑sampling projection,
it now applies a **linear edge‑correction polynomial** to every bounding‑box
coordinate.  The correction compensates for the perspective squeezing that is
particularly noticeable in Counter‑Strike 2 when a target approaches the edge
of the player’s view‑frustum.

The correction works as follows:

*   After the tight AABB has been obtained in pixel space we find the centre of
    that box and convert it to Normalised Device Coordinates (NDC).
*   Horizontal and vertical angles *in degrees* are derived from the centre’s
    NDC coordinates and the viewport field‑of‑view.
*   Every *x* (resp. *y*) coordinate is run through :py:meth:`adjust_x` (resp.
    :py:meth:`adjust_y`).  Both functions implement a simple linear polynomial
    of the form::

        x' = x + k_h * \theta_h
        y' = y - k_v * \theta_v

    with *k_h* and *k_v* being empirically determined constants.  A negative
    horizontal angle therefore moves the box further to the **left**, while a
    positive vertical angle moves it **upwards** — exactly what we need to
    push the box "outwards" away from the centre of the screen.

Feel free to tune *EDGE_CORR_H* and *EDGE_CORR_V* to taste.
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
HITBOX_HALF_W: float = 17 #16.0    # half‑width of the 32×32 UU head/torso box
HITBOX_H: float = 90 #73.0         # total hit‑box height (UU)

# ───────────────────── Edge‑correction coefficients ────────────────────
# Tuned on a 16:9 1920×1080 setup; adjust if you use a very different FOV.
# ───────────────────── Edge-correction coefficients ────────────────────
EDGE_CORR_H: float = 3.45    # pixel-shift per degree horizontally
EDGE_CORR_V: float = 1.75       # pixel-shift per degree vertically

# ───────────────────────── Helper functions ────────────────────────────

def ypr_to_vec(yaw: float, pitch: float) -> Vector3:
    """Convert *yaw* & *pitch* (in **degrees**) to a forward *unit* vector."""
    y_rad = math.radians(yaw)
    p_rad = math.radians(pitch)
    return Vector3(
        math.cos(p_rad) * math.cos(y_rad),  # x
        math.cos(p_rad) * math.sin(y_rad),  # y
        -math.sin(p_rad),                   # z (left‑handed CS2 coordinate system)
    )

# ────────────────────────── Core algorithm ─────────────────────────────

class DefaultBBoxAlg:
    """Corner‑sampling bounding‑box projection **with edge‑correction**.

    Parameters
    ----------
    fov_deg : float
        *Vertical* field‑of‑view in **degrees**.
    screen_w, screen_h : int
        Size of the viewport in **pixels**.
    edge_corr_h, edge_corr_v : float, optional
        Slope of the linear edge‑correction (*pixels per degree*) in **x** and
        **y**.  Defaults come from :pydata:`EDGE_CORR_H` and
        :pydata:`EDGE_CORR_V`.
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

        # Model half‑width (constant).
        self.hit_w: float = HITBOX_HALF_W

        # Edge‑correction slopes
        self.k_h: float = EDGE_CORR_H if edge_corr_h is None else edge_corr_h
        self.k_v: float = EDGE_CORR_V if edge_corr_v is None else edge_corr_v

    # ───────────────────────── Public API ──────────────────────────

    def calcbb(
        self,
        from_tuple: Tuple[float, float, float, float, float, bool],
        to_tuple:   Tuple[float, float, float, float, float, bool],
    ) -> Tuple[int, int, int, int] | None:
        """Project *to_tuple* into *from_tuple*’s screen space.

        Performs exactly the same steps as the legacy implementation **plus** a
        final call to :pymeth:`_apply_edge_correction` immediately before the
        bounding box is returned.
        """

        # Unpack camera & target state ------------------------------------------------
        px, py, pz, yaw, pitch, from_crouch = from_tuple
        tx, ty, tz, _, _, tgt_crouch = to_tuple

        # Build the POV camera basis (right‑handed, CS2 convention) --------------
        eye_z = pz + (CROUCH_Z if from_crouch else STAND_Z)
        pov_eye = Vector3(px, py, eye_z)

        fwd = ypr_to_vec(yaw, pitch).normalize()
        right = fwd.cross(Vector3(0, 0, 1)).normalize()
        up = right.cross(fwd)

        # Hit‑box height depends on stance ----------------------------------------
        tgt_height = CROUCH_Z if tgt_crouch else HITBOX_H

        # Sample the 8 corners of the bounding‑box -------------------------------
        pts_px: List[Tuple[float, float]] = []
        for dx in (-self.hit_w, self.hit_w):
            for dy in (-self.hit_w, self.hit_w):
                for dz in (0.0, tgt_height):
                    world_pt = Vector3(tx + dx, ty + dy, tz + dz)
                    x_c, y_c, z_c = self._world_to_cam(world_pt, pov_eye, right, up, fwd)
                    x_ndc, y_ndc = self._cam_to_ndc(x_c, y_c, z_c)
                    if x_ndc is None or y_ndc is None:
                        # Corner is behind the near plane → skip
                        continue
                    px_x, px_y = self._ndc_to_px(x_ndc, y_ndc)
                    pts_px.append((px_x, px_y))

        if not pts_px:
            # All eight corners ended up behind the camera → nothing to draw
            return None

        # Tight AABB in pixel space ---------------------------------------------
        xs, ys = zip(*pts_px)
        xmin = max(0.0, min(xs))
        ymin = max(0.0, min(ys))
        xmax = min(self.screen_w, max(xs))
        ymax = min(self.screen_h, max(ys))

        # Apply edge‑correction *before* integer cast ---------------------------
        xmin, ymin, xmax, ymax = self._apply_edge_correction(xmin, ymin, xmax, ymax)

        # Clamp & return as ints -------------------------------------------------
        xmin_i = int(max(0, min(self.screen_w, xmin)))
        ymin_i = int(max(0, min(self.screen_h, ymin)))
        xmax_i = int(max(0, min(self.screen_w, xmax)))
        ymax_i = int(max(0, min(self.screen_h, ymax)))
        return xmin_i, ymin_i, xmax_i, ymax_i

    # ──────────────────────── Internal helpers ─────────────────────────

    def _world_to_cam(
        self,
        pt: Vector3,
        eye: Vector3,
        r: Vector3,
        u: Vector3,
        f: Vector3,
    ) -> Tuple[float, float, float]:
        """Transform *pt* from world to camera space."""
        d = pt - eye
        return d.dot(r), d.dot(u), d.dot(f)

    def _cam_to_ndc(
        self,
        x_c: float,
        y_c: float,
        z_c: float,
    ) -> Tuple[float | None, float | None]:
        """Perspective‑divide camera‑space coordinates to NDC."""
        if z_c <= 0:  # behind near plane
            return None, None
        x_ndc = (x_c / z_c) * (1 / self.aspect) / self.tan_half_vfov
        y_ndc = (y_c / z_c) / self.tan_half_vfov
        return x_ndc, y_ndc

    def _ndc_to_px(self, x_ndc: float, y_ndc: float) -> Tuple[float, float]:
        """Map NDC ∈ [−1, 1]² to pixel coordinates."""
        x_px = (x_ndc + 1) * self.screen_w / 2
        y_px = (1 - y_ndc) * self.screen_h / 2
        return x_px, y_px

    # ───────────────────── Edge‑correction helpers ────────────────────

    def _apply_edge_correction(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> Tuple[float, float, float, float]:
        """Shift the bounding‑box *outwards* based on its centre angle."""
        # Centre of the box in pixel coordinates
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        # Convert centre to NDC
        x_ndc = (2 * cx / self.screen_w) - 1.0                       # ∈ [−1, 1]
        y_ndc = 1.0 - (2 * cy / self.screen_h)                       # ∈ [−1, 1]

        # Derive horizontal / vertical angles (degrees)
        theta_h = x_ndc * (self.h_fov_deg / 2)
        theta_v = y_ndc * (self.v_fov_deg / 2)

        # Apply the linear polynomials
        xmin = self.adjust_x(theta_h, xmin)
        xmax = self.adjust_x(theta_h, xmax)
        ymin = self.adjust_y(theta_v, ymin)
        ymax = self.adjust_y(theta_v, ymax)

        return xmin, ymin, xmax, ymax

    # Public so they can be monkey‑patched during experiments ------------
    def adjust_x(self, theta_h: float, x_px: float) -> float:
        """Linear horizontal edge‑correction.

        Negative *theta_h* ⇒ shift further **left** (subtract).  Positive ⇒
        shift further **right** (add).
        """
        return x_px + self.k_h * theta_h

    def adjust_y(self, theta_v: float, y_px: float) -> float:
        """Linear vertical edge‑correction.

        Positive *theta_v* (above screen‑centre) ⇒ shift **upwards** (subtract
        from *y* because pixel‑space grows *downwards*).  Negative ⇒ shift
        **downwards**.
        """
        return y_px - self.k_v * theta_v
