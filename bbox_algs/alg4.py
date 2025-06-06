from __future__ import annotations
from typing import Tuple, List, Optional

import math
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# Toggle debugging visualization here:
DEBUG = False


class CS2BBox:
    """
    Camera-space 3-D → 2-D bounding-box projector for Counter-Strike 2 / Source 2.

    Parameters
    ----------
    fov_deg : float
        *Vertical* field-of-view in degrees (e.g. 90.0).
    screen_w, screen_h : int
        Frame buffer size in pixels.

    Behavior
    --------
    When DEBUG is True (at the module level), calcbb() will pop up an OpenCV window
    showing:
      1) the projected 3-D hit-box edges (in red),
      2) the projected corner points (in blue),
      3) and the resulting 2-D bounding rectangle (in green).
    The background will be white to ensure good contrast.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Static player constants (UU = “Source units”)
    HULL_WIDTH    = 32.0            # ±16 UU on X/Y axes
    HEIGHT_STAND  = 73.0            # UU top of hit-box when standing
    HEIGHT_CROUCH = 54.0            # UU top when crouched
    EYE_STAND     = 72.0            # UU from feet to POV when standing
    EYE_CROUCH    = 54.0            # UU from feet to POV when crouched
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self, fov_deg: float, screen_w: int, screen_h: int) -> None:
        if DEBUG and cv2 is None:
            raise ImportError("DEBUG=True requires OpenCV (cv2) to be installed.")

        self.fov_y = math.radians(fov_deg)
        self.w = screen_w
        self.h = screen_h
        self.aspect = screen_w / screen_h
        self.tan_y = math.tan(self.fov_y / 2)
        self.tan_x = self.tan_y * self.aspect
        self.debug = DEBUG

    def calcbb(
        self,
        from_tuple: Tuple[float, float, float, float, float, bool],
        to_tuple:   Tuple[float, float, float, float, float, bool],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Project the target’s 3-D hit-box to screen space.

        Returns
        -------
        (xmin, ymin, xmax, ymax) in pixel coords, or None if every corner is
        behind the camera. When DEBUG is True, a window labeled "CS2BBox Debug"
        will show:
          - Projected hit-box edges (red)
          - Projected corner points (blue)
          - 2-D bounding rectangle (green)
          on a white background.
        """
        # ── Unpack inputs ───────────────────────────────────────────────────
        px, py, pz, yaw_deg, pitch_deg, crouch = from_tuple
        tx, ty, tz, _, _, t_crouch = to_tuple

        # Compute camera (eye) position:
        eye_height = self.EYE_CROUCH if crouch else self.EYE_STAND
        cam_pos = np.array([px, py, pz + eye_height])

        # Convert angles to radians:
        yaw, pitch = map(math.radians, (yaw_deg, pitch_deg))
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw),   math.sin(yaw)

        # Build camera basis vectors using Source AngleVectors convention:
        forward = np.array([cp * cy, cp * sy, -sp])
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            # Camera looking straight up or down: choose alternate up
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        # ── Build target AABB corners ───────────────────────────────────────
        half = self.HULL_WIDTH / 2.0
        h_max = self.HEIGHT_CROUCH if t_crouch else self.HEIGHT_STAND

        # Corner ordering:
        # 0: (-half, -half, 0)
        # 1: (-half, -half, h_max)
        # 2: (-half,  half, 0)
        # 3: (-half,  half, h_max)
        # 4: ( half, -half, 0)
        # 5: ( half, -half, h_max)
        # 6: ( half,  half, 0)
        # 7: ( half,  half, h_max)
        corners: List[np.ndarray] = [
            np.array([tx + dx, ty + dy, tz + dz])
            for dx in (-half, half)
            for dy in (-half, half)
            for dz in (0.0, h_max)
        ]

        # Prepare storage for projected 2-D points (None if behind camera)
        proj_points: List[Optional[tuple[int, int]]] = [None] * len(corners)

        for idx, corner in enumerate(corners):
            vec = corner - cam_pos
            z_cam = float(np.dot(vec, forward))
            if z_cam <= 0.0:
                # Corner is behind camera
                continue

            x_cam = float(np.dot(vec, right))
            y_cam = float(np.dot(vec, up))

            # Normalize to NDC (−1…+1)
            x_ndc = (x_cam / z_cam) / self.tan_x
            y_ndc = (y_cam / z_cam) / self.tan_y

            # Convert NDC to pixel coords
            px_scr = int((x_ndc + 1.0) * 0.5 * self.w)
            py_scr = int((1.0 - y_ndc) * 0.5 * self.h)

            proj_points[idx] = (px_scr, py_scr)

        # Collect only the visible (in-front) coords for bounding-box clamping
        visible_pts = [p for p in proj_points if p is not None]
        if not visible_pts:
            return None

        xs = [p[0] for p in visible_pts]
        ys = [p[1] for p in visible_pts]
        xmin = max(0,      min(xs))
        ymin = max(0,      min(ys))
        xmax = min(self.w, max(xs))
        ymax = min(self.h, max(ys))
        bbox = (xmin, ymin, xmax, ymax)

        # ── Optional debug frame ────────────────────────────────────────────
        if self.debug:
            # White background
            dbg_img = np.ones((self.h, self.w, 3), np.uint8) * 255

            # Draw projected edges of the 3-D hit-box (red)
            edges = [
                # bottom face
                (0, 2), (2, 6), (6, 4), (4, 0),
                # top face
                (1, 3), (3, 7), (7, 5), (5, 1),
                # vertical edges
                (0, 1), (2, 3), (4, 5), (6, 7),
            ]
            for i0, i1 in edges:
                p0 = proj_points[i0]
                p1 = proj_points[i1]
                if p0 is not None and p1 is not None:
                    cv2.line(dbg_img, p0, p1, (0, 0, 255), 1)

            # Draw projected corner points (blue)
            for p in visible_pts:
                cv2.circle(dbg_img, p, 3, (255, 0, 0), -1)

            # Draw 2-D bounding rectangle (green)
            cv2.rectangle(dbg_img, (xmin, ymin), (xmax, ymax),
                          (0, 255, 0), 2)

            # Label the bbox
            text_pos = (xmin, max(15, ymin - 10))
            cv2.putText(dbg_img, f"{bbox}", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.imshow("CS2BBox Debug", dbg_img)
            cv2.waitKey(1)  # 1ms delay to refresh

        return bbox
