import math
from typing import Tuple, List, Optional


class BoundingBoxCS2:
    """
    Field-of-view-based 3-D → 2-D hit-box projector for Source-style coordinates.

    World-space conventions assumed
    --------------------------------
    • +X  = forward                   (yaw 0° looks along +X)
    • +Y  = left  (so +yaw = CCW)     (same as Source)
    • +Z  = up
    • pitch:  0° = level, +90° look straight down, -90° up
    • Player origin sits at the *centre of the feet*.
    • Standing hull  : 32 × 32 × 72 UU
      Crouching hull : 32 × 32 × 54 UU
    • Eye heights    : 64 UU (standing), 46 UU (crouch)

    Parameters exposed for later fine-tuning
    ----------------------------------------
    • scale_x, scale_y (≥0): post-projection multipliers applied
      to the computed box width/height around its centre.
    """

    # --- constants you might fine-tune later -------------------------------
    HULL_W            = 32.0               # width & depth (square)
    HULL_H_STAND      = 72.0
    HULL_H_CROUCH     = 54.0
    EYE_H_STAND       = 64.0
    EYE_H_CROUCH      = 46.0
    # ----------------------------------------------------------------------

    def __init__(
        self,
        fov_deg: float,
        screen_w: int,
        screen_h: int,
        *,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)

        self.vfov_rad = math.radians(fov_deg)
        # vertical-fov-based focal length (pixels)
        self.f = (self.screen_h / 2.0) / math.tan(self.vfov_rad / 2.0)

        # optional post-projection scale factors
        self.scale_x = float(scale_x)
        self.scale_y = float(scale_y)

        # pre-compute half-width
        self._half_w = self.HULL_W / 2.0

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def calcbb(
        self,
        from_tuple: Tuple[float, float, float, float, float, bool],
        to_tuple:   Tuple[float, float, float, float, float, bool],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Return (xmin, ymin, xmax, ymax) in pixel coordinates, clamped to
        the screen rectangle, or None if the entire hit-box is behind the
        camera frustum.
        """
        # ------------- unpack & derive camera parameters -----------------
        px, py, pz, yaw_deg, pitch_deg, crouch_cam = from_tuple
        eye_z = pz + (self.EYE_H_CROUCH if crouch_cam else self.EYE_H_STAND)
        cam_pos = (px, py, eye_z)

        # camera axes (right, up, forward)
        right, up, fwd = self._build_axes(yaw_deg, pitch_deg)

        # ------------- enumerate target-hull corners ---------------------
        tx, ty, tz, _, _, crouch_tgt = to_tuple
        hull_h = self.HULL_H_CROUCH if crouch_tgt else self.HULL_H_STAND

        # generate the 8 corner points (feet-centred)
        corners_world: List[Tuple[float, float, float]] = []
        for dx in (-self._half_w, self._half_w):
            for dy in (-self._half_w, self._half_w):
                for dz in (0.0, hull_h):
                    corners_world.append((tx + dx, ty + dy, tz + dz))

        # ------------- project visible corners ---------------------------
        xs, ys = [], []
        for wx, wy, wz in corners_world:
            x_cam, y_cam, z_cam = self._to_cam_space(
                (wx, wy, wz), cam_pos, right, up, fwd
            )
            if z_cam <= 0.0:            # behind or on the camera plane
                continue

            u = (x_cam * self.f) / z_cam + self.screen_w / 2.0
            v = (-y_cam * self.f) / z_cam + self.screen_h / 2.0
            xs.append(u)
            ys.append(v)

        if not xs or not ys:            # nothing in front of the camera
            return None

        # ------------- build & scale the bounding box --------------------
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        w_scaled = (xmax - xmin) * self.scale_x
        h_scaled = (ymax - ymin) * self.scale_y

        xmin_s = cx - w_scaled / 2.0
        xmax_s = cx + w_scaled / 2.0
        ymin_s = cy - h_scaled / 2.0
        ymax_s = cy + h_scaled / 2.0

        # ------------- clamp to screen & finish --------------------------
        xmin_i = max(0, min(self.screen_w, int(round(xmin_s))))
        xmax_i = max(0, min(self.screen_w, int(round(xmax_s))))
        ymin_i = max(0, min(self.screen_h, int(round(ymin_s))))
        ymax_i = max(0, min(self.screen_h, int(round(ymax_s))))

        if xmin_i >= xmax_i or ymin_i >= ymax_i:  # degenerate box
            return None
        return (xmin_i, ymin_i, xmax_i, ymax_i)

    # ------------------------------------------------------------------ #
    #  Internals                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_axes(yaw_deg: float, pitch_deg: float):
        """Return (right, up, forward) unit vectors for the given yaw/pitch."""
        y_rad = math.radians(yaw_deg)
        p_rad = math.radians(pitch_deg)

        cp, sp = math.cos(p_rad), math.sin(p_rad)
        cy, sy = math.cos(y_rad), math.sin(y_rad)

        # Source-style forward/right/up construction
        fwd = (cp * cy, cp * sy, -sp)
        right = (-sy, cy, 0.0)
        up = (sp * cy, sp * sy, cp)
        return right, up, fwd

    @staticmethod
    def _to_cam_space(
        world_pt,
        cam_pos,
        right, up, fwd,
    ):
        """Convert a world-space point to (x,y,z) in camera coordinates."""
        dx = world_pt[0] - cam_pos[0]
        dy = world_pt[1] - cam_pos[1]
        dz = world_pt[2] - cam_pos[2]

        x_cam = dx * right[0] + dy * right[1] + dz * right[2]
        y_cam = dx * up[0]    + dy * up[1]    + dz * up[2]
        z_cam = dx * fwd[0]   + dy * fwd[1]   + dz * fwd[2]
        return x_cam, y_cam, z_cam
