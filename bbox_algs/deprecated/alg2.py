import math
from typing import Tuple, List, Optional


class BoundingBoxCS2:
    """
    Projects a 32×73 UU (standing) or 32×54 UU (crouch) hit‐box,
    given feet‐level world coordinates, into 2D screen‐space.

    ────────────────────────────────────────────────────────────────────
    Public constructor and methods (unchanged signature):
        __init__(fov_deg, screen_w, screen_h, *, fov_axis="horizontal", scale_x=1.0, scale_y=1.0)
        calcbb(from_tuple, to_tuple) -> (xmin, ymin, xmax, ymax) | None
    ────────────────────────────────────────────────────────────────────
    
    Assumptions:
      • +X = forward, +Y = left, +Z = up (Source‐engine convention)
      • yaw=0 → +X, positive yaw rotates CCW (toward +Y)
      • pitch=0 → level, +90° → straight down, -90° → straight up
      • Input “from_tuple = (px, py, pz, yaw, pitch, crouch)” uses (px,py,pz) at the PLAYER’S FEET.
      • We add eye‐height = 64 UU (standing) or 46 UU (crouched) to pz to get the camera location.
      • Input “to_tuple = (tx, ty, tz, tyaw, tpitch, tcrouch)” uses (tx,ty,tz) at the TARGET’S FEET.
      • Standing hit‐box = 32×32×72 UU, crouch hit‐box = 32×32×54 UU, directly on top of tz (no origin drop).
      • The FOV passed in is treated by default as a HORIZONTAL FOV (Source / CS2 convention).
      • scale_x, scale_y default to 1.0 (no post‐projection scaling), but you can tweak them.
    """

    # ────────────────────────────────────────────────────────────────────────
    # Tunable / “constant” values
    # ────────────────────────────────────────────────────────────────────────
    HULL_W          = 32.0   # hit‐box width & depth
    HULL_H_STAND    = 72.0   # hit‐box height when standing
    HULL_H_CROUCH   = 54.0   # hit‐box height when crouching

    EYE_H_STAND     = 64.0   # eye above feet when standing
    EYE_H_CROUCH    = 46.0   # eye above feet when crouching
    # ────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        fov_deg: float,
        screen_w: int,
        screen_h: int,
        *,
        fov_axis: str = "horizontal",  # treat fov_deg as "horizontal" by default
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        fov_deg   : Field of view in degrees (SOURCE/CS2 uses HORIZONTAL FOV by default).
        screen_w  : Framebuffer width (pixels).
        screen_h  : Framebuffer height (pixels).
        fov_axis  : "horizontal" or "vertical" → which axis fov_deg refers to.
        scale_x   : (optional) post‐projection x‐scale on the box width (default 1.0).
        scale_y   : (optional) post‐projection y‐scale on the box height (default 1.0).
        """
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.aspect   = self.screen_w / self.screen_h

        if fov_axis not in {"horizontal", "vertical"}:
            raise ValueError("fov_axis must be 'horizontal' or 'vertical'")

        # ───────────────────────────────────────────────────────────────
        # Compute fx, fy (pixel‐space focal lengths) from whichever FOV axis.
        # For a pinhole camera:
        #   fx = (w/2) / tan(hfov/2)
        #   fy = (h/2) / tan(vfov/2)
        #
        # If user gave horizontal FOV:
        #   hfov_rad = radians(fov_deg)
        #   fx = (w/2) / tan(hfov/2)
        #   vfov_rad = 2 * atan(tan(hfov/2) * (h/w))
        #   fy = (h/2) / tan(vfov/2)
        #
        # If user gave vertical FOV:
        #   vfov_rad = radians(fov_deg)
        #   fy = (h/2) / tan(vfov/2)
        #   hfov_rad = 2 * atan(tan(vfov/2) * (w/h))
        #   fx = (w/2) / tan(hfov/2)
        # ───────────────────────────────────────────────────────────────
        if fov_axis == "horizontal":
            hfov_rad = math.radians(fov_deg)
            self.fx = (self.screen_w / 2.0) / math.tan(hfov_rad / 2.0)
            # derive vertical FOV from aspect ratio
            vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * (1.0 / self.aspect))
            self.fy = (self.screen_h / 2.0) / math.tan(vfov_rad / 2.0)
        else:
            vfov_rad = math.radians(fov_deg)
            self.fy = (self.screen_h / 2.0) / math.tan(vfov_rad / 2.0)
            hfov_rad = 2.0 * math.atan(math.tan(vfov_rad / 2.0) * self.aspect)
            self.fx = (self.screen_w / 2.0) / math.tan(hfov_rad / 2.0)

        # post‐projection scaling factors
        self.scale_x = float(scale_x)
        self.scale_y = float(scale_y)

        # half‐width of the hull (X and Y extents are ± half of this)
        self._half_w = self.HULL_W / 2.0

    # ────────────────────────────────────────────────────────────────────────
    # Public method
    # ────────────────────────────────────────────────────────────────────────
    def calcbb(
        self,
        from_tuple: Tuple[float, float, float, float, float, bool],
        to_tuple:   Tuple[float, float, float, float, float, bool],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Project the 3D hit‐box of a target into 2D screen coordinates.

        Parameters
        ----------
        from_tuple = (px, py, pz, yaw, pitch, crouch) for the CAMERA:
          • px, py, pz : world coords of the *feet*.  We add eye‐height internally.
          • yaw, pitch : view angles in degrees.
          • crouch      : True or False.

        to_tuple = (tx, ty, tz, tyaw, tpitch, tcrouch) for the TARGET:
          • tx, ty, tz : world coords of the *feet*.  We build a 72 UU (standing) or 54 UU (crouch) box on top.
          • tyaw, tpitch: target’s view angles (ignored for box).
          • tcrouch    : True if target is crouched, else False.

        Returns
        -------
        (xmin, ymin, xmax, ymax): pixel‐coordinate bounding box, clamped to [0..screen_w]×[0..screen_h],
                                 or None if *all* 8 corners fall at z_cam ≤ 0.
        """
        # ──────── 1. Compute camera‐eye position in world space ─────────
        px, py, pz, yaw_deg, pitch_deg, cam_crouch = from_tuple

        # Since (px,py,pz) is at the feet, add eye‐height so camera is at eye:
        cam_z = pz + (self.EYE_H_CROUCH if cam_crouch else self.EYE_H_STAND)
        cam_pos = (px, py, cam_z)

        # Build camera‐space axes (right, up, forward) from yaw/pitch
        right, up, fwd = self._build_axes(yaw_deg, pitch_deg)

        # ──────── 2. Enumerate the 8 corners of the target’s hull ────────
        tx, ty, tz, _, _, tgt_crouch = to_tuple

        # Height of the hit‐box
        hull_h = self.HULL_H_CROUCH if tgt_crouch else self.HULL_H_STAND

        # Since tz is the target’s feet‐position, base_z = tz exactly
        base_z = tz

        # Build 8 world‐space corner points: (±half_w, ±half_w, 0 or hull_h)
        corners_world: List[Tuple[float, float, float]] = []
        for dx in (-self._half_w, self._half_w):
            for dy in (-self._half_w, self._half_w):
                for dz in (0.0, hull_h):
                    corners_world.append((tx + dx, ty + dy, base_z + dz))

        # ──────── 3. Project each corner that’s in front of the camera ───
        xs, ys = [], []
        for wx, wy, wz in corners_world:
            x_cam, y_cam, z_cam = self._to_cam_space(
                (wx, wy, wz), cam_pos, right, up, fwd
            )
            # If z_cam ≤ 0, that corner is behind the camera plane → skip
            if z_cam <= 0.0:
                continue

            # Standard pinhole projection:
            u = (x_cam * self.fx) / z_cam + (self.screen_w  / 2.0)
            v = (-y_cam * self.fy) / z_cam + (self.screen_h / 2.0)
            xs.append(u)
            ys.append(v)

        # If NO corner was in front, return None
        if not xs or not ys:
            return None

        # ──────── 4. Build axis‐aligned 2D bbox around all projected points ─
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Apply optional post‐projection scaling about the center
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        w_scaled = (xmax - xmin) * self.scale_x
        h_scaled = (ymax - ymin) * self.scale_y

        xmin_s = cx - w_scaled / 2.0
        xmax_s = cx + w_scaled / 2.0
        ymin_s = cy - h_scaled / 2.0
        ymax_s = cy + h_scaled / 2.0

        # ──────── 5. Clamp to [0..screen_w]×[0..screen_h] & return ───────
        xmin_i = max(0, min(self.screen_w,  int(round(xmin_s))))
        xmax_i = max(0, min(self.screen_w,  int(round(xmax_s))))
        ymin_i = max(0, min(self.screen_h,  int(round(ymin_s))))
        ymax_i = max(0, min(self.screen_h,  int(round(ymax_s))))

        # If degenerately inverted (or zero‐area), treat as “no box”
        if xmin_i >= xmax_i or ymin_i >= ymax_i:
            return None

        return (xmin_i, ymin_i, xmax_i, ymax_i)

    # ────────────────────────────────────────────────────────────────────────
    # Internals: build camera axes & transform world→camera space
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_axes(yaw_deg: float, pitch_deg: float):
        """
        Build a Source‐engine style (right, up, forward) orthonormal basis
        from yaw/pitch. Conventions:
         • +X = forward, +Y = left, +Z = up
         • yaw=0 => +X, positive yaw rotates CCW toward +Y
         • pitch=0 => level, +90 => straight down, -90 => straight up
        """
        y_rad = math.radians(yaw_deg)
        p_rad = math.radians(pitch_deg)

        cp, sp = math.cos(p_rad), math.sin(p_rad)
        cy, sy = math.cos(y_rad), math.sin(y_rad)

        fwd   = ( cp * cy,   cp * sy,   -sp   )
        right = ( -sy,       cy,         0.0  )
        up    = ( sp * cy,   sp * sy,    cp   )
        return right, up, fwd

    @staticmethod
    def _to_cam_space(
        world_pt: Tuple[float, float, float],
        cam_pos: Tuple[float, float, float],
        right: Tuple[float, float, float],
        up:    Tuple[float, float, float],
        fwd:   Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Convert a world-space point to camera-space (x_cam, y_cam, z_cam).
        camera‐space coords are defined so that:
         • +z_cam = “in front of” the camera
         • +x_cam = to the camera’s right
         • +y_cam = up from the camera
        """
        dx = world_pt[0] - cam_pos[0]
        dy = world_pt[1] - cam_pos[1]
        dz = world_pt[2] - cam_pos[2]

        x_cam = dx * right[0] + dy * right[1] + dz * right[2]
        y_cam = dx * up[0]    + dy * up[1]    + dz * up[2]
        z_cam = dx * fwd[0]   + dy * fwd[1]   + dz * fwd[2]
        return x_cam, y_cam, z_cam
