from core.vec3 import Vec3
import math
import numpy as np


class Camera:
    def __init__(self, position, look_at, fov_deg, aspect_ratio):
        self.position = position
        self.forward = (look_at - position).normalize()
        self.right = Vec3(0,1,0).cross(self.forward).normalize()
        self.up = self.forward.cross(self.right).normalize()
        self.fov = math.radians(fov_deg)
        self.aspect_ratio = aspect_ratio

    def get_ray(self, px, py, screen_width, screen_height):
        """Compute a single ray direction for pixel (px, py). Original loop-based API.

        Args:
            px: Pixel x-coordinate (column index, 0-based).
            py: Pixel y-coordinate (row index, 0-based).
            screen_width: Image width in pixels.
            screen_height: Image height in pixels.

        Returns:
            Vec3: Normalized ray direction for this pixel.
        """
        # normalize pixel coordinates to [-1,1]
        x = (2 * (px + 0.5)/screen_width - 1) * math.tan(self.fov/2) * self.aspect_ratio
        y = (1 - 2*(py + 0.5)/screen_height) * math.tan(self.fov/2)
        ray_dir = (self.forward + self.right*x + self.up*y).normalize()
        return ray_dir

    def get_rays_vectorized(self, screen_width, screen_height):
        """Compute ALL ray directions for the full image in one vectorized operation.

        Instead of looping over every (x, y) pixel and calling get_ray() 480,000
        times for an 800×600 image, this method uses np.meshgrid to build pixel
        coordinate grids and NumPy broadcasting to compute every ray direction in
        a single batched operation.

        Vectorization strategy:
            1. np.meshgrid creates two 2-D grids of shape (H, W):
               - px_grid[row, col] = col   (x pixel coordinate)
               - py_grid[row, col] = row   (y pixel coordinate)
            2. These grids are mapped to NDC (normalized device coordinates) in
               [-1, 1] using the same formula as get_ray().
            3. The camera basis vectors (forward, right, up) are stored as
               NumPy arrays of shape (3,) and combined via broadcasting:
                   directions = forward + right * ndc_x[..., None] + up * ndc_y[..., None]
               The [..., None] expands (H, W) → (H, W, 1) so that the
               element-wise multiply broadcasts against the (3,) basis vector.
            4. np.linalg.norm with keepdims=True normalizes all H×W direction
               vectors in one call — no per-pixel division.

        Array shapes at each stage:
            px_grid, py_grid : (H, W)        — integer pixel coordinates
            ndc_x, ndc_y     : (H, W)        — floating-point NDC values
            forward_np       : (3,)          — camera basis vector
            right_np, up_np  : (3,)          — camera basis vectors
            directions       : (H, W, 3)    — un-normalized ray directions
            norms            : (H, W, 1)    — per-ray vector lengths
            (return value)   : (H, W, 3)    — normalized ray directions

        Coordinate conventions (unchanged from get_ray):
            - x: NDC maps left-to-right across columns, scaled by aspect_ratio
            - y: NDC maps top-to-bottom (row 0 → +1, row H-1 → -1)
            - The ray direction is forward + right*x + up*y, then normalized

        Args:
            screen_width:  Image width in pixels (W).
            screen_height: Image height in pixels (H).

        Returns:
            np.ndarray: Normalized ray directions, shape (H, W, 3).
                        result[row, col] is the unit direction vector for pixel
                        (col, row), matching get_ray(col, row, W, H).
        """
        H, W = screen_height, screen_width
        half_tan_fov = math.tan(self.fov / 2)

        # --- Step 1: Build pixel coordinate grids ---------------------------
        # np.arange gives [0, 1, ..., W-1] and [0, 1, ..., H-1].
        # indexing='xy' means the first output varies along columns (x) and the
        # second varies along rows (y), giving shapes (H, W) each.
        px_grid, py_grid = np.meshgrid(
            np.arange(W, dtype=np.float64),
            np.arange(H, dtype=np.float64),
            indexing='xy'
        )

        # --- Step 2: Map pixel coords to NDC --------------------------------
        # Exactly mirrors the scalar formula in get_ray():
        #   x = (2*(px+0.5)/W - 1) * tan(fov/2) * aspect_ratio
        #   y = (1 - 2*(py+0.5)/H) * tan(fov/2)
        ndc_x = (2.0 * (px_grid + 0.5) / W - 1.0) * half_tan_fov * self.aspect_ratio  # (H, W)
        ndc_y = (1.0 - 2.0 * (py_grid + 0.5) / H) * half_tan_fov                      # (H, W)

        # --- Step 3: Assemble direction vectors via broadcasting ------------
        # Convert camera basis Vec3 objects to NumPy arrays of shape (3,).
        forward_np = np.array([self.forward.x, self.forward.y, self.forward.z])
        right_np   = np.array([self.right.x,   self.right.y,   self.right.z])
        up_np      = np.array([self.up.x,      self.up.y,      self.up.z])

        # Broadcasting: (3,) + (H,W,1)*(3,) + (H,W,1)*(3,) → (H, W, 3)
        directions = (
            forward_np
            + right_np * ndc_x[..., np.newaxis]
            + up_np    * ndc_y[..., np.newaxis]
        )

        # --- Step 4: Normalize all directions in one vectorized call --------
        norms = np.linalg.norm(directions, axis=2, keepdims=True)  # (H, W, 1)
        directions = directions / norms  # (H, W, 3) — unit ray directions

        return directions