import numpy as np


# ---------------------------------------------------------------------------
# Cubemap face definitions for the envmap baker
#
# Each face is rendered with a 90° FOV camera looking along the forward
# direction (D).  Pixel (i,j) maps to the 3D direction:
#   normalize(D + uc*R + vc*U)
# where (uc, vc) are normalized image coordinates in [-1, +1] and
# R = right, U = up.
# ---------------------------------------------------------------------------

_FACE_PARAMS = {
    'px': {'D': np.array([ 1,  0,  0], dtype=np.float32),
           'R': np.array([ 0,  0, -1], dtype=np.float32),
           'U': np.array([ 0, -1,  0], dtype=np.float32)},
    'nx': {'D': np.array([-1,  0,  0], dtype=np.float32),
           'R': np.array([ 0,  0,  1], dtype=np.float32),
           'U': np.array([ 0, -1,  0], dtype=np.float32)},
    'py': {'D': np.array([ 0,  1,  0], dtype=np.float32),
           'R': np.array([ 1,  0,  0], dtype=np.float32),
           'U': np.array([ 0,  0,  1], dtype=np.float32)},
    'ny': {'D': np.array([ 0, -1,  0], dtype=np.float32),
           'R': np.array([ 1,  0,  0], dtype=np.float32),
           'U': np.array([ 0,  0, -1], dtype=np.float32)},
    'pz': {'D': np.array([ 0,  0,  1], dtype=np.float32),
           'R': np.array([ 1,  0,  0], dtype=np.float32),
           'U': np.array([ 0, -1,  0], dtype=np.float32)},
    'nz': {'D': np.array([ 0,  0, -1], dtype=np.float32),
           'R': np.array([-1,  0,  0], dtype=np.float32),
           'U': np.array([ 0, -1,  0], dtype=np.float32)},
}
_FACE_NAMES = ['px', 'nx', 'py', 'ny', 'pz', 'nz']


def _equirect_direction(lon, lat):
    """Convert longitude/latitude grids to world-space direction vectors (Z-up).

    Args:
        lon: (H, W) float32 array, longitude in [-pi, pi], azimuth around Z
        lat: (H, W) float32 array, latitude in [-pi/2, pi/2], elevation from XY plane

    Returns:
        (H, W, 3) float32 array of unit direction vectors
    """
    cos_lat = np.cos(lat)
    dx = cos_lat * np.sin(lon)
    dy = cos_lat * np.cos(lon)
    dz = np.sin(lat)
    return np.stack([dx, dy, dz], axis=-1)


# ---------------------------------------------------------------------------
# Cubemap sampling helpers
# ---------------------------------------------------------------------------

def _cubemap_face_for_direction(dirs):
    """Find the best cubemap face for each direction vector.

    Args:
        dirs: (N, 3) float32 array of unit direction vectors.

    Returns:
        (N,) int32 array of face indices (0-5), and (N,) float32 of the dot product.
    """
    N = dirs.shape[0]
    best_face = np.zeros(N, dtype=np.int32)
    best_dot = np.full(N, -1.0, dtype=np.float32)
    for fi, name in enumerate(_FACE_NAMES):
        D = _FACE_PARAMS[name]['D']
        dot = np.dot(dirs, D)
        mask = dot > best_dot
        best_face[mask] = fi
        best_dot[mask] = dot[mask]
    return best_face, best_dot


def _sample_cubemap(faces, dirs):
    """Bilinearly sample cubemap faces at given directions.

    Args:
        faces: dict of {name: (res, res, 4) float32}
        dirs: (N, 3) float32 unit direction vectors

    Returns:
        (N, 4) float32 RGBA values
    """
    res = faces['px'].shape[0]
    N = dirs.shape[0]
    result = np.zeros((N, 4), dtype=np.float32)

    best_face, best_dot = _cubemap_face_for_direction(dirs)

    for fi, name in enumerate(_FACE_NAMES):
        p = _FACE_PARAMS[name]
        mask = best_face == fi
        if not mask.any():
            continue

        dD = best_dot[mask]
        dirs_m = dirs[mask]
        uc = np.dot(dirs_m, p['R']) / dD
        vc = np.dot(dirs_m, p['U']) / dD

        col_f = (uc + 1.0) * 0.5 * (res - 1)
        row_f = (vc + 1.0) * 0.5 * (res - 1)

        col_f = np.clip(col_f, 0.0, float(res - 1))
        row_f = np.clip(row_f, 0.0, float(res - 1))

        col0 = col_f.astype(np.int32)
        col1 = np.minimum(col0 + 1, res - 1)
        row0 = row_f.astype(np.int32)
        row1 = np.minimum(row0 + 1, res - 1)

        fx = col_f - col0.astype(np.float32)
        fy = row_f - row0.astype(np.float32)

        img = faces[name]
        top_l = img[row0, col0]
        top_r = img[row0, col1]
        bot_l = img[row1, col0]
        bot_r = img[row1, col1]

        top = top_l + (top_r - top_l) * fx[:, None]
        bot = bot_l + (bot_r - bot_l) * fx[:, None]
        result[mask] = top + (bot - top) * fy[:, None]

    return result


# ---------------------------------------------------------------------------
# Specular convolution (pre-filtered envmap via GGX importance sampling)
# ---------------------------------------------------------------------------

def _ggx_local_samples(roughness, num_samples, seed=0):
    """Generate GGX importance-sampled directions in tangent space.

    Returns (num_samples, 3) where Z is the "center direction".  The samples
    are deterministic for a given seed so the convolution is flicker-free.
    """
    rng = np.random.default_rng(seed)
    xi1 = rng.random(num_samples).astype(np.float32)
    xi2 = rng.random(num_samples).astype(np.float32)

    alpha = max(roughness * roughness, 1e-6)
    theta = np.arctan2(alpha * np.sqrt(xi1), np.sqrt(1.0 - xi1))
    phi = 2.0 * np.pi * xi2

    st = np.sin(theta)
    return np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def _sample_dirs_from_center(local_samples, centers):
    """Rotate local tangent-space sample directions to align with each center.

    Args:
        local_samples: (S, 3) local directions where Z = center
        centers: (N, 3) center direction vectors

    Returns:
        (N, S, 3) world-space direction vectors
    """
    N = centers.shape[0]
    S = local_samples.shape[0]

    all_samples = np.empty((N, S, 3), dtype=np.float32)
    for i in range(N):
        cz = centers[i]
        # Build tangent frame (right, up, center)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(cz, up)) > 0.999:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        rx = np.cross(up, cz)
        rx /= np.linalg.norm(rx)
        ry = np.cross(cz, rx)
        ry /= np.linalg.norm(ry)

        all_samples[i] = (local_samples[:, 0:1] * rx +
                          local_samples[:, 1:2] * ry +
                          local_samples[:, 2:3] * cz)

    norms = np.linalg.norm(all_samples, axis=2, keepdims=True)
    all_samples /= np.maximum(norms, 1e-8)
    return all_samples


def generate_specular_mip_chain(faces, probe_w, levels=6, samples_per_pixel=256):
    """Generate a roughness mip chain via GGX specular convolution.

    Each level convolves the original cubemap with the GGX distribution at
    increasing roughness in *direction space*, correctly handling spherical
    geometry and cubemap face boundaries.

    Args:
        faces: dict of {name: (res, res, 4) float32}
        probe_w: width of level-0 equirectangular output (pixels)
        levels: number of mip levels
        samples_per_pixel: GGX importance samples per output pixel

    Returns:
        list of (H_i, W_i, 4) equirectangular images, one per mip level.
    """
    chain = []

    for level in range(levels):
        roughness = level / max(levels - 1, 1)
        eq_w = max(probe_w >> level, 1)
        eq_h = eq_w // 2

        print(f"[Envmap] Convolving level {level}  roughness={roughness:.3f}  "
              f"{eq_w}×{eq_h}  samples={samples_per_pixel}")

        # Equirectangular direction grid
        lon = np.linspace(-np.pi, np.pi, eq_w, dtype=np.float32)
        lat = np.linspace(np.pi / 2, -np.pi / 2, eq_h, dtype=np.float32)
        lon2d, lat2d = np.meshgrid(lon, lat)
        cos_lat = np.cos(lat2d)
        dirs = np.stack([
            cos_lat * np.sin(lon2d),  # X
            cos_lat * np.cos(lon2d),  # Y
            np.sin(lat2d),             # Z (pole)
        ], axis=-1)  # (eq_h, eq_w, 3)

        if level == 0:
            # Level 0 = sharp: single cubemap lookup per pixel
            flat = dirs.reshape(-1, 3)
            colors = _sample_cubemap(faces, flat)
            eq = colors.reshape(eq_h, eq_w, 4)
        else:
            # GGX convolution
            flat = dirs.reshape(-1, 3)
            N = flat.shape[0]

            # Pre-generate GGX sample pattern for this roughness
            local_samples = _ggx_local_samples(roughness, samples_per_pixel, seed=level)

            # Process in batches to keep memory manageable
            batch = 4096
            eq = np.zeros((eq_h * eq_w, 4), dtype=np.float32)

            for start in range(0, N, batch):
                end = min(start + batch, N)
                batch_centers = flat[start:end]
                world = _sample_dirs_from_center(local_samples, batch_centers)
                # (B, S, 3) → (B*S, 3)
                B = end - start
                world_flat = world.reshape(B * samples_per_pixel, 3)
                colors = _sample_cubemap(faces, world_flat)
                colors = colors.reshape(B, samples_per_pixel, 4)
                eq[start:end] = colors.mean(axis=1)

            eq = eq.reshape(eq_h, eq_w, 4)

        chain.append(eq)
        print(f"[Envmap] Level {level} done")

    return chain



# ---------------------------------------------------------------------------
# Atlas packing — vertical stacking of mip levels
# ---------------------------------------------------------------------------

def pack_mip_atlas(mip_chain):
    """Pack mip levels into a vertically stacked image with horizontal tiling.

    Each mip level is flipped vertically then tiled horizontally so that every
    row has the same width (the width of level 0).  Level 0 is not tiled (1
    copy).  Level 1 is tiled to 2 copies, level 2 to 4 copies, etc.

    Level 0 sits at the top of the atlas, level 1 below it, etc.

    Args:
        mip_chain: list of (H_i, W_i, 4) float32 arrays.

    Returns:
        (total_h, common_w, 4) float32 RGBA array, and (common_w, total_h) tuple.
    """
    common_w = mip_chain[0].shape[1]
    total_h = sum(img.shape[0] for img in mip_chain)
    C = 4
    atlas = np.zeros((total_h, common_w, C), dtype=np.float32)

    y_offset = 0
    for level, img in enumerate(mip_chain):
        H = img.shape[0]
        repeats = 1 << level  # 2^level copies
        flipped = img[::-1]  # flip V so atlas isn't upside down in Blender
        tiled = np.tile(flipped, (1, repeats, 1))
        atlas[y_offset:y_offset + H] = tiled[:, :common_w]
        y_offset += H

    return atlas, (common_w, total_h)


def estimate_atlas_resolution(probe_w, levels=6):
    """Compute the natural atlas dimensions for a vertical stack of mips.

    Returns (width, height) tuple.
    """
    total_h = 0
    w = probe_w
    h = probe_w // 2
    for _ in range(levels):
        total_h += h
        w = max(w // 2, 1)
        h = max(h // 2, 1)
    return probe_w, total_h
