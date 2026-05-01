"""Pure Python SPZ decoder based on Spark v2.0.0 format spec.

Supports versions 1 (float16 centers), 2 (24-bit fixed-point, int8 quat),
and 3 (24-bit fixed-point, smallest-three quat).
"""

import struct
import gzip
import numpy as np

SH_C0 = 0.28209479177387814


def load_spz(spz_path):
    """Decode an SPZ file and return splat data matching ``read_ply_attributes``.

    Returns:
        Tuple of ``(positions, colors, opacities, scales, rotations,
                     raw_dc, sh_coeffs, sh_degree)``.
    """
    with open(spz_path, 'rb') as f:
        raw = f.read()

    if raw[:2] == b'\x1f\x8b':
        data = gzip.decompress(raw)
    else:
        assert raw[:4] == b'NGSP', f"Bad magic: {raw[:4]}"
        data = raw

    magic = struct.unpack_from('<I', data, 0)[0]
    assert magic == 0x5053474e, f"Bad magic: {magic:#x}"

    version = struct.unpack_from('<I', data, 4)[0]
    assert 1 <= version <= 3, f"Unsupported SPZ version {version}"

    num_points = struct.unpack_from('<I', data, 8)[0]
    sh_degree = data[12]
    fractional_bits = data[13]
    off = 16

    # --- Positions ---
    if version == 1:
        pos_u16 = np.frombuffer(data, dtype=np.uint16, count=num_points * 3, offset=off)
        off += num_points * 3 * 2
        positions = pos_u16.view(np.float16).astype(np.float32).reshape(-1, 3)
    else:
        fixed = 1 << fractional_bits
        pos_bytes = num_points * 9
        pos_u8 = np.frombuffer(data, dtype=np.uint8, count=pos_bytes, offset=off)
        off += pos_bytes
        pos_u8 = pos_u8.reshape(-1, 9)
        positions = np.empty((num_points, 3), dtype=np.float32)
        for c in range(3):
            b = pos_u8[:, c*3:c*3+3].astype(np.int32)
            val = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            val = np.where(val >= 0x800000, val - 0x1000000, val)
            positions[:, c] = val.astype(np.float32) / fixed

    # --- Alphas ---
    alphas = np.frombuffer(data, dtype=np.uint8, count=num_points, offset=off)
    off += num_points
    opacities = (alphas / 255.0).astype(np.float32).reshape(-1, 1)

    # --- Colors (SH DC, pre-sigmoid) ---
    rgb_raw = np.frombuffer(data, dtype=np.uint8, count=num_points * 3, offset=off)
    off += num_points * 3
    rgb_f = rgb_raw.astype(np.float32).reshape(-1, 3) / 255.0
    # Spark decode: r = (raw/255 - 0.5) * (SH_C0/0.15) + 0.5
    raw_dc = (rgb_f - 0.5) * (SH_C0 / 0.15) + 0.5  # (N, 3) pre-sigmoid SH DC

    # Display color for batch rendering (sigmoid + gamma 2.2)
    linear = 1.0 / (1.0 + np.exp(-raw_dc))
    colors = np.power(linear, 2.2).astype(np.float32)

    # --- Scales (actual scales, always positive) ---
    scale_raw = np.frombuffer(data, dtype=np.uint8, count=num_points * 3, offset=off)
    off += num_points * 3
    log_scales = scale_raw.astype(np.float32) / 16.0 - 10.0
    scales = np.exp(log_scales).reshape(-1, 3)

    # --- Rotations ---
    if version == 3:
        quat_u32 = np.frombuffer(data, dtype=np.uint32, count=num_points, offset=off)
        off += num_points * 4
        rotations = _decode_quat_v3(quat_u32)
    else:
        quat_u8 = np.frombuffer(data, dtype=np.uint8, count=num_points * 3, offset=off)
        off += num_points * 3
        quat_f = quat_u8.astype(np.float32).reshape(-1, 3) / 127.5 - 1.0  # [-1, 1]
        x, y, z = quat_f[:, 0], quat_f[:, 1], quat_f[:, 2]
        w_sq = 1.0 - (x*x + y*y + z*z)
        w = np.sqrt(np.maximum(w_sq, 0.0))
        rotations = np.column_stack([w, x, y, z])
        # Renormalize to unit length (handles quantization drift)
        q_norm = np.sqrt(np.sum(rotations * rotations, axis=1, keepdims=True))
        rotations /= np.maximum(q_norm, 1e-10)

    # --- SH coefficients ---
    if sh_degree >= 1:
        sh_vecs = {1: 3, 2: 8, 3: 15}[sh_degree]
        sh_bytes = num_points * sh_vecs * 3
        sh_raw = np.frombuffer(data, dtype=np.uint8, count=sh_bytes, offset=off)
        off += sh_bytes
        sh_f = sh_raw.astype(np.float32) / 128.0 - 1.0  # (raw - 128) / 128
        sh_coeffs = sh_f.reshape(-1, sh_vecs * 3)
    else:
        sh_coeffs = None

    return positions, colors, opacities, scales, rotations, raw_dc, sh_coeffs, sh_degree


def _decode_quat_v3(quat_u32):
    """Vectorized decode of SPZ v3 smallest-three quaternion encoding.

    32 bits per quaternion: 2-bit largest-component index + 3 × 10-bit
    components (1 sign + 9 magnitude).  Magnitude range [0, 1/√2].
    """
    N = len(quat_u32)

    mags = np.empty((N, 3), dtype=np.int32)
    mags[:, 0] = (quat_u32 >> 0) & 0x1FF
    mags[:, 1] = (quat_u32 >> 10) & 0x1FF
    mags[:, 2] = (quat_u32 >> 20) & 0x1FF

    signs = np.empty((N, 3), dtype=np.int32)
    signs[:, 0] = (quat_u32 >> 9) & 0x1
    signs[:, 1] = (quat_u32 >> 19) & 0x1
    signs[:, 2] = (quat_u32 >> 29) & 0x1

    largest_idx = (quat_u32 >> 30).astype(np.uint8)

    max_val = 1.0 / np.sqrt(2.0)
    decoded = max_val * mags.astype(np.float32) / 511.0
    decoded[signs.astype(bool)] *= -1.0

    # Map decoded[3] → quaternion components based on largest_idx.
    # Encoding order (LSB-first): component with highest index not equal
    # to largest_idx is stored first, then next highest, etc.
    rotations = np.zeros((N, 4), dtype=np.float32)

    # largest_idx=0: decoded order → q[3], q[2], q[1]
    mask = largest_idx == 0
    rotations[mask, 3] = decoded[mask, 0]
    rotations[mask, 2] = decoded[mask, 1]
    rotations[mask, 1] = decoded[mask, 2]

    # largest_idx=1: decoded order → q[3], q[2], q[0]
    mask = largest_idx == 1
    rotations[mask, 3] = decoded[mask, 0]
    rotations[mask, 2] = decoded[mask, 1]
    rotations[mask, 0] = decoded[mask, 2]

    # largest_idx=2: decoded order → q[3], q[1], q[0]
    mask = largest_idx == 2
    rotations[mask, 3] = decoded[mask, 0]
    rotations[mask, 1] = decoded[mask, 1]
    rotations[mask, 0] = decoded[mask, 2]

    # largest_idx=3: decoded order → q[2], q[1], q[0]
    mask = largest_idx == 3
    rotations[mask, 2] = decoded[mask, 0]
    rotations[mask, 1] = decoded[mask, 1]
    rotations[mask, 0] = decoded[mask, 2]

    # Derive the largest component from the other three
    sum_sq = np.sum(rotations * rotations, axis=1)
    for idx in range(4):
        mask = largest_idx == idx
        rotations[mask, idx] = np.sqrt(np.maximum(0.0, 1.0 - sum_sq[mask]))

    return rotations
