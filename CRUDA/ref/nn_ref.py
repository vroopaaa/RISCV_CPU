"""Python reference for the lego library (see docs/design/).

One function per lego, with exactly the integer semantics of the C code, so
results can be compared bit for bit. All functions work on plain row-major
numpy arrays; to_tiles / from_tiles convert to and from the library's padded
16 x 16 tile layout (docs/design/00-overview.md section 2).
"""

import numpy as np

TILE = 16
TILE_BYTES = TILE * TILE
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


def pad16(x):
    return (x + TILE - 1) // TILE * TILE


def rshift(v, s):
    """Rounding right shift, halves toward +inf (overview section 1).
    Works on Python ints and on int64 numpy arrays."""
    if s == 0:
        return v
    return (v + (1 << (s - 1))) >> s


def sat8(v):
    return np.clip(v, -128, 127)


def to_tiles(mat, ldt=None):
    """(rows, cols) int8 matrix -> bytes in the padded tile layout.
    ldt (tiles per tile-row) defaults to pad16(cols) / 16; a larger ldt
    leaves extra zero tiles at the end of every tile-row."""
    mat = np.asarray(mat, dtype=np.int64)
    rows, cols = mat.shape
    tr, tc = pad16(rows) // TILE, pad16(cols) // TILE
    ldt = tc if ldt is None else ldt
    assert ldt >= tc
    buf = np.zeros((tr, ldt, TILE, TILE), dtype=np.int8)
    padded = np.zeros((tr * TILE, tc * TILE), dtype=np.int64)
    padded[:rows, :cols] = mat
    for r in range(tr):
        for c in range(tc):
            buf[r, c] = padded[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
    return buf.tobytes()


def from_tiles(buf, rows, cols, ldt=None):
    """Inverse of to_tiles: (padded matrix, real (rows, cols) part)."""
    tr, tc = pad16(rows) // TILE, pad16(cols) // TILE
    ldt = tc if ldt is None else ldt
    t = np.frombuffer(bytes(buf[:tr * ldt * TILE_BYTES]), dtype=np.int8).reshape(tr, ldt, TILE, TILE)
    padded = np.zeros((tr * TILE, tc * TILE), dtype=np.int64)
    for r in range(tr):
        for c in range(tc):
            padded[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = t[r, c]
    return padded, padded[:rows, :cols]


def nn_gemm(X, W, bias, shift):
    """Y = sat8(rshift(X . W + bias, shift)) -- docs/design/gemm.md.
    X: (M, K) int8. W: (K, N) int8. bias: (N,) int32 or None.
    Returns (M, N) int8 as int64."""
    acc = np.asarray(X, dtype=np.int64) @ np.asarray(W, dtype=np.int64)
    if bias is not None:
        acc = acc + np.asarray(bias, dtype=np.int64)[None, :]
    assert acc.min() >= INT32_MIN and acc.max() <= INT32_MAX, \
        "accumulator outside int32 -- test breaks the overview's range rule"
    return sat8(rshift(acc, shift))
