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


def nn_gemm_bt(X, Wt, bias, shift):
    """Y = sat8(rshift(X . Wt^T + bias, shift)) -- docs/design/gemm_bt.md.
    Wt: (N, K) int8, the transpose of gemm's W."""
    return nn_gemm(X, np.asarray(Wt).T, bias, shift)


def nn_lut8(X, table):
    """Y = table[X + 128] -- docs/design/lut8.md.
    X: (rows, cols) int8. table: 256 int8 values, table[i] = f(i - 128).
    Returns (rows, cols) int8 as int64."""
    table = np.asarray(table, dtype=np.int64)
    assert table.shape == (256,)
    return table[np.asarray(X, dtype=np.int64) + 128]


def _requant(v, shift):
    assert v.min() >= INT32_MIN and v.max() <= INT32_MAX, "int32 overflow"
    return sat8(rshift(v, shift))


def nn_add(A, B, sa, sb, shift):
    """Y = sat8(rshift(A * 2^sa + B * 2^sb, shift)) -- docs/design/eltwise.md.
    A, B: (rows, cols) int8. Returns (rows, cols) int8 as int64."""
    v = np.asarray(A, dtype=np.int64) * (1 << sa) + np.asarray(B, dtype=np.int64) * (1 << sb)
    return _requant(v, shift)


def nn_mul(A, B, shift):
    """Y = sat8(rshift(A * B, shift)) -- docs/design/eltwise.md."""
    v = np.asarray(A, dtype=np.int64) * np.asarray(B, dtype=np.int64)
    return _requant(v, shift)


def nn_softmax(X, table):
    """Row-wise softmax -- docs/design/softmax.md.
    X: (rows, cols) int8. table: 256 uint16 values, table[0] >= 1.
    Returns (rows, cols) int8 as int64: min(127, (e*128 + S//2) // S)."""
    X = np.asarray(X, dtype=np.int64)
    table = np.asarray(table, dtype=np.int64)
    assert table.shape == (256,) and table[0] >= 1 and table.min() >= 0 and table.max() <= 0xFFFF
    m = X.max(axis=1, keepdims=True)
    e = table[m - X]
    S = e.sum(axis=1, keepdims=True)
    assert S.max() < (1 << 32), "row sum outside uint32"
    return np.minimum((e * 128 + S // 2) // S, 127)   # all non-negative: // is C's unsigned divide


def nn_layernorm(X, gamma, beta, shift):
    """Row-wise layer normalization -- docs/design/layernorm.md.
    X: (rows, cols) int8. gamma: (cols,) int8. beta: (cols,) int32.
    Returns (rows, cols) int8 as int64."""
    import math
    X = np.asarray(X, dtype=np.int64)
    gamma = np.asarray(gamma, dtype=np.int64)
    beta = np.asarray(beta, dtype=np.int64)
    n = X.shape[1]
    s1 = X.sum(axis=1, keepdims=True)
    s2 = (X * X).sum(axis=1, keepdims=True)
    V = n * s2 - s1 * s1
    assert V.min() >= 0 and V.max() < (1 << 44)
    q = np.array([[max(math.isqrt(int(v)), 1)] for v in V[:, 0]], dtype=np.int64)
    D = n * X - s1
    N = D * gamma[None, :]
    assert np.abs(N).max() < (1 << 30)
    t = np.sign(N) * ((2 * np.abs(N) + q) // (2 * q))   # round half away from zero
    return _requant(t + beta[None, :], shift)


def exp_table(s_in):
    """Softmax table for inputs of scale s_in: table[d] = round(2^15 * exp(-d * s_in)).
    Floating point is used only to build this constant."""
    d = np.arange(256)
    return np.round(32768.0 * np.exp(-d * s_in)).astype(np.int64)
