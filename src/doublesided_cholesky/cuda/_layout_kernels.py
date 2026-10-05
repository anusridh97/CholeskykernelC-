"""Layout copies with int64 tile addresses (F1)."""
import triton
import triton.language as tl

@triton.jit
def _dense_to_tile(dense, out, r, n, matrix_size, B: tl.constexpr, STRIPE: tl.constexpr):
    """Copy dense (r,r) to 64x64 lower tiles; integer addressing is int64."""
    ti, tj, sg = (tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64))
    batch = 0
    lr = sg * STRIPE + tl.arange(0, STRIPE).to(tl.int64)[:, None]
    lc = tl.arange(0, B).to(tl.int64)[None, :]
    rows = ti * B + lr
    cols = tj * B + lc
    mask = (tj <= ti) & (rows < r) & (cols < r)
    tid = ti * (ti + 1) // 2 + tj
    val = tl.load(dense + batch * matrix_size + rows * r + cols, mask=mask, other=0.0)
    tl.store(out + batch * (n * (n + 1) // 2 * B * B) + tid * B * B + lr * B + lc, val, mask=mask)

@triton.jit
def _tile_to_dense(inp, out, r, n, matrix_size, B: tl.constexpr, STRIPE: tl.constexpr):
    """Copy lower tiles to dense symmetric (r,r); integer addressing is int64."""
    ti, tj, sg = (tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64))
    batch = 0
    lr = sg * STRIPE + tl.arange(0, STRIPE).to(tl.int64)[:, None]
    lc = tl.arange(0, B).to(tl.int64)[None, :]
    rows = ti * B + lr
    cols = tj * B + lc
    mask = (rows < r) & (cols < r)
    lo = tl.maximum(ti, tj)
    hi = tl.minimum(ti, tj)
    tid = lo * (lo + 1) // 2 + hi
    off = tid * B * B + tl.where(ti >= tj, lr * B + lc, lc * B + lr)
    total = n * (n + 1) // 2 * B * B
    val = tl.load(inp + batch * total + off, mask=mask, other=0.0)
    tl.store(out + batch * matrix_size + rows * r + cols, val, mask=mask)

@triton.jit
def _pack_lower(dense, packed, r, matrix_size, pn, BLOCK: tl.constexpr):
    """Copy dense (batch,r,r) to row-packed lower (batch,r*(r+1)/2)."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    mask = (row < r) & (cols <= row)
    value = tl.load(dense + batch * matrix_size + row * r + cols, mask=mask, other=0.0)
    tl.store(packed + batch * pn + row * (row + 1) // 2 + cols, value, mask=mask)

@triton.jit
def _unpack_lower(packed, dense, r, matrix_size, pn, BLOCK: tl.constexpr):
    """Copy row-packed lower to symmetric dense (batch,r,r)."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    mask = (row < r) & (cols < r)
    lo, hi = (tl.maximum(row, cols), tl.minimum(row, cols))
    value = tl.load(packed + batch * pn + lo * (lo + 1) // 2 + hi, mask=mask, other=0.0)
    tl.store(dense + batch * matrix_size + row * r + cols, value, mask=mask)
