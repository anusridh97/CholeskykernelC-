"""GPU application experiments for the double-sided Cholesky kernel.

Two experiments live here, both built on the batched Triton kernel in
``doublesided_cholesky.py``:

  * ``curvature_optimizer`` — a GPU-resident, per-block-batched incremental
    curvature (natural-gradient) optimizer. One low-rank whitened-Fisher chart
    per parameter block; every training step refreshes *all* charts in a single
    batched kernel launch (O(r^2) each) instead of the amortized O(r^3)
    eigendecomposition that Shampoo/SOAP/K-FAC periodically pay.

  * ``parallel_gp_thompson`` — hundreds to thousands of independent GP
    surrogates conditioned in parallel, one rank-1 update per arm per round,
    all fused into one launch. The purest expression of the "T is where we win"
    batching story on real hardware.

Both modules import the batched kernel lazily and fall back to the pure-PyTorch
reference (``incremental_update_reference``) when CUDA/Triton is unavailable, so
the training/BO loops and their correctness checks run on CPU too.
"""
