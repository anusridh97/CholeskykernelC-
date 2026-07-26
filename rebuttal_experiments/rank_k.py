from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import socket
import statistics
import subprocess
import time
from pathlib import Path

import torch

from .common import (
    ROOT,
    add_student_pkg,
    atomic_json_dump,
    configure_threads,
    result_path,
    set_reproducible_seed,
    tensor_relative_error,
)

PILOT_SEED = 1000
PILOT_RANK = 1024
PILOT_KS = (1, 2, 4, 8, 16, 32)
PILOT_THREADS = 1
PILOT_REPEATS = 20
PILOT_VARIANT = "compact-v4-hybrid16-bs32"
CONFIRMATION_CAMPAIGN = "e6-compact-direct-confirm-v2"
CONFIRMATION_RANKS = (256, 1024)
CONFIRMATION_KS = (1, 2, 4, 8, 16, 32)
CONFIRMATION_SEEDS = tuple(range(88001, 88011))
CONFIRMATION_THREADS = 1
CONFIRMATION_REPEATS = 30
CONFIRMATION_PROTOCOL = (
    Path(__file__).resolve().with_name(
        "e6_compact_direct_confirmation_protocol.json"
    )
)
COMPACT_BLOCK_SIZE = 32
COMPACT_CPP_MAX_K = 32
NATIVE_PHASE_NAMES = (
    "vector_copy_and_pad_reset",
    "cholesky_rank1_update",
    "recover_vz",
    "forward_solve",
    "operator_left_sweep",
    "operator_right_sweep_and_rank1",
)
COMPACT_PHASE_NAMES = (
    "solve_Z",
    "solve_W",
    "operator_rank_k_correction",
    "compact_parameter_recurrence",
    "compact_factor_product",
    "operator_left_compact_apply",
    "operator_direct_right_compact_apply",
)


def whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(L, left.mT, upper=False).mT.contiguous()


def apply_compact_inverse_left(
    X: torch.Tensor,
    output: torch.Tensor,
    U: torch.Tensor,
    Q: torch.Tensor,
    d: torch.Tensor,
    *,
    block_size: int,
    prefix: torch.Tensor,
    coefficient: torch.Tensor,
) -> torch.Tensor:
    """Apply P=C^{-1} from the left using its rank-k semiseparable form.

    If H=I+UU^T=CC^T, the compact-factor recurrence produces d and Q such
    that P_ii=1/d_i and P_ij=-q_i^T u_j for j<i.  A row block therefore
    needs only the k-by-r prefix U_prior^T X_prior plus one small strictly
    lower block.  All storage is supplied by the caller.
    """
    if X.shape != output.shape or X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("X and output must be distinct square matrices")
    if X.data_ptr() == output.data_ptr():
        raise ValueError("compact inverse application does not support aliasing")
    r = X.shape[0]
    k = U.shape[1]
    if U.shape != Q.shape or U.shape[0] != r or d.shape != (r,):
        raise ValueError("incompatible U, Q, or d shape")
    if prefix.shape != (k, r):
        raise ValueError("prefix must have shape (k, r)")
    if coefficient.shape != (block_size, block_size):
        raise ValueError("coefficient must match the declared block size")

    prefix.zero_()
    for start in range(0, r, block_size):
        stop = min(r, start + block_size)
        width = stop - start
        U_block = U[start:stop]
        Q_block = Q[start:stop]
        X_block = X[start:stop]
        output_block = output[start:stop]
        coefficient_block = coefficient[:width, :width]

        torch.mm(Q_block, U_block.mT, out=coefficient_block)
        coefficient_block.neg_().tril_(-1)
        output_block.copy_(X_block).div_(d[start:stop, None])
        output_block.addmm_(Q_block, prefix, beta=1.0, alpha=-1.0)
        output_block.addmm_(
            coefficient_block,
            X_block,
            beta=1.0,
            alpha=1.0,
        )
        prefix.addmm_(U_block.mT, X_block, beta=1.0, alpha=1.0)
    return output


def median_time(function, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def median_prepared_time(prepare, function, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        arguments = prepare()
        start = time.perf_counter()
        function(*arguments)
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_pilot(args: argparse.Namespace) -> None:
    expected = {
        "rank": PILOT_RANK,
        "ks": list(PILOT_KS),
        "seed": PILOT_SEED,
        "repeats": PILOT_REPEATS,
        "threads": PILOT_THREADS,
        "implementation": "compact",
    }
    actual = {
        "rank": args.rank,
        "ks": list(args.ks),
        "seed": args.seed,
        "repeats": args.repeats,
        "threads": args.threads,
        "implementation": args.implementation,
    }
    if actual != expected:
        raise ValueError(
            "compact pilot configuration is frozen; "
            f"expected {expected}, got {actual}"
        )


def _validate_confirmation(args: argparse.Namespace) -> dict:
    if not CONFIRMATION_PROTOCOL.is_file():
        raise FileNotFoundError(CONFIRMATION_PROTOCOL)
    with CONFIRMATION_PROTOCOL.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    expected_protocol = {
        "campaign_tag": CONFIRMATION_CAMPAIGN,
        "status": "frozen_before_confirmation",
        "ranks": list(CONFIRMATION_RANKS),
        "ks": list(CONFIRMATION_KS),
        "seeds": list(CONFIRMATION_SEEDS),
        "threads": CONFIRMATION_THREADS,
        "timed_repeats_per_method": CONFIRMATION_REPEATS,
        "implementation": "compact",
    }
    actual_protocol = {
        key: protocol.get(key) for key in expected_protocol
    }
    if actual_protocol != expected_protocol:
        raise ValueError(
            "E6 confirmation protocol differs from the locked constants: "
            f"expected {expected_protocol}, got {actual_protocol}"
        )
    expected_run = {
        "rank": args.rank,
        "ks": list(CONFIRMATION_KS),
        "seed": args.seed,
        "repeats": CONFIRMATION_REPEATS,
        "threads": CONFIRMATION_THREADS,
        "implementation": "compact",
    }
    actual_run = {
        "rank": args.rank,
        "ks": list(args.ks),
        "seed": args.seed,
        "repeats": args.repeats,
        "threads": args.threads,
        "implementation": args.implementation,
    }
    if args.rank not in CONFIRMATION_RANKS:
        raise ValueError(f"rank {args.rank} is outside the E6 confirmation")
    if args.seed not in CONFIRMATION_SEEDS:
        raise ValueError(f"seed {args.seed} is outside the E6 confirmation")
    if actual_run != expected_run:
        raise ValueError(
            "E6 confirmation arguments differ from the frozen protocol: "
            f"expected {expected_run}, got {actual_run}"
        )
    required_environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
    }
    invalid_environment = {
        name: os.environ.get(name)
        for name, expected in required_environment.items()
        if os.environ.get(name) != expected
    }
    if invalid_environment:
        raise RuntimeError(
            "E6 confirmation requires a hermetic single-thread environment; "
            f"invalid values: {invalid_environment}"
        )
    if os.environ.get("SLURM_JOB_ACCOUNT") != "marlowe-m000151-pm06":
        raise RuntimeError("E6 confirmation requires the locked Slurm account")
    if os.environ.get("SLURM_JOB_PARTITION") != "batch":
        raise RuntimeError("E6 confirmation requires the batch partition")
    declared_commit = os.environ.get("SOURCE_COMMIT")
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if not declared_commit or declared_commit != actual_commit:
        raise RuntimeError(
            f"E6 source-lock mismatch: declared={declared_commit!r}, "
            f"actual={actual_commit!r}"
        )
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0:
        raise RuntimeError("E6 tracked worktree differs from the locked commit")
    if (
        subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode
        != 0
    ):
        raise RuntimeError("E6 index differs from the locked commit")
    return protocol


def _validate_confirmation_runtime() -> None:
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError(
            "E6 confirmation did not obtain the required one-thread runtime"
        )
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("E6 confirmation requires observable CPU affinity")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError(
            "E6 confirmation requires one isolated physical CPU; "
            f"observed affinity={affinity}"
        )


def _pilot_timing_samples(
    *,
    legacy,
    candidate,
    candidate_name: str,
    refactor,
    prepare,
    repeats: int,
) -> tuple[dict[str, list[float]], list[list[str]]]:
    """Counterbalance all six method orders without timing state clones."""
    functions = {
        "legacy": (legacy, True),
        candidate_name: (candidate, True),
        "refactor": (refactor, False),
    }
    samples = {name: [] for name in functions}
    orders = list(itertools.permutations(functions))
    schedule = []
    for repeat in range(repeats):
        order = orders[repeat % len(orders)]
        schedule.append(list(order))
        for name in order:
            function, prepared = functions[name]
            arguments = prepare() if prepared else ()
            started = time.perf_counter()
            function(*arguments)
            samples[name].append(time.perf_counter() - started)
    return samples, schedule


def run(args: argparse.Namespace) -> dict:
    args.implementation = getattr(args, "implementation", "legacy")
    args.pilot = getattr(args, "pilot", False)
    args.confirmation = getattr(args, "confirmation", False)
    if args.pilot and args.confirmation:
        raise ValueError("pilot and confirmation modes are mutually exclusive")
    confirmation_protocol = None
    if args.pilot:
        _validate_pilot(args)
    if args.confirmation:
        confirmation_protocol = _validate_confirmation(args)
    configure_threads(args.threads)
    if args.confirmation:
        _validate_confirmation_runtime()
    set_reproducible_seed(args.seed)
    add_student_pkg()
    import fast_cholesky

    r = args.rank
    raw = torch.randn(r, r, dtype=torch.float64)
    G = raw @ raw.mT + r * torch.eye(r, dtype=torch.float64)
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(r, r, dtype=torch.float64)
    Aw = whiten(L, M)
    rows = []

    for k in args.ks:
        Z = torch.randn(r, k, dtype=torch.float64) * 0.05
        W = torch.randn(r, k, dtype=torch.float64) * 0.05

        def sequential(L_work, Aw_work):
            for column in range(k):
                Aw_work, _ = fast_cholesky.fast_incremental_update(
                    L_work,
                    Aw_work,
                    Z[:, column].clone(),
                    W[:, column],
                )
            return L_work, Aw_work

        use_native = args.implementation in ("native", "both", "all")
        use_compact = args.implementation in ("compact", "all")

        workspace = torch.empty(7 * r, dtype=L.dtype)
        vz_out = torch.empty(k, r, dtype=L.dtype)
        phase_seconds = torch.empty(len(NATIVE_PHASE_NAMES), dtype=torch.float64)

        def native(L_work, Aw_work):
            Aw_result, vz_result = (
                fast_cholesky.fast_incremental_update_multi_out(
                    L_work,
                    Aw_work,
                    Z,
                    W,
                    workspace,
                    vz_out,
                )
            )
            return L_work, Aw_result, vz_result

        def native_profile(L_work, Aw_work):
            fast_cholesky.fast_incremental_update_multi_profile_out(
                L_work,
                Aw_work,
                Z,
                W,
                workspace,
                vz_out,
                phase_seconds,
            )
            return phase_seconds.clone()

        compact_U = torch.empty_like(Z)
        compact_Y = torch.empty_like(W)
        compact_d = torch.empty(r, dtype=L.dtype)
        compact_Q = torch.empty_like(Z)
        compact_factor_workspace = torch.empty(
            k * k + 2 * k,
            dtype=L.dtype,
        )
        compact_buffer = torch.empty_like(Aw)
        compact_scratch = torch.empty_like(Aw)
        compact_prefix = torch.empty(k, r, dtype=L.dtype)
        compact_apply_workspace = torch.empty(
            max(k * r, r + k), dtype=L.dtype
        )
        compact_coefficient = torch.empty(
            COMPACT_BLOCK_SIZE,
            COMPACT_BLOCK_SIZE,
            dtype=L.dtype,
        )
        compact_cpp_phase = torch.empty(2, dtype=torch.float64)
        compact_apply_backend = (
            "fused_cpp_left_direct_cpp_right"
            if k <= COMPACT_CPP_MAX_K
            else "blocked_blas_semiseparable"
        )

        def compact_left(X, output):
            if k <= COMPACT_CPP_MAX_K:
                return fast_cholesky.fast_compact_apply_left_out(
                    X,
                    compact_U,
                    compact_Q,
                    compact_d,
                    output,
                    compact_apply_workspace,
                )
            return apply_compact_inverse_left(
                X,
                output,
                compact_U,
                compact_Q,
                compact_d,
                block_size=COMPACT_BLOCK_SIZE,
                prefix=compact_prefix,
                coefficient=compact_coefficient,
            )

        def compact(L_work, Aw_work):
            torch.linalg.solve_triangular(
                L_work,
                Z,
                upper=False,
                out=compact_U,
            )
            torch.linalg.solve_triangular(
                L_work,
                W,
                upper=False,
                out=compact_Y,
            )
            torch.mm(compact_U, compact_Y.mT, out=compact_buffer)
            compact_buffer.add_(Aw_work)
            fast_cholesky.fast_compact_factor_out(
                L_work,
                compact_U,
                compact_d,
                compact_Q,
                compact_factor_workspace,
            )
            compact_left(
                compact_buffer,
                compact_scratch,
            )
            fast_cholesky.fast_compact_apply_right_out(
                compact_scratch,
                compact_U,
                compact_Q,
                compact_d,
                Aw_work,
                compact_apply_workspace,
            )
            return L_work, Aw_work

        def compact_profile(L_work, Aw_work):
            samples = []
            started = time.perf_counter()
            torch.linalg.solve_triangular(
                L_work,
                Z,
                upper=False,
                out=compact_U,
            )
            samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            torch.linalg.solve_triangular(
                L_work,
                W,
                upper=False,
                out=compact_Y,
            )
            samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            torch.mm(compact_U, compact_Y.mT, out=compact_buffer)
            compact_buffer.add_(Aw_work)
            samples.append(time.perf_counter() - started)
            fast_cholesky.fast_compact_factor_profile_out(
                L_work,
                compact_U,
                compact_d,
                compact_Q,
                compact_factor_workspace,
                compact_cpp_phase,
            )
            samples.extend(float(value) for value in compact_cpp_phase.tolist())
            started = time.perf_counter()
            compact_left(
                compact_buffer,
                compact_scratch,
            )
            samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            fast_cholesky.fast_compact_apply_right_out(
                compact_scratch,
                compact_U,
                compact_Q,
                compact_d,
                Aw_work,
                compact_apply_workspace,
            )
            samples.append(time.perf_counter() - started)
            return samples

        def refactor():
            G_new = G + Z @ Z.mT
            M_new = M + Z @ W.mT
            L_new = torch.linalg.cholesky(G_new).contiguous()
            return L_new, whiten(L_new, M_new)

        for _ in range(2):
            sequential(L.clone(), Aw.clone())
            if use_native:
                native(L.clone(), Aw.clone())
            if use_compact:
                compact(L.clone(), Aw.clone())
            refactor()
        L_seq, Aw_seq = sequential(L.clone(), Aw.clone())
        if use_native:
            L_native = L.clone()
            Aw_native = Aw.clone()
            _, Aw_native, _ = native(L_native, Aw_native)
            native_l_legacy_error = tensor_relative_error(L_native, L_seq)
            native_aw_legacy_error = tensor_relative_error(Aw_native, Aw_seq)
        else:
            L_native = None
            Aw_native = None
            native_l_legacy_error = None
            native_aw_legacy_error = None
        if use_compact:
            L_compact = L.clone()
            Aw_compact = Aw.clone()
            compact(L_compact, Aw_compact)
            compact_l_legacy_error = tensor_relative_error(L_compact, L_seq)
            compact_aw_legacy_error = tensor_relative_error(
                Aw_compact,
                Aw_seq,
            )
        else:
            L_compact = None
            Aw_compact = None
            compact_l_legacy_error = None
            compact_aw_legacy_error = None
        L_exact, Aw_exact = refactor()
        l_error = tensor_relative_error(L_seq, L_exact)
        aw_error = tensor_relative_error(Aw_seq, Aw_exact)
        if l_error > 1e-9 or aw_error > 1e-9:
            raise RuntimeError(
                f"rank-k mismatch r={r} k={k}: L={l_error:.3e}, Aw={aw_error:.3e}"
            )
        if L_native is not None and Aw_native is not None:
            native_l_error = tensor_relative_error(L_native, L_exact)
            native_aw_error = tensor_relative_error(Aw_native, Aw_exact)
            if native_l_error > 1e-9 or native_aw_error > 1e-9:
                raise RuntimeError(
                    f"native rank-k mismatch r={r} k={k}: "
                    f"L={native_l_error:.3e}, Aw={native_aw_error:.3e}"
                )
        else:
            native_l_error = None
            native_aw_error = None
        if L_compact is not None and Aw_compact is not None:
            compact_l_error = tensor_relative_error(L_compact, L_exact)
            compact_aw_error = tensor_relative_error(Aw_compact, Aw_exact)
            if compact_l_error > 1e-9 or compact_aw_error > 1e-9:
                raise RuntimeError(
                    f"compact rank-k mismatch r={r} k={k}: "
                    f"L={compact_l_error:.3e}, Aw={compact_aw_error:.3e}"
                )
        else:
            compact_l_error = None
            compact_aw_error = None

        repeats = args.repeats if k <= 8 else max(3, args.repeats // 2)
        if args.pilot or args.confirmation:
            # The pilot and confirmation retain the same number of timing
            # samples for every k so uncertainty is comparable across the
            # complete ladder.
            repeats = args.repeats
            timing_samples, timing_order_schedule = _pilot_timing_samples(
                legacy=sequential,
                candidate=compact,
                candidate_name="compact",
                refactor=refactor,
                prepare=lambda: (L.clone(), Aw.clone()),
                repeats=repeats,
            )
            sequential_seconds = statistics.median(timing_samples["legacy"])
            compact_seconds = statistics.median(timing_samples["compact"])
            native_seconds = None
            refactor_seconds = statistics.median(timing_samples["refactor"])
            compact_phase_samples = []
            for _ in range(repeats):
                compact_phase_samples.append(
                    compact_profile(
                        L.clone(),
                        Aw.clone(),
                    )
                )
            compact_phase_medians = {
                name: statistics.median(
                    sample[index] for sample in compact_phase_samples
                )
                for index, name in enumerate(COMPACT_PHASE_NAMES)
            }
            compact_phase_total_samples = [
                sum(sample) for sample in compact_phase_samples
            ]
            compact_phase_total_median = statistics.median(
                compact_phase_total_samples
            )
            compact_phase_median_sum = sum(compact_phase_medians.values())
            compact_phase_fractions = {
                name: value / compact_phase_median_sum
                for name, value in compact_phase_medians.items()
            }
            native_phase_samples = None
            native_phase_medians = None
            native_phase_total_samples = None
            native_phase_total_median = None
            native_phase_fractions = None
        else:
            timing_samples = None
            timing_order_schedule = None
            native_phase_samples = None
            native_phase_medians = None
            native_phase_total_samples = None
            native_phase_total_median = None
            native_phase_fractions = None
            compact_phase_samples = None
            compact_phase_medians = None
            compact_phase_total_samples = None
            compact_phase_total_median = None
            compact_phase_fractions = None
            sequential_seconds = median_prepared_time(
                lambda: (L.clone(), Aw.clone()), sequential, repeats
            )
            native_seconds = (
                median_prepared_time(
                    lambda: (L.clone(), Aw.clone()),
                    native,
                    repeats,
                )
                if use_native
                else None
            )
            compact_seconds = (
                median_prepared_time(
                    lambda: (L.clone(), Aw.clone()),
                    compact,
                    repeats,
                )
                if use_compact
                else None
            )
            refactor_seconds = median_time(refactor, repeats)

        row = {
            "rank": r,
            "k": k,
            "repeats": repeats,
            "sequential_seconds": sequential_seconds,
            "refactor_seconds": refactor_seconds,
            "speedup": refactor_seconds / sequential_seconds,
            "L_relative_error": l_error,
            "Aw_relative_error": aw_error,
        }
        if native_seconds is not None:
            row.update(
                {
                    "native_seconds": native_seconds,
                    "legacy_over_native_speedup": (
                        sequential_seconds / native_seconds
                    ),
                    "refactor_over_native_speedup": (
                        refactor_seconds / native_seconds
                    ),
                    "native_L_relative_error": native_l_error,
                    "native_Aw_relative_error": native_aw_error,
                    "native_vs_legacy_L_relative_error": (
                        native_l_legacy_error
                    ),
                    "native_vs_legacy_Aw_relative_error": (
                        native_aw_legacy_error
                    ),
                    "native_workspace_elements": 7 * r,
                    "native_workspace_bytes": (
                        workspace.numel() * workspace.element_size()
                    ),
                    "native_vz_output_bytes": (
                        vz_out.numel() * vz_out.element_size()
                    ),
                    "native_timed_dense_allocations": 0,
                    "legacy_dense_allocations_per_call": k,
                    "native_phase_names": list(NATIVE_PHASE_NAMES),
                    "native_phase_seconds_samples": native_phase_samples,
                    "native_phase_seconds_medians": native_phase_medians,
                    "native_profile_total_seconds_samples": (
                        native_phase_total_samples
                    ),
                    "native_profile_total_seconds_median": (
                        native_phase_total_median
                    ),
                    "native_phase_median_fractions": native_phase_fractions,
                }
            )
        if compact_seconds is not None:
            compact_workspace_elements = (
                compact_U.numel()
                + compact_Y.numel()
                + compact_d.numel()
                + compact_Q.numel()
                + compact_factor_workspace.numel()
                + compact_buffer.numel()
                + compact_scratch.numel()
                + compact_prefix.numel()
                + compact_apply_workspace.numel()
                + compact_coefficient.numel()
            )
            row.update(
                {
                    "compact_seconds": compact_seconds,
                    "legacy_over_compact_speedup": (
                        sequential_seconds / compact_seconds
                    ),
                    "refactor_over_compact_speedup": (
                        refactor_seconds / compact_seconds
                    ),
                    "compact_L_relative_error": compact_l_error,
                    "compact_Aw_relative_error": compact_aw_error,
                    "compact_vs_legacy_L_relative_error": (
                        compact_l_legacy_error
                    ),
                    "compact_vs_legacy_Aw_relative_error": (
                        compact_aw_legacy_error
                    ),
                    "compact_block_size": COMPACT_BLOCK_SIZE,
                    "compact_apply_backend": compact_apply_backend,
                    "compact_cpp_max_k": COMPACT_CPP_MAX_K,
                    "compact_workspace_elements": compact_workspace_elements,
                    "compact_workspace_bytes": (
                        compact_workspace_elements * L.element_size()
                    ),
                    "compact_dense_workspace_matrices": 2,
                    "compact_timed_dense_allocations": 0,
                    "compact_phase_names": list(COMPACT_PHASE_NAMES),
                    "compact_phase_seconds_samples": compact_phase_samples,
                    "compact_phase_seconds_medians": compact_phase_medians,
                    "compact_profile_total_seconds_samples": (
                        compact_phase_total_samples
                    ),
                    "compact_profile_total_seconds_median": (
                        compact_phase_total_median
                    ),
                    "compact_phase_median_fractions": (
                        compact_phase_fractions
                    ),
                }
            )
        if timing_samples is not None:
            row["timing_seconds_samples"] = timing_samples
            row["timing_method_order_schedule"] = timing_order_schedule
        rows.append(row)
        native_text = (
            f" native={1e3*native_seconds:.3f}ms "
            f"legacy/native={sequential_seconds/native_seconds:.3f}x"
            if native_seconds is not None
            else ""
        )
        compact_text = (
            f" compact={1e3*compact_seconds:.3f}ms "
            f"legacy/compact={sequential_seconds/compact_seconds:.3f}x "
            f"refact/compact={refactor_seconds/compact_seconds:.3f}x"
            if compact_seconds is not None
            else ""
        )
        print(
            f"r={r} k={k} sequential={1e3*sequential_seconds:.3f}ms "
            f"{native_text}"
            f"{compact_text}"
            f" refact={1e3*refactor_seconds:.3f}ms "
            f"speedup={row['speedup']:.3f}x",
            flush=True,
        )

    payload = {
        "experiment": "rank_k",
        "rank": r,
        "seed": args.seed,
        "threads": args.threads,
        "implementation": args.implementation,
        "pilot": args.pilot,
        "confirmation": args.confirmation,
        "campaign_tag": (
            CONFIRMATION_CAMPAIGN if args.confirmation else None
        ),
        "pilot_variant": PILOT_VARIANT if args.pilot else None,
        "confirmation_protocol": (
            confirmation_protocol if args.confirmation else None
        ),
        "confirmation_protocol_sha256": (
            _sha256(CONFIRMATION_PROTOCOL) if args.confirmation else None
        ),
        "predicted_flop_break_even_k": (7.0 / 54.0) * r,
        "rows": rows,
        "native_diagnosis": (
            {
                "legacy_python_calls_per_case": "k",
                "native_python_calls_per_case": 1,
                "legacy_dense_operator_allocations_per_case": "k",
                "native_timed_dense_operator_allocations_per_case": 0,
                "native_workspace_reused_across_repetitions": True,
                "native_operator_updated_in_place": True,
                "remaining_operator_sweeps_per_update": 2,
                "remaining_expected_bottleneck": (
                    "the sequential left and right dense operator sweeps; "
                    "a compact blocked rank-k transformation is not yet used"
                ),
            }
            if use_native
            else None
        ),
        "compact_diagnosis": (
            {
                "identity": (
                    "U=L^-1 Z, Y=L^-1 W, H=I+UU^T=CC^T, P=C^-1; "
                    "L+=LC and Aw+=P(Aw+UY^T)P^T"
                ),
                "semiseparable_inverse": (
                    "P_ii=1/d_i and P_ij=-q_i^T u_j for j<i"
                ),
                "factor_complexity": "O(r*k^2 + r^2*k)",
                "operator_complexity": "O(r^2*k)",
                "operator_dense_sweeps": 3,
                "operator_dense_sweep_explanation": (
                    "rank-k correction plus fused direct left and right "
                    "compact applications; no dense transpose/copy pass"
                ),
                "timed_dense_allocations": 0,
                "workspace_reused_across_repetitions": True,
                "block_size": COMPACT_BLOCK_SIZE,
                "cpp_row_recurrence_max_k": COMPACT_CPP_MAX_K,
            }
            if use_compact
            else None
        ),
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "process_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_job_qos": os.environ.get("SLURM_JOB_QOS"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "source_commit": (
                os.environ.get("SOURCE_COMMIT") if args.confirmation else None
            ),
            "driver_sha256": _sha256(Path(__file__).resolve()),
            "kernel_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "student_pkg"
                / "fast_update.cpp"
            ),
            "extension_path": fast_cholesky.__file__,
            "extension_sha256": _sha256(Path(fast_cholesky.__file__).resolve()),
        },
    }
    if args.pilot:
        output = result_path(
            "rank_k_pilot",
            (
                f"rank_k_compact_pilot_r{r}_seed{args.seed}"
                f"_{PILOT_VARIANT}"
            ),
        )
        if output.exists():
            raise FileExistsError(f"refusing to overwrite pilot result: {output}")
    elif args.confirmation:
        output = (
            ROOT
            / "rebuttal_results"
            / "rank_k_confirmation"
            / CONFIRMATION_CAMPAIGN
            / f"rank_k_compact_r{r}_seed{args.seed}.json"
        )
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite confirmation result: {output}"
            )
    else:
        output = result_path("rank_k", f"rank_k_r{r}_seed{args.seed}")
    atomic_json_dump(payload, output)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--implementation",
        choices=("legacy", "native", "both", "compact", "all"),
        default="legacy",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "require the frozen seed-1000 r=1024 compact pilot configuration "
            "and write to rank_k_pilot without overwriting"
        ),
    )
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help=(
            "require the frozen multi-seed compact E6 confirmation "
            "configuration and write a non-overwriting campaign artifact"
        ),
    )
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
