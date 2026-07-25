# Frozen protocol: real-data black-hole rank scaling

Campaign: `blackhole-s4-v1`

Reviewer target: URn7, question 2 (“analyze scalability when rank becomes
large”)

Protocol status: frozen before feature generation or any benchmark timing
result was observed.

## Scientific question

This experiment tests whether the experimental inverse-factor representation
used in E5 preserves its expected quadratic-versus-cubic scaling advantage on
updates derived from a real high-dimensional physical trajectory, rather than
synthetic Gaussian vectors.

It is a streaming Koopman/operator-regression workload, not a predictive-model
accuracy comparison. For a physical state \(x_t\), a fixed feature map gives
\(\phi(x_t)\). At rank \(r\),

\[
z_t = \phi_r(x_t)/s_r,\qquad
w_t = \phi_r(x_{t+1})/s_r.
\]

The maintained state after the calibration window is

\[
G_0 = I + \sum_{t=0}^{63} z_t z_t^\top,\qquad
M_0 = \sum_{t=0}^{63} z_t w_t^\top,
\]

with \(L_0L_0^\top=G_0\),
\(L_0^{-1}\), and
\(A_{w,0}=L_0^{-1}M_0L_0^{-\top}\).
The ridge coefficient is fixed at one.

## Pinned real dataset

The primary dataset is The Well
[`post_neutron_star_merger`](https://huggingface.co/datasets/polymathic-ai/post_neutron_star_merger),
training scenario 4 (`torus_MBH_10`), a disk from a black-hole–neutron-star
merger with a 10-solar-mass black hole.

- Hugging Face revision:
  `1fd582f3e29749ac4d0c14bc816bddb2394258cc`
- Repository path:
  `data/train/post_neutron_star_merger_scenario_4.hdf5`
- Exact size: `14,109,638,656` bytes
- SHA-256 / Hugging Face linked-object digest:
  `21e587c90c006c4b7d37e2b04899bd7e928967ae506e9ceb5afd05dfca6909fd`
- Shape: one trajectory, 181 snapshots, spatial grid \(192\times128\times66\)
- Dynamic channels: density, internal energy, pressure, temperature, electron
  fraction, entropy, three magnetic-field components, and three velocity
  components
- Exclusion: the time-independent contravariant spacetime metric is not a
  dynamic channel

The raw file must pass byte-count, SHA-256, root-attribute, field-name,
shape, dtype, and HDF5-layout checks before preprocessing.

## Frozen feature map

1. Channel order is:
   density, internal energy, pressure, temperature, electron fraction,
   entropy, magnetic field in `(log_r, theta, phi)` order, then velocity in
   the same component order.
2. Per-channel scalar mean and population standard deviation are accumulated
   in FP64 over every grid point in snapshots 0–63. Every snapshot is
   standardized with those fixed statistics.
3. Each standardized channel receives an orthonormal separable 3-D DCT-II.
4. Spatial frequencies are ordered by

   \[
   (k_{\log r}/192)^2+(k_\theta/128)^2+(k_\phi/66)^2,
   \]

   with ascending lexicographic index as the exact tie-break.
5. Coefficients are interleaved frequency-major and channel-minor. A single
   contiguous FP32 trace of shape `(181, 65536)` is persisted. Every smaller
   rank is an exact prefix of that trace.
6. For each tested rank,

   \[
   s_r^2=\frac1{64}\sum_{t=0}^{63}
   \left\|\phi_r(x_t)\right\|_2^2.
   \]

   The persisted trace is unscaled; consumers divide each prefix by \(s_r\).

The tested ranks are
\(2{,}048,4{,}096,8{,}192,16{,}384,32{,}768,49{,}152,65{,}536\).
Their frozen inverse-kernel tile sizes are respectively
\(8,8,8,16,32,32,32\); setup warps are four and sweep warps are one.
Timing repeats are 10 through rank 8,192, five through rank 32,768, and three
at larger ranks, matching the final E5 ladder.

## Isolated real-transition ladder

Transition 64 is used only for compilation/warm-up. The ten evaluation
transitions are fixed as

`80, 91, 102, 113, 124, 135, 146, 157, 168, 179`.

Each transition is applied separately to the same state after transition 63.
Even-numbered cases execute the inverse path first; odd-numbered cases execute
fresh refactorization first.

The timed interval includes the dense \(G\) and \(M\) outer-product updates
and either:

- `inverse`: the in-place inverse-factor update of \(L^{-1}\) and \(A_w\); or
- `fresh_refact`: Cholesky factorization followed by the two triangular solves
  that form the newly whitened operator.

HDF5 reads, DCT work, host/device transfer, initial-state construction,
compilation, and restoration of prepared state are outside the timed interval
and reported separately. Every timing sample is retained. Each case is
summarized by its within-case median; speedup is formed within transition, then
reported across the ten transitions as mean, sample standard deviation, and a
95% Student-\(t\) interval. There is no timing-based filtering.

## Correctness and conditioning

Every row must have Cholesky `info=0` and finite output in every matrix
element. Sixteen normalized Rademacher probes with seed 29421 compare:

\[
L_{\mathrm{inc}}^{-1}P
\quad\text{with}\quad
\operatorname{solve}(L_{\mathrm{fresh}},P),
\]

\[
A_{w,\mathrm{inc}}P
\quad\text{with}\quad
A_{w,\mathrm{fresh}}P,
\]

and also check

\[
L^{-1}GL^{-\top}P\approx P,\qquad
A_wP\approx L^{-1}ML^{-\top}P.
\]

Full dense relative-error comparisons are retained through rank 32,768.
Probe checks extend through rank 65,536. A row is headline-valid only if every
predeclared relative error is at most \(10^{-3}\); failures remain reported
and do not trigger retuning.

Because \(G=I+ZZ^\top\) and there are at most 65 stored update vectors in an
isolated case, its exact spectral condition number is obtained from the small
Gram matrix:

\[
\kappa(G)=1+\lambda_{\max}(Z^\top Z).
\]

## Repeated-update secondary trace

At ranks 8,192 and 32,768, three methods independently start from the same
state after transition 63 and process transitions 64–163 in order:

- fresh refactorization at every step;
- inverse-factor updates without refresh; and
- inverse-factor updates with the already frozen \(q=50\) refresh interval.

Refresh work is included in timing. Diagnostics are recorded after steps
1, 10, 25, 50, 75, and 100. The refresh interval will not be tuned from these
outcomes.

## Hardware, provenance, and stopping rules

The final timing campaign requires one NVIDIA H100 with exactly
85,045,477,376 bytes of device memory. Every artifact records the raw-data,
feature-trace, feature-manifest, benchmark-driver, and CUDA-kernel source
hashes; software versions; GPU identity; Slurm job; and source Git commit.

Preprocessing, ladder, and repeated-update stages fail closed on any
provenance, schema, hardware, rank, transition, method-order, sample-count, or
correctness mismatch. Configuration is not changed in response to performance
or numerical outcomes. The black-hole result is the primary real-data result
regardless of whether its measured speedup is favorable.
