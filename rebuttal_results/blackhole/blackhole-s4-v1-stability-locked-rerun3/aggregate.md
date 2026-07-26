# Locked black-hole stability trace

Outer campaign `blackhole-s4-v1-stability-locked-rerun3`; unmodified inner protocol `blackhole-s4-v1` at `96c8018cdd0b56ffe1fd93bddf48fa4affb34697`.

| rank | method | event seconds, mean ± SD | total event seconds | valid |
|---:|:---|---:|---:|:---:|
| 8192 | fresh_refact | 0.0398066 ± 4.31e-05 | 3.98066 | yes |
| 8192 | inverse_no_refresh | 0.00792455 ± 4.27e-05 | 0.792455 | yes |
| 8192 | inverse_refresh_q50 | 0.00903588 ± 0.00776 | 0.903588 | yes |
| 32768 | fresh_refact | 1.76872 ± 0.00268 | 176.872 | yes |
| 32768 | inverse_no_refresh | 0.057676 ± 0.000217 | 5.7676 | yes |
| 32768 | inverse_refresh_q50 | 0.10732 ± 0.349 | 10.732 | yes |

Paired speedups are emitted only when the frozen correctness gate passes.
