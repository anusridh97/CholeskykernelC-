# Local validation of the corrected CPU application runners

These results demonstrate the recovered corrected implementation and new entry
points. They are not original submitted-paper BO/NG tables.

- bo-smoke: synthetic landscape, rank 8, seed 0, two rounds.
- ng-smoke: synthetic regression, rank 8, seed 42, one epoch, both chart modes.
- ng-protein: recovered Protein MAT, rank 64, seed 42, five epochs,
  hidden width 128, both chart modes, native signed CPU extension.

Each directory retains raw curves, summary.csv, source/input hashes, environment
and complete configuration in run.json. Run status indicates successful
execution; numerical oracle tests are separate. Timing order is fixed and
initial warmup is not excluded, as in the recovered driver. These local timings
are not a new performance claim. Dataset bytes are excluded from Git.

The Protein input SHA-256 is 5ce538fa05f6834de3acc727e220e777a02fb1ebc3a7bfebcb5aaad9769fed13.
The GB1 arrays were unavailable; no real GB1 result is claimed.
