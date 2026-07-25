# UCI Protein data used by corrected E2

`protein.mat` is the recovered MATLAB-format copy of the UCI Machine Learning
Repository dataset “Physicochemical Properties of Protein Tertiary Structure”
(dataset 265):

- Prashant Rana (2013), DOI: <https://doi.org/10.24432/C5QW3H>
- Source: <https://archive.ics.uci.edu/dataset/265/>
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- File size: 1,725,537 bytes
- SHA-256:
  `5ce538fa05f6834de3acc727e220e777a02fb1ebc3a7bfebcb5aaad9769fed13`

The corrected E2 driver reads this exact file and then creates its fixed split
and training-only normalization. The imported repository did not preserve the
preprocessing/permutation recipe that produced this byte-level `.mat` file
from `data/uci_protein/CASP.csv`, so the two files should not be assumed to be
byte-reconstructible from one another. Tracking this small recovered artifact
preserves exact E2 input provenance.
