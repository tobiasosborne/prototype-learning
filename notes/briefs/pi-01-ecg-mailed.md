# Brief pi-01 — ECG: add the co-author's "original" dataset as a data source

You are a coding agent working in the Python repo at /home/tobias/Projects/prototype-learning
(package `src/ptqnn`, uv-managed venv, Python 3.12). Work ONLY in these files:
`src/ptqnn/tasks/ecg/data.py`, `src/ptqnn/tasks/ecg/train.py`, `experiments/ecg_table1.py`,
`tests/test_ecg.py`, `docs/tasks.md` (ECG paragraph only), `data/README.md`. Do NOT commit.
Do NOT touch any other file. Do not run anything that takes more than ~2 minutes.

## Context
The repo reproduces the ECG-denoising task of the paper "Prototype Learning for Quantum Neural
Networks" (Table I, Fig 12/13). Today the task supports two data sources, `source="synthetic"`
and `source="physionet"` (see `train.prepare`, `train.Source`, `experiments/ecg_table1.py
--source`). The co-author Bodo Rosenhahn e-mailed the ORIGINAL signal on 2026-08-21
("Please find attached "original" ECG-dataset"). The files are staged (gitignored) at
`data/raw/ecg-rosenhahn/ecgfilteredData.mat` (MATLAB variable `ecg_filtered`) and
`data/raw/ecg-rosenhahn/ecgnoiseData.mat` (variable `ecg_noisy`). Each is a (1, 7700) float64
row vector, already in [0, 1]; `ecg_noisy` is clipped to exactly [0, 1]; noisy − filtered has
std ≈ 0.0354. The first three samples of BOTH are 0.13049, 0.16907, 0.12890. The first 1800
samples of `ecg_filtered` span 0.1112…0.9008 (the paper's Fig 13 spans 0.117…0.903).
SHA256: filtered 9ed65523b30e588012dba0ece34e8d0bb6951254357cf0757ef1692b7439b456,
noisy cc94548d1c40b63b597b532edbf0bd198ac3a12ac9253d0efc4b40255d0753b9.

## Repo rules you must follow
- Every physics/data fact in a docstring is pinned to a source, e.g. `[Rosenhahn e-mail 2026-08-21]`.
  Our own choices are marked `[ASSUMPTION]`.
- Red → green TDD: write the failing tests FIRST in `tests/test_ecg.py`, run them and see them
  fail, then implement. Tests that need the .mat files use
  `pytest.mark.skipif(not DATA_DIR.exists(), reason=...)`.
- Gates before you finish (all must pass):
  `uv run ruff check src tests experiments && uv run ruff format src tests experiments &&
   uv run mypy src && uv run pytest tests/test_ecg.py -q`.
- Read `src/ptqnn/tasks/ecg/data.py`, `train.py` and `experiments/ecg_table1.py` fully before
  editing; mirror their style (type hints, `Real` alias, docstrings with pins).

## Deliverable
1. `data.py`:
   - `MAILED_DIR = Path(__file__).resolve().parents[4] / "data" / "raw" / "ecg-rosenhahn"`
     (check the parents count actually lands on the repo root; the repo root contains `pyproject.toml`).
   - `load_mailed(directory: Path = MAILED_DIR, n_samples: int | None = 1800) -> tuple[Real, Real]`
     returning `(clean, noisy)` as 1-D float64 arrays via `scipy.io.loadmat`, keys
     `ecg_filtered` / `ecg_noisy`, sliced to the first `n_samples` when not None. Raise a clear
     `FileNotFoundError` naming the expected files when absent. Do NOT renormalise, do NOT clip.
   - `mailed_provenance(directory: Path = MAILED_DIR) -> dict[str, str]` returning the SHA256 hex
     digest of each file keyed by file name, plus `"sender": "B. Rosenhahn, e-mail 2026-08-21"`.
   - Docstring: quote the e-mail line above, state `[ASSUMPTION]` that this is the signal behind
     Fig 13 (evidence: the 1800-sample range match), state that the noise is already present in
     the file so `add_noise` is NOT applied, and that its empirical std (≈0.035) differs from
     `DEFAULT_SIGMA`.
2. `train.py`: extend `Source` with `"mailed"`; in `prepare`, for `source == "mailed"` call
   `data.load_mailed(n_samples=n_samples, **source_kwargs)`, use the file's noisy trace directly
   (no `add_noise`), and set the dataset's `sigma` to `float(np.std(noisy - clean))` (document
   that for this source sigma is measured, not applied). Keep every existing code path unchanged.
3. `experiments/ecg_table1.py`: add `"mailed"` to `--source` choices; add `--mailed-dir`
   (default `data.MAILED_DIR`); let `--n-samples 0` mean "the full trace" (pass `None`); write
   `data.mailed_provenance(...)` into the `meta.json` under key `"mailed"` when the source is
   mailed (`None` otherwise). Update the module docstring usage lines.
4. Tests (`tests/test_ecg.py`, new tests only, keep existing ones):
   - loader returns two float64 arrays of shape (7700,) with `n_samples=None`, (1800,) by default;
   - first three samples of both equal the values above to 1e-5;
   - clean lies within [0, 1]; noisy min == 0.0 and max == 1.0 exactly;
   - `np.std(noisy - clean)` within 0.030…0.040 on the full trace;
   - `prepare(source="mailed", n_samples=1800)` has `source == "mailed"`, sigma ≈ std above
     (tolerance 0.005), and the same number of windows as the synthetic default;
   - `mailed_provenance` returns the two SHA256 digests above.
5. Docs: one sentence in `docs/tasks.md` (ECG section) and one bullet in `data/README.md`
   describing the new source, its location, and the provenance rule.

Finish by printing: the list of files changed, the test output summary line, and any
`[ASSUMPTION]` you had to make that is not listed above.
