# data/

Local, gitignored inputs. Nothing here is needed for the default (synthetic) ECG run.

* `raw/mitdb/100.dat`, `100.hea` — PhysioNet MIT-BIH Arrhythmia Database record 100 (360 Hz,
  two channels; we use channel 0, MLII), fetched by `ptqnn.tasks.ecg.data.load_physionet` via
  `wfdb` (`uv sync --extra ecg`) and cached here (`cache_dir`). Licence: PhysioNet open-data
  (ODC-By 1.0); cite Moody & Mark, IEEE Eng Med Biol 20(3) 2001, and Goldberger et al.,
  Circulation 101(23) 2000, if published.

The paper does not name its ECG recording `[PTQNN §III.D]`; record 100 is our `[ASSUMPTION]`
and Table I is reported for both the synthetic signal and this record
(`experiments/runs/ecg/20260820T162014` and `…165912`). Iris comes from scikit-learn's bundled
copy; Flappy Bird is a headless clone with no data.
