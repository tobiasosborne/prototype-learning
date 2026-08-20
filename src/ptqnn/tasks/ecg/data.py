"""ECG signals, additive noise and sliding windows ``[PTQNN §III.D, Fig 12, 13]``.

The paper does not name its recording: "As time-series data an ECG-signal use case has been
used. The goal is to denoise a signal overlaid with stochastic noise." ``[PTQNN §III.D]`` — no
database, no sampling rate, no noise level.  We therefore supply two sources and report both:

* :func:`synthetic_ecg` — a deterministic sum-of-Gaussians beat model, tuned to the landmarks
  visible in Fig 13 so that a run is reproducible without network access;
* :func:`load_physionet` — MIT-BIH Arrhythmia record 100, channel 0, via ``wfdb``.

Fig 13 read-off (docs/physics/rosenhahn_hirche_ptqnn.md §6.3, measured from the raster; every
default below is tuned to these): 1800 plotted samples on an axis running to 2000; ground truth
spanning 0.117 … 0.903; two large beats at t ≈ 370 and t ≈ 1269 (separation ≈ 899 samples) with
amplitudes 0.866 and 0.853, and five smaller humps (t ≈ 80, 732, 797, 1003, 1573; 0.334 … 0.378);
troughs at 0.13 … 0.20.  The noisy trace has peak-to-peak noise ≈ 0.094 (amplitude ≈ ±0.047) and
a full range of 0.052 … 0.954.  "The waveform is a smooth quasi-periodic envelope rather than a
QRS-shaped ECG" — Fig 13 is a low-pass-filtered / decimated view, and our default waveform
matches *that* view, not a raw clinical trace.
"""

from __future__ import annotations

import socket
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

Real = npt.NDArray[np.float64]

# [ASSUMPTION] Noise level.  The paper gives none.  σ = 0.02 is fixed by three independent
# reads, which agree to within 10 %:
#   (i) Fig 13's noisy trace spans 0.052 … 0.954 against a ground truth of 0.117 … 0.903, i.e.
#       the largest excursions are 0.065 and 0.051; over 1800 Gaussian draws E max|z| ≈ 3.3 σ,
#       giving σ ≈ 0.018;
#  (ii) the measured peak-to-peak noise of ≈ 0.094 (amplitude ±0.047) is ≈ 2.4 σ either way;
# (iii) with R = max − min ≈ 0.79 the *noisy input itself* scores 20 log10(0.79/0.02) = 31.9 dB,
#       which is where Table I's single-neuron rows sit (CQNN 32.72, PTQNN 32.9) — the smallest
#       models barely denoise, so their PSNR is essentially the input PSNR.
DEFAULT_SIGMA = 0.02


@dataclass(frozen=True)
class Wave:
    """One Gaussian PQRST component: centre ``theta_deg`` in the beat phase (R peak = 0°),
    amplitude ``a`` and width ``b_deg`` (standard deviation, in degrees of beat phase)."""

    name: str
    theta_deg: float
    a: float
    b_deg: float


# ``[McSharry 2003]`` P. E. McSharry, G. D. Clifford, L. Tarassenko, L. A. Smith, "A dynamical
# model for generating synthetic electrocardiogram signals", IEEE Trans. Biomed. Eng. 50(3),
# 289–294 (2003).  Verbatim from the reference implementation ``ecgsyn.m`` (PhysioNet
# https://physionet.org/files/ecgsyn/1.0.0/Matlab/ecgsyn.m, retrieved 2026-08-20):
#     ti = [-70 -15 0 15 100];        (P Q R S T angles, degrees)
#     ai = [1.2 -5 30 -7.5 0.75];
#     bi = [0.25 0.1 0.1 0.1 0.4];    (radians)
#     sfecg = 256;  hrmean = 60;
# In that model the a_i/b_i drive *dz/dt* (z is the integral of a sum of θ-weighted Gaussians);
# we use the closely related and far simpler "sum of Gaussians per beat" waveform
# ``z(θ) = Σ_i a_i exp(−Δθ_i²/(2 b_i²))``, for which the same θ_i are the wave centres.  The
# amplitudes/widths therefore cannot be carried over verbatim, and MCSHARRY_WAVES below keeps
# the pinned angles while rescaling a_i, b_i for the sum-of-Gaussians form.
MCSHARRY_WAVES: tuple[Wave, ...] = (
    Wave("P", -70.0, 0.10, 0.25 * 180 / np.pi / 4),
    Wave("Q", -15.0, -0.12, 0.1 * 180 / np.pi / 4),
    Wave("R", 0.0, 1.00, 0.1 * 180 / np.pi / 4),
    Wave("S", 15.0, -0.25, 0.1 * 180 / np.pi / 4),
    Wave("T", 100.0, 0.22, 0.4 * 180 / np.pi / 4),
)

# [ASSUMPTION] Fig-13 waveform.  Same functional form, parameters fitted by hand to the
# landmarks listed in the module docstring: a broad pre-beat hump at −109° (= −270 samples at a
# 890-sample beat), the smooth QRS peak at 0°, a broad post-beat hump at +150° (= +370 samples).
# The widths (σ ≈ 27° ≈ 67 samples) are what makes the trace as smooth as Fig 13's green curve.
FIG13_WAVES: tuple[Wave, ...] = (
    Wave("P", -109.0, 0.235, 26.0),
    Wave("R", 0.0, 0.78, 27.0),
    Wave("T", 150.0, 0.245, 26.0),
)


def _beat(phase_deg: Real, waves: Sequence[Wave]) -> Real:
    """``z(θ) = Σ_i a_i exp(−Δθ_i²/(2 b_i²))`` with ``Δθ`` wrapped into (−180°, 180°]."""
    z = np.zeros_like(phase_deg)
    for w in waves:
        d = (phase_deg - w.theta_deg + 180.0) % 360.0 - 180.0
        z += w.a * np.exp(-(d**2) / (2.0 * w.b_deg**2))
    return z


def _canonical_range(waves: Sequence[Wave]) -> tuple[float, float]:
    """Min and max of one jitter-free beat, sampled densely — the affine normalisation must not
    depend on ``n_samples`` or on the seed, or short excerpts would be scaled differently."""
    z = _beat(np.linspace(-180.0, 180.0, 3601), waves)
    return float(z.min()), float(z.max())


def synthetic_ecg(
    n_samples: int = 1800,
    seed: int = 0,
    fs: float = 1000.0,
    heart_rate: float = 66.74,
    waves: Sequence[Wave] = FIG13_WAVES,
    first_peak: int = 370,
    lo: float = 0.117,
    hi: float = 0.903,
    rr_jitter: float = 0.02,
    amplitude_jitter: float = 0.03,
) -> Real:
    """A clean ECG-like signal in ``[0, 1]``, shaped like the green curve of ``[PTQNN Fig 13]``.

    Sum of Gaussian PQRST kernels in the beat phase, the waveform family of ``[McSharry 2003]``
    (see :data:`MCSHARRY_WAVES` for the pinned angles).  Defaults: ``fs = 1000`` Hz and
    ``heart_rate = 67.4`` bpm give a beat every ``60·fs/hr = 890`` samples, i.e. exactly the two
    beats per 1800 samples of Fig 13 at that figure's peak positions (t ≈ 380 and 380 + 890 =
    1270).  [ASSUMPTION] Fig 13's abscissa cannot be raw MIT-BIH samples: 890 samples at the
    database's 360 Hz would be 24 bpm.  Either the authors resampled or the axis counts
    something else; we keep the *plotted* geometry and choose fs so the rate is physiological.

    ``rr_jitter`` (relative s.d. of the RR interval) and ``amplitude_jitter`` (relative s.d. of
    the per-beat amplitude) are driven by ``seed``, so different seeds give different — but
    equally Fig-13-like — signals; set both to 0 for a strictly periodic trace.  The result is
    mapped affinely so that a jitter-free beat spans ``[lo, hi] = [0.13, 0.93]`` (Fig 13's
    visible band) and then clipped into ``[0, 1]``.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    period = 60.0 * fs / heart_rate
    t = np.arange(n_samples, dtype=np.float64)

    # R-peak times, walking outwards from `first_peak` so the first beat sits where Fig 13 has
    # it, plus one beat of margin on each side so the phase interpolation covers every sample.
    peaks: list[float] = [float(first_peak)]
    while peaks[-1] < n_samples + period:
        peaks.append(peaks[-1] + period * (1.0 + rr_jitter * rng.normal()))
    while peaks[0] > -period:
        peaks.insert(0, peaks[0] - period * (1.0 + rr_jitter * rng.normal()))
    pk = np.asarray(peaks, dtype=np.float64)

    # Phase-driven synthesis (the structure of [McSharry 2003], whose state is a phase on a
    # limit cycle): the beat phase advances by exactly 360° between consecutive R peaks, so a
    # jittered RR interval stretches the beat instead of tearing the waveform at the seam.
    phase = np.interp(t, pk, 360.0 * np.arange(pk.size))
    gains = 1.0 + amplitude_jitter * rng.normal(size=pk.size)
    z = np.interp(t, pk, gains) * _beat((phase + 180.0) % 360.0 - 180.0, waves)

    z_lo, z_hi = _canonical_range(waves)
    s = lo + (hi - lo) * (z - z_lo) / (z_hi - z_lo)
    return np.clip(s, 0.0, 1.0)


def load_physionet(
    record: str = "100",
    n_samples: int = 1800,
    channel: int = 0,
    start: int = 0,
    database: str = "mitdb",
    cache_dir: str | Path = "data/raw",
    timeout: float = 30.0,
) -> Real:
    """MIT-BIH Arrhythmia record ``record``, channel ``channel``, min–max normalised to ``[0, 1]``.

    The paper does not say which recording it used ``[PTQNN §III.D]``; record 100 is the
    conventional default.  The record is downloaded once through ``wfdb`` and cached under
    ``cache_dir`` (gitignored); later calls are offline.  A failed download raises with the
    manual-download instruction rather than silently falling back to synthetic data.
    """
    cache = Path(cache_dir) / database
    header = cache / f"{record}.hea"
    if not header.exists():
        cache.mkdir(parents=True, exist_ok=True)
        try:
            import wfdb  # optional dependency, imported on use

            old = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            try:
                wfdb.dl_files(database, str(cache), [f"{record}.hea", f"{record}.dat"])
            finally:
                socket.setdefaulttimeout(old)
        except Exception as exc:  # any failure means "no data": one message, one action
            raise RuntimeError(
                f"could not fetch PhysioNet {database}/{record} into {cache} ({exc!r}). "
                f"Download {record}.hea and {record}.dat from "
                f"https://physionet.org/files/{database}/1.0.0/ into that directory, or use "
                "--source synthetic."
            ) from exc
    import wfdb

    rec = wfdb.rdrecord(str(cache / record), sampfrom=start, sampto=start + n_samples)
    sig = np.asarray(rec.p_signal, dtype=np.float64)[:, channel]
    if sig.size < n_samples:
        raise RuntimeError(f"record {record} yielded {sig.size} < {n_samples} samples")
    span = float(sig.max() - sig.min())
    if span <= 0:
        raise RuntimeError(f"record {record} channel {channel} is constant")
    return (sig - float(sig.min())) / span


def add_noise(clean: Real, sigma: float = DEFAULT_SIGMA, seed: int = 0) -> Real:
    """``s̃ = s + N(0, σ²)`` — "a signal overlaid with stochastic noise" ``[PTQNN §III.D]``.

    Not clipped to ``[0, 1]``: the noisy trace of Fig 13 does leave the ground-truth band, and
    clipping would bias the input the network sees.  See :data:`DEFAULT_SIGMA` for σ.
    """
    rng = np.random.default_rng(seed)
    return np.asarray(clean, dtype=np.float64) + sigma * rng.normal(size=np.shape(clean))


Edge = Literal["reflect", "valid"]


def windows(signal: Real, half_width: int = 2, edge: Edge = "reflect") -> tuple[Real, Real]:
    """Sliding windows ``[s(t−2) … s(t+2)] → s(t)`` ``[PTQNN §III.D, Fig 12]``.

    Returns ``(X, y)`` with ``X[i]`` the window centred at sample ``i`` (or at ``i + half_width``
    in ``"valid"`` mode) and ``y[i]`` its centre sample, so ``y[i] == X[i, half_width]``.

    Edge handling — the paper is silent, and there are only two sensible choices:

    * ``"reflect"`` (default): the signal is mirror-padded by ``half_width`` samples
      (``[s2 s1 s0 s1 s2 …]``), so ``N = len(signal)`` windows come out and the denoised trace
      covers the whole abscissa, as in Fig 13 where the blue curve starts at t = 0.
    * ``"valid"``: no padding, ``N = len(signal) − 2·half_width`` windows, all made of real
      samples.  Metrics differ from ``"reflect"`` only in the ``2·half_width`` edge samples.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    w = 2 * half_width + 1
    if s.size < w:
        raise ValueError(f"signal of length {s.size} is shorter than the window ({w})")
    if edge == "reflect":
        s = np.pad(s, half_width, mode="reflect")
    n = s.size - 2 * half_width
    X = np.lib.stride_tricks.sliding_window_view(s, w)[:n].copy()
    return X, X[:, half_width].copy()


def supervised_windows(
    noisy: Real, clean: Real, half_width: int = 2, edge: Edge = "reflect"
) -> tuple[Real, Real]:
    """Training pairs: windows of the *noisy* signal, targets from the *clean* one."""
    X, _ = windows(noisy, half_width, edge)
    _, y = windows(clean, half_width, edge)
    if X.shape[0] != y.shape[0]:
        raise ValueError("noisy and clean signals must have the same length")
    return X, y


def contiguous_split(
    n_windows: int, train_frac: float = 0.5, half_width: int = 2
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Split window indices into two contiguous blocks with a guard band.

    Windows overlap, so a random split leaks: window ``i`` shares four of its five samples with
    window ``i+1``.  We cut the *signal* in two and drop the ``2·half_width`` windows straddling
    the cut, after which no training window shares a sample with any test window.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    cut = round(train_frac * n_windows)
    train = np.arange(0, max(cut - half_width, 0), dtype=np.int64)
    test = np.arange(min(cut + half_width, n_windows), n_windows, dtype=np.int64)
    if train.size == 0 or test.size == 0:
        raise ValueError("split leaves an empty side; use more windows")
    return train, test
