"""Iris → 12-bit one-hot-per-feature codes ``[SNN §IV.B]``, ``[PTQNN §III.C]``.

``[SNN §IV.B]``: "Taking the iris dataset as a toy example, it consists of 4 dimensional data
encoding sepal length, sepal width, petal length and petal width. After separating training and
test data, a kMeans clustering on each dimension with k = 3 is used on the training data. Thus,
every datapoint can be encoded in a 4 × 3 = 12−dimensional binary vector which contains exactly
4 non-zero entries. For the given cluster centers, the same can be done with the test data."

``[PTQNN §III.C]``: "each binary encoding corresponds to a unique entry in a 2^12-dimensional
vector space. Thus, 12 qubits can be brought in a superposition to represent the entire dataset."

Bit order: bit ``3·f + c`` of the 12-bit word is "feature ``f`` fell in cluster ``c``", clusters
sorted by increasing centre.  The word is read as a computational-basis index with the *first*
bit most significant (``ptqnn.circuits.ir`` convention, qubit 0 = MSB), so the basis index of a
sample is ``Σ_j b_j 2^(11-j)`` and qubit 0 carries "feature 0, cluster 0".

Choices the papers do not fix (``PLAN.md §0``):

* ``[ASSUMPTION]`` train/test ratio.  Neither paper states it.  We use a stratified 1/3 train,
  2/3 test split: ``[PTQNN Fig 8]`` shows 15 histogram bins for the training set and 30 for the
  test set, i.e. a test set roughly twice the size of the training set; a 50-sample training
  split gives 12–17 distinct codes, matching the 15 bins of Fig 8 (top).  ``[SNN Table I]``
  reports 95 % ± 5 on Iris with an unstated split.
* ``[ASSUMPTION]`` code collisions.  The binarisation is lossy: two samples of different classes
  can share a 12-bit code, and a quantum state cannot hold two amplitudes for one basis state.
  We merge duplicates and average the labels (:func:`merge_duplicate_codes`), and report how
  many distinct codes carried more than one class.  This caps the attainable accuracy at
  96–99 % on our splits — in the range of the paper's "98 %".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

Codes = npt.NDArray[np.int64]
Real = npt.NDArray[np.float64]

N_FEATURES = 4
"""sepal length, sepal width, petal length, petal width ``[SNN §IV.B]``."""

K = 3
"""``"a kMeans clustering on each dimension with k = 3"`` ``[SNN §IV.B]``."""

N_BITS = N_FEATURES * K
"""``"a 4 × 3 = 12−dimensional binary vector"`` ``[SNN §IV.B]``."""


def load_iris_xy() -> tuple[Real, Codes]:
    """The 150 × 4 feature matrix and integer labels 0, 1, 2."""
    bunch = load_iris()
    return (
        np.asarray(bunch.data, dtype=np.float64),
        np.asarray(bunch.target, dtype=np.int64),
    )


def fit_centers(x_train: Real, seed: int = 0) -> Real:
    """Per-feature 1-D k-means centres, shape ``(4, 3)``, each row sorted ascending.

    Fitted on the *training* rows only ``[SNN §IV.B]``; sorting makes the bit order
    (low cluster → high cluster) reproducible across seeds.
    """
    x = np.asarray(x_train, dtype=np.float64)
    centers = np.empty((N_FEATURES, K), dtype=np.float64)
    for f in range(N_FEATURES):
        km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(x[:, f : f + 1])
        centers[f] = np.sort(np.asarray(km.cluster_centers_, dtype=np.float64).ravel())
    return centers


def encode_dataset(x: Real, centers: Real, y: Codes | None = None) -> tuple[Codes, Codes]:
    """``(codes, labels)``: one basis index per sample, MSB = "feature 0, cluster 0".

    Each feature is assigned to its nearest of the three training centres and the resulting
    one-hot blocks are concatenated, giving a 12-bit word with exactly four 1-bits.
    """
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim != 2 or xa.shape[1] != N_FEATURES:
        raise ValueError(f"expected an (n, {N_FEATURES}) feature matrix, got {xa.shape}")
    codes = np.zeros(xa.shape[0], dtype=np.int64)
    for f in range(N_FEATURES):
        cluster = np.abs(xa[:, f, None] - centers[None, f, :]).argmin(axis=1)
        codes |= np.left_shift(1, N_BITS - 1 - (K * f + cluster)).astype(np.int64)
    labels = np.zeros(xa.shape[0], dtype=np.int64) if y is None else np.asarray(y, dtype=np.int64)
    return codes, labels


def code_bits(codes: Codes) -> npt.NDArray[np.uint8]:
    """``(n, 12)`` array of bits, column 0 = MSB = "feature 0, cluster 0"."""
    c = np.asarray(codes, dtype=np.int64)[:, None]
    shifts = np.arange(N_BITS - 1, -1, -1, dtype=np.int64)[None, :]
    return ((c >> shifts) & 1).astype(np.uint8)


def format_code(code: int) -> str:
    """The 12-character string used on the x axis of ``[PTQNN Fig 8]``."""
    return format(int(code), f"0{N_BITS}b")


@dataclass(frozen=True)
class MergedCodes:
    """Distinct codes with their (possibly fractional) merged labels."""

    codes: Codes
    labels: Real
    counts: Codes
    n_collisions: int
    """Distinct codes carrying more than one class label."""


def merge_duplicate_codes(codes: Codes, labels: npt.ArrayLike) -> MergedCodes:
    """Collapse samples sharing a 12-bit code; ``[ASSUMPTION]`` the merged label is the mean.

    A basis state carries one amplitude, so duplicates *must* be merged; the paper is silent on
    how.  Averaging keeps the merged target between the two class levels rather than silently
    picking one.  Already-merged (fractional) labels pass through unchanged.
    """
    c = np.asarray(codes, dtype=np.int64)
    y = np.asarray(labels, dtype=np.float64)
    uniq, inverse, counts = np.unique(c, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=y, minlength=uniq.size)
    means = sums / counts
    spread = np.zeros(uniq.size, dtype=np.float64)
    np.maximum.at(spread, inverse, np.abs(y - means[inverse]))
    n_coll = int(np.count_nonzero(spread > 1e-12))
    return MergedCodes(
        codes=uniq.astype(np.int64),
        labels=means.astype(np.float64),
        counts=counts.astype(np.int64),
        n_collisions=n_coll,
    )


def encoding_accuracy_ceiling(codes: Codes, labels: Codes) -> float:
    """Best accuracy any function of the code can reach — majority label per distinct code."""
    c = np.asarray(codes, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    hits = sum(int(np.bincount(y[c == u], minlength=3).max()) for u in np.unique(c))
    return hits / c.size


@dataclass(frozen=True)
class IrisSplit:
    """A seeded stratified split with its train-only binarisation."""

    train_index: Codes
    test_index: Codes
    train_codes: Codes
    train_labels: Codes
    test_codes: Codes
    test_labels: Codes
    centers: Real
    seed: int
    train_size: float

    @property
    def train_collisions(self) -> int:
        return merge_duplicate_codes(self.train_codes, self.train_labels).n_collisions

    @property
    def test_collisions(self) -> int:
        return merge_duplicate_codes(self.test_codes, self.test_labels).n_collisions

    def summary(self) -> dict[str, float | int]:
        tr = merge_duplicate_codes(self.train_codes, self.train_labels)
        te = merge_duplicate_codes(self.test_codes, self.test_labels)
        return {
            "seed": self.seed,
            "train_size": self.train_size,
            "n_train": int(self.train_codes.size),
            "n_test": int(self.test_codes.size),
            "distinct_train_codes": int(tr.codes.size),
            "distinct_test_codes": int(te.codes.size),
            "shared_codes": int(np.intersect1d(tr.codes, te.codes).size),
            "train_collisions": tr.n_collisions,
            "test_collisions": te.n_collisions,
            "train_ceiling": encoding_accuracy_ceiling(self.train_codes, self.train_labels),
            "test_ceiling": encoding_accuracy_ceiling(self.test_codes, self.test_labels),
        }


def load_split(seed: int = 0, train_size: float = 1.0 / 3.0) -> IrisSplit:
    """Stratified split, k-means fitted on the training rows, both sides encoded ``[SNN §IV.B]``.

    ``[ASSUMPTION]`` ``train_size`` — see the module docstring.
    """
    x, y = load_iris_xy()
    idx = np.arange(x.shape[0], dtype=np.int64)
    tr, te = train_test_split(idx, train_size=train_size, random_state=seed, stratify=y)
    tr, te = np.sort(np.asarray(tr, dtype=np.int64)), np.sort(np.asarray(te, dtype=np.int64))
    centers = fit_centers(x[tr], seed=seed)
    train_codes, train_labels = encode_dataset(x[tr], centers, y[tr])
    test_codes, test_labels = encode_dataset(x[te], centers, y[te])
    return IrisSplit(
        train_index=tr,
        test_index=te,
        train_codes=train_codes,
        train_labels=train_labels,
        test_codes=test_codes,
        test_labels=test_labels,
        centers=centers,
        seed=seed,
        train_size=train_size,
    )
