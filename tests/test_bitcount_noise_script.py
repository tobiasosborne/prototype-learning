"""Unit tests for the pure helpers of ``experiments/bitcount_noise.py``.

The experiment script itself is a driver (it owns no physics), but its three small estimators —
total-variation distance, the count marginaliser and the finite-shot KL estimator — are new code
and are tested here on toy inputs where the answer is known by hand.
"""

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from ptqnn.noise import presets
from ptqnn.train import losses

_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "bitcount_noise.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bitcount_noise", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bitcount_noise"] = module
    spec.loader.exec_module(module)
    return module


bn = _load()


# --- total variation ----------------------------------------------------------------------------


def test_total_variation_known_values() -> None:
    p = np.array([0.5, 0.5])
    assert bn.total_variation(p, p) == pytest.approx(0.0)
    # Two point masses on disjoint outcomes: TV = 1 (the maximum).
    assert bn.total_variation([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    # ½(|0.5−0.25| + |0.5−0.75|) = 0.25.
    assert bn.total_variation([0.5, 0.5], [0.25, 0.75]) == pytest.approx(0.25)


def test_total_variation_is_the_max_event_probability_gap() -> None:
    # TV(p,q) = max_A |p(A) − q(A)|; check against brute force over all subsets of 4 outcomes.
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4))
    q = rng.dirichlet(np.ones(4))
    best = max(
        abs(p[list(a)].sum() - q[list(a)].sum())
        for k in range(5)
        for a in itertools.combinations(range(4), k)
    )
    assert bn.total_variation(p, q) == pytest.approx(best)


def test_total_variation_symmetric_and_validating() -> None:
    p, q = [0.2, 0.8], [0.7, 0.3]
    assert bn.total_variation(p, q) == pytest.approx(bn.total_variation(q, p))
    with pytest.raises(ValueError):
        bn.total_variation([0.5, 0.5], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        bn.total_variation([0.5, 0.5], [np.nan, 0.5])


def test_total_variation_bounded_by_pinsker() -> None:
    # Pinsker: TV ≤ √(KL/2) with KL in nats.
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = rng.dirichlet(np.ones(8))
        q = rng.dirichlet(np.ones(8))
        assert bn.total_variation(p, q) <= np.sqrt(losses.kl_divergence(p, q) / 2) + 1e-12


# --- count marginalisation ----------------------------------------------------------------------


def test_conditioned_input_counts_selects_out_equals_one() -> None:
    # 3 qubits: input 0, hidden 1, output 2.  Index = 4*q0 + 2*q1 + q2 (qubit 0 most significant).
    counts = np.arange(8, dtype=float)  # counts[i] = i
    got = bn.conditioned_input_counts(counts, input_qubits=(0,), output_qubit=2)
    # x = 0 → indices 001, 011 = 1 + 3 = 4;  x = 1 → 101, 111 = 5 + 7 = 12.
    np.testing.assert_allclose(got, [4.0, 12.0])


def test_conditioned_input_counts_matches_exact_distribution() -> None:
    # On the exact probability vector the helper must reproduce the numerator of
    # distribution.conditioned_input_distribution.
    from ptqnn.backends import reference as ref
    from ptqnn.tasks import bitcount
    from ptqnn.train import distribution

    net = bitcount.network(3, 2, "prototype")
    params = net.random_params(np.random.default_rng(7))
    be = ref.StatevectorBackend()
    p_full = be.probabilities(net.circuit(params, bitcount.superposition_prep(net)))
    num = bn.conditioned_input_counts(p_full, net.input_qubits, net.output_qubit)
    ref_q = distribution.conditioned_input_distribution(
        net, params, bitcount.superposition_prep(net), be
    )
    np.testing.assert_allclose(num / num.sum(), ref_q, atol=1e-12)


def test_conditioned_input_counts_rejects_negative() -> None:
    with pytest.raises(ValueError):
        bn.conditioned_input_counts([1.0, -1.0, 0.0, 0.0], (0,), 1)


# --- finite-shot KL -----------------------------------------------------------------------------


def test_finite_shot_kl_exact_counts_reproduce_exact_kl() -> None:
    target = np.array([0.1, 0.2, 0.3, 0.4])
    counts = np.array([100.0, 200.0, 300.0, 400.0])  # empirical == target
    assert bn.finite_shot_kl(target, counts) == pytest.approx(0.0, abs=1e-15)
    counts2 = np.array([400.0, 300.0, 200.0, 100.0])
    assert bn.finite_shot_kl(target, counts2) == pytest.approx(
        losses.kl_divergence(target, counts2 / counts2.sum())
    )


def test_finite_shot_kl_uses_the_eps_floor_for_unobserved_outcomes() -> None:
    target = np.array([0.5, 0.5])
    counts = np.array([10.0, 0.0])  # outcome 1 never seen
    eps = 1e-12
    expected = 0.5 * np.log(0.5 / 1.0) + 0.5 * np.log(0.5 / eps)
    assert bn.finite_shot_kl(target, counts, eps) == pytest.approx(expected)


def test_finite_shot_kl_all_zero_counts_is_the_large_finite_penalty() -> None:
    # No shot had out = 1: the model has no distribution at all, and the estimator must return
    # the same finite penalty as the exact path (distribution.conditioned_input_distribution
    # returns zeros in this case), not NaN or +inf.
    target = np.array([0.25, 0.25, 0.25, 0.25])
    v = bn.finite_shot_kl(target, np.zeros(4), 1e-12)
    assert np.isfinite(v)
    assert v == pytest.approx(losses.kl_divergence(target, np.zeros(4), 1e-12))
    assert v == pytest.approx(np.log(0.25 / 1e-12))


def test_finite_shot_kl_is_biased_upwards_and_converges() -> None:
    # The plug-in estimator over-estimates KL at finite N and decays like 1/N; this is the
    # statement study D quantifies, so it is pinned here on a toy 4-outcome distribution.
    target = np.array([0.1, 0.2, 0.3, 0.4])
    rng = np.random.default_rng(0)
    means = []
    for shots in (100, 10_000):
        est = [bn.finite_shot_kl(target, rng.multinomial(shots, target)) for _ in range(200)]
        means.append(float(np.mean(est)))
    assert means[0] > means[1] > 0.0
    assert means[1] < means[0] / 10  # ~1/N decay of the (d−1)/(2N) bias


def test_finite_shot_kl_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        bn.finite_shot_kl([0.5, 0.5], [-1.0, 3.0])


# --- module contract the sweep relies on ---------------------------------------------------------


def test_noise_model_scaled_zero_is_the_noiseless_limit() -> None:
    # Study C's s = 0 point is the noiseless model, so the sweep depends on scaled(0) being
    # trivial rather than raising (it used to raise ZeroDivisionError on t1_us / factor).
    base = presets.device("ibm_fez").noise_model()
    assert base.scaled(0.0).is_trivial()
    assert not base.scaled(1.0).is_trivial()
    assert base.scaled(2.0).t1_us == pytest.approx(base.t1_us / 2.0)


# --- compiled 2q counts -------------------------------------------------------------------------


def test_compiled_2q_counts_parses_the_hardware_table(tmp_path: Path) -> None:
    table = tmp_path / "hardware-compile-table.md"
    table.write_text(
        "# Transpiled circuit cost\n\n"
        "## flappy Network(2,4), 7q\n\n"
        "| gate set | topology | depth | n_1q | n_2q | n_swap |\n"
        "|---|---|---:|---:|---:|---:|\n"
        "| generic | linear | 129 | 109 | 120 | 32 |\n"
        "\n## bit-count Network(5,5), 11q\n\n"
        "| gate set | topology | depth | n_1q | n_2q | n_swap |\n"
        "|---|---|---:|---:|---:|---:|\n"
        "| generic | linear | 320 | 284 | 540 | 160 |\n"
        "| generic | heavy-hex | 255 | 275 | 327 | 89 |\n"
        "| generic | all-to-all | 69 | 224 | 60 | 0 |\n"
        "| ibm_heron | linear | 819 | 3285 | 540 | 160 |\n"
    )
    assert bn.compiled_2q_counts(table, 5, 5) == {
        "linear": 540,
        "heavy-hex": 327,
        "all-to-all": 60,
    }
    assert bn.compiled_2q_counts(table, 2, 4) == {"linear": 120}
    assert bn.compiled_2q_counts(table, 9, 9) == {}
    assert bn.compiled_2q_counts(tmp_path / "absent.md", 5, 5) == {}
