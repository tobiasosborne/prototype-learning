You are writing an INDEPENDENT reference quantum simulator used only as a cross-check oracle in a test suite.
Work in /home/tobiasosborne/Projects/prototype-learning. Create exactly two files:
  tests/oracle/__init__.py   (empty)
  tests/oracle/indep_sim.py
and run your own sanity checks with `uv run python -c ...` or a throwaway script under /tmp (do not add other files to the repo).
Constraints: Python 3.12, numpy only (no scipy). Deliberately simple and obviously correct: build full 2^n x 2^n
matrices with explicit np.kron products; no clever index tricks, no in-place bit manipulation. Performance is irrelevant (n ≤ 8).

Convention (MUST follow exactly): n qubits, qubit 0 is the MOST significant bit of the computational basis index,
i.e. basis index = Σ_q b_q · 2^(n-1-q); the state vector of |b_0 b_1 ... b_{n-1}⟩ is kron(e_{b_0}, e_{b_1}, ..., e_{b_{n-1}}).

API:
  run(n: int, ops: list[tuple], rho0: np.ndarray | None = None) -> np.ndarray  # returns 2^n x 2^n complex128 density matrix;
     rho0 default |0...0⟩⟨0...0|. ops are applied in order. Supported op tuples:
       ("h", q) ("x", q) ("y", q) ("z", q) ("s", q) ("t", q)
       ("rx", q, theta) ("ry", q, theta) ("rz", q, theta)         # RX(θ)=exp(-iθX/2)=[[cos θ/2, -i sin θ/2],[-i sin θ/2, cos θ/2]], RY, RZ analogous with standard convention RZ(θ)=diag(e^{-iθ/2}, e^{iθ/2})
       ("crx", c, t, theta)   # controlled RX: |0⟩⟨0|_c ⊗ I_t + |1⟩⟨1|_c ⊗ RX(θ)_t ; c and t arbitrary distinct qubits, any order
       ("cx", c, t) ("cz", c, t) ("swap", a, b)
       ("u", [q...], U)       # arbitrary unitary U of shape (2^k, 2^k) acting on the listed qubits, whose own basis ordering is kron over the listed qubits in the listed order (first listed qubit = most significant)
       ("kraus", [q...], [K_1, ..., K_m])   # channel ρ -> Σ K_i ρ K_i† on the listed qubits, same ordering rule as "u"
       ("reset", q)           # ρ -> |0⟩⟨0|_q ⊗ Tr_q ρ
  probabilities(rho) -> np.ndarray of length 2^n (diagonal, real)
  marginal(rho, keep: list[int]) -> reduced density matrix on the kept qubits (partial trace of the others), in the SAME msb-first ordering restricted to `keep` in the given order
  embed(n, qubits, M) -> the 2^n x 2^n operator that acts as M on `qubits` (in listed order) and identity elsewhere  # the workhorse; implement via explicit permutation matrices built from np.kron and basis-index relabelling, and check it on a few small cases
Implement the embedding by: build M ⊗ I_rest on the ordering (listed qubits..., remaining qubits ascending), then conjugate with the permutation matrix P mapping that ordering to the canonical ordering. Build P by iterating over all 2^n basis indices.

Self-checks you MUST run and report the numbers for: (a) ("crx",0,1,θ) vs ("crx",1,0,θ) on 2 qubits give the expected block structure; (b) CRX(θ) via "crx" equals ("u",[c,t],CRX matrix) for random c≠t on n=4; (c) a random Kraus set from a Haar-ish random isometry is trace preserving; (d) embedding of a random 2-qubit unitary on qubits [3,1] of n=4 equals swapping tensor factors by hand; (e) h on all 3 qubits gives uniform probabilities; (f) marginal of |Φ+⟩ on each qubit is I/2.
Finally run `uv run ruff check tests/oracle && uv run mypy tests/oracle/indep_sim.py --strict --ignore-missing-imports` and fix until clean (numpy typing: use np.ndarray annotations loosely via `npt.NDArray[np.complex128]`).
Progress protocol: print one line after each deliverable. Final answer: list the files written and the six self-check numbers.
