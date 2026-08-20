Adversarial review (read-only) of a quantum-noise library. Repo: /home/tobiasosborne/Projects/prototype-learning.
Files: src/ptqnn/noise/channels.py, src/ptqnn/noise/model.py, src/ptqnn/noise/presets.py, src/ptqnn/backends/trajectories.py,
tests/test_noise_channels.py, tests/test_noise_model.py, tests/test_trajectories.py.
Conventions: qubit 0 = most significant bit; Choi C = Σ_ij |i><j| ⊗ E(|i><j|); RX(θ)=exp(-iθX/2); depolarising E(ρ)=(1-p)ρ+p I/d;
average gate fidelity F=(d F_e+1)/(d+1), F_e = Σ|Tr K_i|²/d²; thermal relaxation: populations relax exp(-t/T1) towards n_eq, coherences exp(-t/T2), CP iff T2 ≤ 2T1;
NoiseModel.apply inserts, per gate: idle thermal relaxation on waiting qubits (ASAP schedule), the gate, relaxation for the gate duration on its qubits, then a depolarising channel.
Your job: try to REFUTE correctness. For each suspected defect write a short runnable check with `uv run python -c ...` or a /tmp script (numpy, scipy available; qiskit + qiskit-aer available for independent comparison) and report the output. Specifically attack:
 1. choi()/choi_to_kraus() vectorisation convention consistency (round trip on random channels incl. non-Hermitian Kraus ops; compare E(ρ) from both).
 2. thermal_relaxation for T2 > T1 (e.g. T1=50, T2=90, t=20): compare the resulting channel's action on all 4 basis operators with qiskit_aer.noise.thermal_relaxation_error(t1,t2,t) (units: Aer uses the same time unit for all three) — full process-matrix comparison, not one state.
 3. depolarizing(p, 2): is it exactly (1-p)ρ + p I/4 for random ρ? Is average_gate_fidelity(depolarizing(p,n)) == 1 - p(d-1)/d for n=1,2,3?
 4. NoiseModel.apply scheduling: construct a circuit where ASAP scheduling and "each op is its own time step" differ; compute total_ns and idle channel durations by hand; check the inserted channels' durations by fitting exp(-t/T2) from their Kraus action. Also check that gate-duration relaxation is applied on BOTH qubits of a 2q gate and that the 2q depolarising acts on (control,target) in the right order.
 5. ReadoutError.apply on 3 qubits with per-qubit asymmetric p01/p10: compare against an explicit 8x8 Kronecker confusion matrix.
 6. TrajectoryBackend: find a channel for which the branch-sampling code misbehaves (zero-norm branches, Kraus sets with a K_i that annihilates the state, channels with >2 Kraus ops on 2 qubits, batch_size not dividing n_trajectories); compare to reference.run_density_matrix at 20k trajectories with a 4σ criterion.
 7. presets.py: IBM TOML p01/p10 semantics vs ReadoutError's definition (p01 = P(read 1 | true 0)); quantinuum files with no durations → no relaxation: is that what noise_model() does, and is is_trivial() right for rigetti_ankaa_3 (1q error 0)?
 8. NoiseModel.scaled(): does scaling by 2 double the depolarising p and halve T1/T2 exactly? What happens at factor 0 (should be a noiseless model)?
Report format: numbered list, each item: VERDICT (bug / no bug / design concern), the command you ran, its output, and if a bug, the minimal fix. Do not edit any repo file. Print a one-line progress message after each item.
