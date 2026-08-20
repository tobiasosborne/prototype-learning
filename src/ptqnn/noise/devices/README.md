# Vendor noise/calibration presets

One TOML file per device, readable with `tomllib` (stdlib, Python 3.11+). Schema is a
simplified version of `../../../../../QuantumHardware.jl/schema/SCHEMA.md`: a `[device]`
table, a `[noise]` table, and an `[[provenance]]` array with one entry per sourced field
carrying `source_url`, `source_kind`, `retrieved_at` and a **verbatim quote** of the
sentence or table row the number was read from.

All files retrieved **2026-08-20**.

## Ground rules used here

- **No number without a quote.** If a field could not be sourced it is *absent*, and
  `noise.notes` says so explicitly and says where it would have to come from.
- **Fidelity → error conversions are stated.** Where a vendor publishes only a fidelity,
  the file records `error = 1 − fidelity` and the provenance `field` line says so.
- **Ranges are not values.** Where only a range or a bound is published (IonQ T1
  "10–100 s", IQM CZ "20 ns – 40 ns", IQM PRX "≤ 40 ns"), the range is in the `quote`, the
  midpoint is the value *only when a midpoint is meaningful*, and `notes` flags it. Pure
  upper bounds (IQM 1q duration) are recorded as bounds in `notes` and left out of `[noise]`.
- **`*_median` sometimes holds a mean.** Quantinuum publishes zone-averaged means and
  Google publishes means ± 1 s.d.; both files say so in the first lines of `notes`. Only
  IBM, IQM and Rigetti numbers are genuine medians.
- **One epoch per file.** Mixing a July-2024 T1 with a Feb-2025 gate time produces a device
  that never existed. Where cross-epoch numbers exist they are in `notes`, not in `[noise]`.

## Devices × fields

Legend for source: **V** = vendor page/docs, **C** = vendor calibration dump, **S** = vendor
spec sheet PDF, **P** = paper, **N** = press release / news, **—** = not published.

| Device | n / topology | T1 | T2 | 1q err | 1q ns | 2q err | 2q ns | Readout err | Readout ns |
|---|---|---|---|---|---|---|---|---|---|
| `ibm_fez` (Heron r2) | 156, heavy-hex | 144.9 µs **C** | 88.0 µs echo **C** | 2.29e-4 **C** | 24 **C** | 3.90e-3 **C** | 84 **C** | 7.57e-3 **C** | 1560 **C** |
| `quantinuum_h2_1` | 56, all-to-all | — | — | 1.9e-5 **C** | — | 1.1e-3 **C** | — | 1.0e-3 **C**/**S** | — |
| `quantinuum_h1_1` | 20, all-to-all | — | — | 1.8e-5 **C** | — | 9.7e-4 **C** | — | 2.3e-3 **C** | — |
| `ionq_forte` | 36, all-to-all | ~55 s **V** | ~1 s **V** | 2.0e-4 **V**+**P** | 1.3e5 **P** | 4.0e-3 **V** | 9.5e5 **P** | 5.0e-3 **V**+**P** | — |
| `iqm_garnet` | 20, square | 27.4 µs **P** | 21.1 µs echo **P** | 8.0e-4 **V** | — | 4.9e-3 **V** | 30 **V**+**P** | 3.0e-2 **P** | — |
| `rigetti_ankaa_3` | 84, square | — | — | — | — | 1.0e-2 **N** | 72 **N** | — | — |
| `google_willow` (105q) | 105, square | 68 µs **P** | 89 µs CPMG **P** | 5.3e-4 **P** | 25 **P** | 4.1e-3 **P** | 42 **P** | 8.0e-3 **P** | — |

### Source per device

| Device | Sources |
|---|---|
| `ibm_fez` | IBM calibration snapshot `props_fez.json` / `conf_fez.json` shipped in `qiskit-ibm-runtime` `fake_provider` (`last_update_date` **2025-02-26**) — the per-qubit/per-edge table is IBM's, the **medians were computed by us** with `statistics.median` (n=156 qubits, n=352 directed CZ edges) and re-verified locally. Definitions (Hahn echo, RB conventions, EPLG, readout error) from IBM's `qpu-information` and `processor-types` guides. Cross-checks in `notes`: IBM's live public dashboard `quantum.cloud.ibm.com/computers` (median 2q 2.61e-3, EPLG₁₀₀ 4.54e-3) and arXiv:2410.00916 Table IV (published medians as of 2024-07-04). Per-qubit T1/T2 arrays (156 entries each) are included. |
| `quantinuum_h2_1`, `quantinuum_h1_1` | Quantinuum live **Performance Validation** page (latest RB sweep, per machine) for gate/SPAM/memory errors; H1 Product Data Sheet **v8.0.1 (2025-06-09)** and H2 Product Data Sheet **v4.00 (2025-09-17)** for qubit count, connectivity, native gates and the Typical/Max envelope. `p01`/`p10` come straight from the vendor's directional `SPAM Error (0)` / `SPAM Error (1)`. Durations: arXiv:2305.03828 (H2) and arXiv:2003.01293 (H1) — neither publishes a gate time. Local corroboration: `../../../../../QuantumHardware.jl/sources/2026/04/14/quantinuum-spec-sheet-parameters.csv` (spec-sheet CSV, last row 2024-12-06) and the raw RB JSON dumps under `.../sources/2026/04/14/quantinuum/`. |
| `ionq_forte` | IonQ's public Forte spec panel (`ionq.com/quantum-systems/forte`) for 1q/2q error, SPAM, T1/T2 and qubit count; `api.ionq.co/v0.3/backends` for the native gate set. Durations and the independent DRB medians from arXiv:2308.05071 (IonQ, 30-qubit Forte) and arXiv:2506.07866 (IonQ-coauthored, 2025). |
| `iqm_garnet` | AWS Braket IQM partner page + Braket developer guide (vendor) for qubit count, topology, native gates and median 1q/2q fidelities; arXiv:2408.12433 (IQM's own Garnet paper) for median 1q/CZ errors as errors, CZ duration range and median simultaneous readout error. **T1/T2 are third-party**: arXiv:2512.23009 Table 3 (one calibration snapshot; T1 and T2-echo from the same table), with arXiv:2603.28486 Table I as a second median set. IQM Radiance 20-qubit spec sheet used only for bounds, in `notes`. |
| `rigetti_ankaa_3` | Rigetti press release (2024-12-23) for the headline 99.5 % figure; the full PR text (reprinted verbatim by The Quantum Insider) for the iSWAP/fSim split and gate times; AWS Braket developer guide for qubit count, native gates and the RX angle restriction. |
| `google_willow` | Nature 638, 920–926 (2025) + its Supplementary Information (Tables S4/S5, §I.B) for every number; Google's Willow Spec Sheet (2024-12-09) as corroboration. Note arXiv:2408.13687v1 carries the main text **only** — the device table exists solely in the Nature SI. |

## What is missing

**Per-qubit data.** Only `ibm_fez` carries `per_qubit` arrays, because IBM is the only
vendor that publishes a full per-qubit table for free. Everywhere else you get a single
median (or mean) and the spread is thrown away — and the spread is not small: Willow's 1q
error s.d. is ~80 % of its mean, IonQ's 2q error varies ~3× across pairs with seven pairs
above 3e-2, and IBM's own T1 spans 27–274 µs on one chip. A homogeneous noise model built
from a single median is a *reasonable first cut and a poor last word*; sweep, don't trust.

**T1/T2 for trapped ions and for Rigetti.** Quantinuum publishes no T1/T2 at all — it
reports a "Memory Error per Depth-1 Circuit Time" instead, which is an idling+transport
dephasing probability with no published depth-1 time in seconds, so it *cannot* be
converted to a T2. Rigetti publishes neither. IonQ publishes only order-of-magnitude
ranges. IQM publishes none, so Garnet's come from third-party papers.

**Gate durations.** Absent for both Quantinuum machines (no vendor or paper source exists)
and for IQM's 1q gate (only a bound). Readout duration is published only by IBM.

**Crosstalk.** No device here has a usable `idle.zz_coupling_khz`. IBM claims Heron's
tunable couplers "virtually eliminate cross-talk" but publishes no ZZ rate; Google reports a
CZ *crosstalk error* (~3.3e-4, already folded into its CZ total) rather than a coupling
rate; Quantinuum reports measurement crosstalk only. ZZ numbers found for other
tunable-coupler devices must not be attributed to these chips.

**Readout asymmetry.** `p01`/`p10` are available only for IBM and Quantinuum. Both show
strong asymmetry (IBM ~3×, Quantinuum H1-1 ~3×) in the same direction — |1⟩ decaying during
readout — so a symmetric bit-flip readout model is wrong for those two at least.

**Coherent errors, leakage, non-Markovianity.** Nothing here captures them. Leakage rates
*are* published by Quantinuum (per 1q/2q gate) and Google (CZ leakage, heating) and are
recorded in `notes` but not modelled.

## How to refresh

- **IBM** — with an account: `qiskit_ibm_runtime.QiskitRuntimeService().backend("ibm_fez")`,
  then `backend.properties()` (per-qubit `T1`/`T2`/`readout_error`/`prob_meas0_prep1`/
  `prob_meas1_prep0`/`readout_length`, per-gate `gate_error`/`gate_length`) and
  `backend.configuration()` for `coupling_map` / `basis_gates`. Take
  `statistics.median` over the qubit and directed-edge rows exactly as here, and copy
  `properties().last_update_date` into the file header — the epoch is part of the data.
  Without an account, the `fake_provider` JSONs used here are the same schema and are
  refreshed upstream; the public `quantum.cloud.ibm.com/computers` table gives current
  median 2q / EPLG / median readout but no T1/T2.
- **Quantinuum** — re-read the live Performance Validation page (it tracks the latest RB
  sweep), and re-download the raw RB JSON dumps from
  `github.com/CQCL/quantinuum-hardware-specifications` (`data/<machine>/<date>/{SQ_RB,TQ_RB,
  SPAM,Memory_RB,Measurement_crosstalk}.json` — survival counts vs sequence length, fit them
  yourself) plus `notebooks/Spec sheet parameters.csv`. Re-download both data-sheet PDFs
  from `docs.quantinuum.com/systems/_static/assets/data_sheets/` (note: the
  `/systems/data_sheets/` path 404s).
- **IonQ** — `api.ionq.co/v0.3/backends` is public and gives qubit count and native gates;
  the per-backend `characterization_url` needs an API key (returns 403 "Insufficient scope"
  otherwise). With a key it is the only route to per-qubit/per-pair numbers.
- **IQM / Rigetti** — AWS Braket `GetDevice` (`boto3` `braket` client, or
  `AwsDevice(arn).properties`) returns the current `standardized` calibration block with
  per-qubit T1/T2 and per-pair fidelities. Credentials required; this is the only way to
  fill in Ankaa-3's blanks.
- **Google Willow** — not accessible; refresh only when a new paper or spec sheet appears.

After any refresh, update `retrieved_at` in **both** `[device]` and every touched
`[[provenance]]` entry, and re-quote — a stale quote is worse than no quote.
