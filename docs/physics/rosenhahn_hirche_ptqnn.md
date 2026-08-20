# Ground truth: Rosenhahn & Hirche, *Prototype Learning for Quantum Neural Networks*

Pin tag: **`PTQNN`**. Source: `literature/pdf/QPrototypeLearning-5.pdf` (11 pages, dated
"August 20, 2026"), text dump `literature/QPrototypeLearning-5.txt`, extracted figure rasters
`literature/figs/*.png`. Authors: Bodo Rosenhahn¹, Christoph Hirche¹ (¹ Institute for
Information Processing (tnt/L3S), Leibniz Universität Hannover).

Reference `[26]` of this paper is the prerequisite **SNN** paper (Rosenhahn, Osborne, Hirche,
*Stochastic neural networks for quantum devices*, 2026 = arXiv:2602.22241); see
`docs/physics/rosenhahn_osborne_hirche_snn.md`.

Conventions used in this file:

* Equation/figure/section numbers are the paper's own. Page numbers refer to the printed page
  numbers in the PDF (page 1 is the title page).
* **Verbatim** paper text is in quotation marks. Anything else is our paraphrase.
* Numbers taken off a raster figure are marked "read from figure, ±…". The reading error is the
  honest pixel-level uncertainty of the extraction, not a guess at the paper's precision.
* Circuit figures number qubits 1-based top-to-bottom, as drawn. Code is 0-based (see
  `CLAUDE.md` §Gotchas).

---

## 0. Abstract and claimed contributions

Abstract (p. 1, verbatim, load-bearing sentences):

> "This work proposes a quantum prototype perceptron architecture that fundamentally enhances
> the representational capacity and generalization ability of quantum neural networks."

> "In addition, we present a quantum optimization procedure that exploits quantum superposition
> to encode the entire training dataset as a quantum state, allowing for global,
> distribution-level adjustment of network parameters. The model parameters are optimized by
> minimizing the Kullback-Leibler divergence between the target and model output distributions,
> leading to significantly accelerated convergence and reduced classical memory requirements."

Contributions list (§I, p. 2, verbatim):

> "1. The formulation of a novel quantum prototype perceptron model that enables both 0- and
> 1-valued input features to actively participate in the activation computation, mirroring an
> XOR-type comparison with learnable reference vectors."

> "2. The development of a quantum optimization procedure based on superposition encoding,
> allowing training over the entire data distribution simultaneously and achieving significant
> efficiency gains in comparison to conventional, sample-wise optimization."

> "3. Empirical evidence showing that prototype quantum models exhibit increased stability and
> are less susceptible to overfitting than standard perceptron-based quantum networks."

---

## 1. §II Fundamentals

### 1.1 §II.A Neural networks — Eq (1), classical perceptron

> "In 1958, Frank Rosenblatt published his perceptron model which is based on a scalar product,
> followed by a step function, see Rosenblatt [25]. It can be summarized as"

```latex
\sigma_j =
\begin{cases}
1 & : \sum_i \omega_{ij} x_i + b > 0 \\
0 & : \text{else}
\end{cases}
\tag{1}
```

> "The bias value $b$ corresponds to the decision threshold and $\omega_{ij}$ are learnable
> parameters."

### 1.2 §II.A — Eq (2), attention

> "Let $D$ be a database with corresponding key and values,
> $D = \{(k_1, v_1), \dots, (k_m, v_m)\}$. Let $q$ be a query. Then attention over $D$ can be
> computed as as"

```latex
f(q, D) = \sum_{i=1}^{m} \alpha(q, k_i) v_i
\tag{2}
```

> "where $\alpha(q, k_i)$ are scalar attention weights."

Eq (2) is used only as a conceptual bridge to prototype learning (§III.A); it does not enter
any circuit.

### 1.3 §II.B — the classic (baseline) quantum perceptron, **Fig 1**

Text (p. 3, verbatim):

> "Figure 1 summarizes the core principle: Two input qubits are connected to an output qubit by
> using controlled rotation (CRX) gates. A plain rotation realizes the bias. This concept leads
> to a probabilistic activation of the third qubit, dependent on the input activations."

Caption Fig 1 (verbatim): *"Realization of a perceptron as proposed in [26]. The input qubits
$|q_1\rangle$ and $|q_2\rangle$ modify the probability of activating $|q_3\rangle$."*

**Fig 1 as an explicit gate list.** Vector figure (quantikz), read exactly; 3 qubits.

| wire | role |
|---|---|
| `|q1⟩` (top) | input 1 |
| `|q2⟩` (middle) | input 2 |
| `|q3⟩` (bottom) | output / neuron |

Gate sequence, left to right (all rotations about **x**):

1. `RX(θ)` on `q3` — the bias. No control.
2. `CRX(φ1)`: control `q1`, target `q3`. (Drawn as a box `Rx(φ1)` on the `q3` wire with a
   filled control dot on `q1`.)
3. `CRX(φ2)`: control `q2`, target `q3`.

Nothing at all is applied to `q1`, `q2` — the classic neuron does not touch its inputs.
Parameter count for `n` inputs: `n` CRX angles + 1 bias = `n + 1`.

### 1.4 §II.C Optimization — Kiefer–Wolfowitz (unnumbered) and Eq (3) simulated annealing

> "the parameters of the quantum prototype neural network are optimized using stochastic
> approximation methods. The Kiefer-Wolfowitz algorithm [12], which is closely connected to the
> Robbins-Monro scheme [24]. The method estimates the gradient of an objective function from
> noisy function evaluations. For a sufficiently smooth objective function $M(x)$, the iterative
> update is given by"

```latex
x_{n+1} = x_n + a_n \left( \frac{M(x_n + c_n) - M(x_n - c_n)}{2 c_n} \right)
```

(*unnumbered* in PTQNN; identical to the unnumbered display in SNN §III.A.)

> "where $M(x_n + c_n)$ and $M(x_n - c_n)$ are independently computed. Thus, the gradient of
> $M(x)$ is approximated by a central difference method. A damping factor $a_n$ as step size
> controls the shifts the solution towards an optimum."

Simulated annealing acceptance probability:

```latex
P\big(e(x_t), e(\hat{x}_t), T\big) \propto
\exp\left( - \frac{e(\hat{x}_t) - e(x_t)}{kT} \right)
\tag{3}
```

> "with $k$ a damping factor $k > 0$. At iteration $t$, a candidate parameter vector
> $\hat{x}_t$ is sampled in the neighborhood of the current vector $x_t$, for example using a
> stochastic perturbation. The candidate is accepted according to an acceptance probability that
> depends on the objective function $e(x)$, which corresponds to the optimization loss in this
> work. Candidate solutions that reduce the loss are therefore always accepted. Solutions with an
> increased loss may also be accepted, with a probability that decreases with both the loss
> increase and the temperature. As $T_t$ decreases according to an annealing schedule, the
> optimization gradually changes from exploratory search to local refinement."

And, decisively for us:

> "As shown in XY, common gradient-based optimization is also possible, but due to the risk of
> frequently getting stuck in local minima, **our implementation is solely based on stochastic
> optimization.**" (emphasis ours)

("XY" is an unresolved cross-reference placeholder; it occurs three times in the paper —
see §9.)

---

## 2. §III.B The prototype perceptron — **Fig 2** (the central object)

Text (p. 5, verbatim):

> "Figure shows 2 the core idea for the realization of a prototype sensitive perceptron:
> Rotations before the CR-rotations modify the likelihood for increasing/decreasing an activation
> on $|q_3\rangle$. This allows to capture also 0-entries as valuable information."

> "Please note, that this model itself does require an increase of optimization parameters, as
> the former bias is shifted to the amount of activation when a rotation is triggered and the
> parameters of the controlled rotations are shifted to the rotations before and after the
> CR-gate."

(Sic — "does require an increase" contradicts §III.C's "with the same amount of parameters";
see §9.)

Caption Fig 2 (verbatim): *"Our realization of a prototype sensitive perceptron. rotations
before the CR-rotations modify the likelihood for increasing an activation on $|q_3\rangle$.
This allows to capture also 0-entries as valuable information. Figure 3 shows a tiny example
which leads to the highest activation for a [1 0] input prototype."*

### 2.1 Fig 2 as an explicit gate list

Vector figure (quantikz) on p. 4, read exactly. **3 qubits**, same roles as Fig 1
(`q1`, `q2` = inputs; `q3` = output/neuron). Gate columns, left to right exactly as drawn:

| # | gate | qubits | angle as printed |
|---|------|--------|------------------|
| 1 | `RX(φ1)`  | `q1` | `ϕ₁` |
| 2 | `CRX(θ)`  | control `q1`, target `q3` | `θ` |
| 3 | `RX(-φ1)` | `q1` | `−ϕ₁` |
| 4 | `RX(φ2)`  | `q2` | `ϕ₂` |
| 5 | `CRX(θ)`  | control `q2`, target `q3` | `θ` |
| 6 | `RX(-φ2)` | `q2` | `−ϕ₂` |

Two facts that are easy to get wrong and are **explicit in the drawing**:

1. **There is no plain bias rotation on `q3`.** Fig 1 has `RX(θ)` on `q3` with no control;
   Fig 2 has none. The bias role is taken over by the shared CRX angle
   ("the former bias is shifted to the amount of activation when a rotation is triggered").
2. **Both CRX boxes carry the *same* symbol `θ`.** The figure shows one activation angle per
   neuron, shared across all of that neuron's inputs, plus one prototype angle `φ_i` per input.
   Parameter count for `n` inputs: `n` prototype angles + 1 shared `θ` = `n + 1`, i.e. exactly
   the same as the classic neuron — which is what makes the §III.C claim "(with the same amount
   of parameters)" true. **[INFERENCE from the figure + the parameter-count claim; the paper
   never writes the neuron's unitary down.]** See §9 for why this is not 100 % airtight.
3. The input rotations are **un-computed** (`RX(-φ_i)` after the CRX), so the prototype neuron
   leaves its input register invariant and neurons of one layer commute, exactly as in the
   classic case.

Reading of `q1`/`q2`/`q3` as *drawn*: `q1`'s three boxes occupy circuit columns 1–3, `q2`'s
three boxes columns 4–6 (they are staggered in the drawing, not parallel). Physically all six
gates on disjoint-or-commuting supports commute (all CRX are x-rotations of the same target),
so the drawn ordering carries no extra information.

### 2.2 Fig 3 — activation-probability surface for prototype `|01⟩`

Caption (verbatim): *"activation probability of the prototype $|01\rangle$: If $|q_0\rangle$ is
in 0 and $|q_1\rangle$ is in 1, the probability of activation is maximized. If only one
condition is met, the activation is reduced and if the pattern does not match, the activation
is zero"*

Body text (p. 5, verbatim, note it uses 1-based names and a different prototype than the
caption):

> "Figure 3 shows a tiny example which leads to the highest activation for a [1 0] input
> prototype. Thus, only if qubit $|q_1\rangle = 1$ and $|q_2\rangle = 0$ the qubit $|q_3\rangle$
> is set to 1. But if $|q_1\rangle = 1$ and $|q_2\rangle = 1$ it then leads to a decrease of the
> probability to activate the qubit $|q_3\rangle$."

Figure content (raster, MATLAB 3-D `surf`, page 4 right column):

* **x axis** `q₀`, ticks `0, 0.1, …, 0.9, 1`.
* **y axis** `q₁`, ticks `0, 0.2, 0.4, 0.6, 0.8, 1`.
* **z axis** labelled `activation probability`, ticks `0.2, 0.3, …, 1`.
* Colour bar labelled `0.3 … 1` (jet colormap), i.e. the plotted surface's own range.
* The surface maximum (deep red, `= 1.0`) sits at `q₀ ≈ 0, q₁ ≈ 1`; the minimum (deep blue)
  at `q₀ ≈ 1, q₁ ≈ 0`.
* **The plotted minimum is ≈ 0.2 (read from figure, ±0.05), not 0.** The caption's "the
  activation is zero" is not what the surface shows. See §9.

---

## 3. §III.C Learning under superposition — Eq (4), Eq (5), Figs 4–6

### 3.1 Eq (4) — the training loss

> "Given two probability distributions $P$ and $Q$, the Kullback-Leibler-divergence
> (KL-divergence) or relative entropy is a dissimilarity measure for two distributions,"

```latex
D_{\mathrm{KL}}(P \,\|\, Q) = \sum_{x \in X} p(x) \log \frac{p(x)}{q(x)}
\tag{4}
```

> "In our experiments we use the standard KL-divergence as optimization criteria to perform a
> global optimization of the network weights."

`P` is the **target** and `Q` the **model** distribution (from the abstract: "minimizing the
Kullback-Leibler divergence between the target and model output distributions", target named
first, and `p(x)` in the numerator of Eq 4). Cf. `CLAUDE.md`: `KL(target ‖ model)`.

### 3.2 Eq (5) — the alternative that is *not* used

> "The cosine-dissimilarity of two unit vectors can be expressed as its simple scalar product
> (denoted as $\cdot$),"

```latex
D_{\cos}(P, Q) = 1 - P \cdot Q
\tag{5}
```

Mentioned as a quantum-friendly alternative ("this measure is also directly expressible as
quantum gate sequence [22]") but explicitly not used.

### 3.3 The bit-count task

> "Figure 4 summarizes the basic pipeline on a small example: The task is bit-counting of a
> 5-dimensional bit string. The 5 bits are encoded as input (denoted as shaded area). Prototype
> layers as defined in Figure 2 are connected to 6 hidden neurons. The outcome is then forwarded
> to one output neuron. The output neuron gives a probability of measurement. It yields an
> amplitude encoding of the desired bit count. Strings with many 1-bits result in a higher
> activation probability and bit strings with many 0 entries result in a lower probability."

And on the superposition trick:

> "Figure 5 shows the core idea for efficient optimization: Hadamard gates are used to encode
> all possible input states in superposition. The measured outcome distribution can be compared
> to the target distribution using e.g. a KL-divergence score"

> "The last experiment (bit counting) made it very easy to use Hadamard gates to represent all
> $2^5$ possible states."

### 3.4 **Fig 4** — bit-count architecture, **6 hidden neurons, 12 qubits**

`literature/figs/fig04-bitcount-arch.png` (3217×1300 px). Left panel = the circuit (a MATLAB
`quantumCircuit` plot); right panel = a 32-bin histogram (identical to Fig 5 bottom-right, see
§3.6).

Circuit, exactly as drawn:

* **12 wires**, numbered 1…12 top to bottom.
* Wires **1–5**: the 5 input bits. They carry a light-blue highlight box (the "shaded area" of
  the caption) and **no gates at all** in this figure.
* Wires **6–11**: the **6 hidden neurons**.
* Wire **12**: the single output neuron.
* Six composite gate blocks, drawn as tall boxes labelled `Pro…` (MATLAB truncates the
  composite-gate name, presumably `Prototype`). Block `k` (k = 1…6) has **6 ports**, numbered
  1…6 inside the box: ports 1–5 attach to wires 1–5, port 6 attaches to wire `5 + k`.
  So each block is one prototype neuron over all 5 inputs, writing into its own hidden qubit.
* Wire 12 carries the **classic** (Fig-1 style) neuron: a leading un-controlled `RX`, then six
  controlled `RX` gates, controlled by wires 6, 7, 8, 9, 10, 11 in that order.

Numeric angles printed inside the wire-12 boxes (read from figure; glyph height ≈ 8 px, so the
**last printed digit is uncertain by ±1**, i.e. the values are good to ~±0.00002 in the 5-digit
cases and ±0.0001 in the 4-digit cases):

| gate | control | angle as printed |
|---|---|---|
| `RX` (bias) | — | `-0.38929` |
| `RX` | wire 6  | `-0.92251` |
| `RX` | wire 7  | `-0.88098` |
| `RX` | wire 8  | `-1.4065`  |
| `RX` | wire 9  | `-0.58525` |
| `RX` | wire 10 | `-1.5058`  |
| `RX` | wire 11 | `-0.81206` |

The angles *inside* the six `Pro…` blocks are not shown (the blocks are collapsed).

Caption Fig 4 (verbatim): *"Quantum Prototype Learning architecture: The task is bit-counting of
a 5-dimensional bit string. The 5 bits are encoded as input (denoted as shaded area). Prototype
layers connect to 6 hidden neurons which are then forwarded to one output neuron. The output
neuron gives are probability of measurement, which is an amplitude encoding of the output.
Strings with many 1-bits result in a higher activation probability and bitstrings with many 0
entries result in a lower probability."*

### 3.5 **Fig 5** — superposition training, **5 hidden neurons, 11 qubits**

`literature/figs/fig05-superposition-training.png` (3053×1648 px). Three panels: circuit
(left), model histogram (top right), target histogram (bottom right).

Circuit, exactly as drawn:

* **11 wires**, numbered 1…11.
* Wires **1–5**: inputs, each with a **Hadamard `H`** gate at the very start (inside the
  light-blue highlight box). This is the "all $2^5$ states in superposition" preparation.
* Wires **6–10**: **5 hidden neurons** — only five `Pro…` blocks, each again with 6 ports
  (1–5 → wires 1–5, port 6 → wire `5 + k`, k = 1…5).
* Wire **11**: output neuron — a leading un-controlled `RX`, then **five** controlled `RX`
  gates, controlled by wires 6, 7, 8, 9, 10 in order.

⚠ **Fig 4 has 6 hidden neurons and Fig 5 has 5**, for what the text presents as the same
bit-count experiment. Neither the text nor the captions mention the change. See §9.

The six numeric angle labels on wire 11 are rendered at ≈ 6 px glyph height and are **not
legible** even at native resolution; only the leading sign and (for two of them) the first digit
can be guessed. **Do not use them.** (The Fig 4 labels above *are* legible and should be
preferred.)

Caption Fig 5 (verbatim): *"Efficient optimization: Hadamard gates are used to encode all
possible states in superposition. The measured outcome distrbitution can compared to the target
distribution using e.g. a KL-divergence score. This allows for a global optimization of the
network weights."*

### 3.6 **Fig 5 bottom-right = the target histogram** (and it equals Fig 4's right panel)

Extraction method: MATLAB bar fill colour `RGB(17,113,190)`; plot box located from its frame
(y-axis `0 … 0.06` spanning pixel rows 1547.5 → 1008.5), bar tops corrected by the 1.5 px bar
border. 32 bars, x labels `00000, 00001, …, 11111` in ascending binary order, MSB = wire 1.

The six distinct bar heights are **exactly equally spaced in pixels** (tops at rows
1469, 1389, 1308.5, 1229, 1149, 1068 → spacings 80, 80.5, 79.5, 80, 81 px), which alone proves
the distribution is `(k + const)` for `k = popcount`.

**Hypothesis `p(x) = (popcount(x) + 1) / 112` — CONFIRMED.**
`Σ_x (popcount(x)+1) = 80 + 32 = 112`, so the hypothesis is normalised. Read values vs.
hypothesis:

| popcount `k` | `(k+1)/112` | read from figure | residual |
|---|---|---|---|
| 0 | 0.008929 | 0.00891 | −0.00002 |
| 1 | 0.017857 | 0.01781 | −0.00005 |
| 2 | 0.026786 | 0.02683 / 0.02672 | ±0.0001 |
| 3 | 0.035714 | 0.03562 | −0.00009 |
| 4 | 0.044643 | 0.04453 | −0.00011 |
| 5 | 0.053571 | 0.05354 | −0.00003 |

Reading error **±0.0002** per bar (dominated by the 1–2 px bar-border ambiguity). The read
values sum to 0.9982 (should be 1), consistent with a small systematic under-read.
Every one of the 32 bars sits at the level predicted by its popcount; the full table:

```
00000 k=0   00001 k=1   00010 k=1   00011 k=2   00100 k=1   00101 k=2   00110 k=2   00111 k=3
01000 k=1   01001 k=2   01010 k=2   01011 k=3   01100 k=2   01101 k=3   01110 k=3   01111 k=4
10000 k=1   10001 k=2   10010 k=2   10011 k=3   10100 k=2   10101 k=3   10110 k=3   10111 k=4
11000 k=2   11001 k=3   11010 k=3   11011 k=4   11100 k=3   11101 k=4   11110 k=4   11111 k=5
```

**The right-hand panel of Fig 4 is pixel-for-pixel the same distribution** (bar tops at rows
732, 629, 524, 421, 317, 213 — spacings 104, 104, 103, 105, 103 px; six equally spaced levels,
same 32-bar pattern). So Fig 4's histogram is the *target*, not a trained-model output.

### 3.7 **Fig 5 top-right = the trained-model histogram**

MATLAB bar fill `RGB(112,170,216)`; y-axis label `Probability`, ticks `0, 0.005, …, 0.05`
(tick-label centres verified at 66.25 px per 0.005; `0` at row 811.5, `0.05` at row 149).
Values below use `p = (813 − bar_top_px) / 13250`; the 32 values **sum to 0.9995**, which
validates the calibration. Reading error **±0.0003** each.

| bitstring | k | model (read) | target `(k+1)/112` | model − target |
|---|---|---|---|---|
| 00000 | 0 | 0.01608 | 0.00893 | +0.0072 |
| 00001 | 1 | 0.01909 | 0.01786 | +0.0012 |
| 00010 | 1 | 0.01879 | 0.01786 | +0.0009 |
| 00011 | 2 | 0.02619 | 0.02679 | −0.0006 |
| 00100 | 1 | 0.01932 | 0.01786 | +0.0015 |
| 00101 | 2 | 0.02611 | 0.02679 | −0.0007 |
| 00110 | 2 | 0.02672 | 0.02679 | −0.0001 |
| 00111 | 3 | 0.03547 | 0.03571 | −0.0002 |
| 01000 | 1 | 0.01879 | 0.01786 | +0.0009 |
| 01001 | 2 | 0.02642 | 0.02679 | −0.0004 |
| 01010 | 2 | 0.02642 | 0.02679 | −0.0004 |
| 01011 | 3 | 0.03585 | 0.03571 | +0.0001 |
| 01100 | 2 | 0.02649 | 0.02679 | −0.0003 |
| 01101 | 3 | 0.03547 | 0.03571 | −0.0002 |
| 01110 | 3 | 0.03630 | 0.03571 | +0.0006 |
| 01111 | 4 | 0.04445 | 0.04464 | −0.0002 |
| 10000 | 1 | 0.01992 | 0.01786 | +0.0021 |
| 10001 | 2 | 0.02717 | 0.02679 | +0.0004 |
| 10010 | 2 | 0.02664 | 0.02679 | −0.0001 |
| 10011 | 3 | 0.03547 | 0.03571 | −0.0002 |
| 10100 | 2 | 0.02687 | 0.02679 | +0.0001 |
| 10101 | 3 | 0.03525 | 0.03571 | −0.0005 |
| 10110 | 3 | 0.03547 | 0.03571 | −0.0002 |
| 10111 | 4 | 0.04279 | 0.04464 | −0.0018 |
| 11000 | 2 | 0.02664 | 0.02679 | −0.0001 |
| 11001 | 3 | 0.03592 | 0.03571 | +0.0002 |
| 11010 | 3 | 0.03547 | 0.03571 | −0.0002 |
| 11011 | 4 | 0.04370 | 0.04464 | −0.0009 |
| 11100 | 3 | 0.03532 | 0.03571 | −0.0004 |
| 11101 | 4 | 0.04302 | 0.04464 | −0.0016 |
| 11110 | 4 | 0.04340 | 0.04464 | −0.0012 |
| 11111 | 5 | 0.04845 | 0.05357 | −0.0051 |

Residual KL, computed from these read-off numbers,
`D_KL(target ‖ model) ≈ 2.6 × 10⁻³ nat` (model renormalised to sum 1) — the same order as the converged blue curve in Fig 6.
The largest errors are at the extremes (`00000` too high by +0.0072, `11111` too low by
−0.0051), i.e. the trained model compresses the dynamic range.

⚠ Which of the two right-hand panels is target and which is model is **not stated** in the
caption. The assignment above is forced by the fact that the bottom-right panel is an *exactly*
linear 6-level staircase (analytic) while the top-right one is not, and the top-right panel is
the one labelled `Probability`. **[INFERENCE, high confidence.]**

### 3.8 **Fig 6** — convergence, iterative vs. superposition

`literature/figs/fig06-convergence.png` (1234×441 px). Single axes.

* **x axis** `# Iterations`, ticks `0, 20, 40, …, 200`; `0` at px 83.0, `200` at px 1213.5
  (5.6525 px per iteration).
* **y axis** `KL divergence`, ticks `0, 0.01, …, 0.06`; `0` at row 371.5, `0.06` at row 8.5
  (6050 px per unit).
* Legend: **red** = `Iterative Optimization`, **blue** = `Optimization using superposition`.

Read values (curve mid-line; reading error **±0.0005**, set by the ≈ 6 px line width):

**Blue (superposition):**

| iteration | KL |
|---|---|
| ~1.4 (first plotted point) | 0.0535 (band 0.0525–0.0543) |
| 2 | 0.0505 |
| 3 | 0.0417 |
| 4 | 0.0264 |
| 5 | 0.0153 |
| 6 | 0.0046 |
| 7 | 0.0022 |
| 8 | 0.0013 |
| 9 | 0.00083 |
| 10 | 0.00058 |
| ≥ 12 | 0.0005 and flat to iteration 200 |

**Red (iterative, sample-wise, the scheme of [26]):** starts clipped at the axis top
(≥ 0.06 at iteration ≲ 3; first fully visible mid-value 0.0593 at iteration 3) and then

| first iteration at which the red mid-line falls below | |
|---|---|
| 0.05 | ≈ 15 |
| 0.04 | ≈ 22 |
| 0.03 | ≈ 25 |
| 0.025 | ≈ 26 |
| 0.02 | ≈ 35 |
| 0.01 | ≈ 49 |
| 0.005 | ≈ 64 |
| 0.002 | ≈ 116 |
| 0.0015 | ≈ 135 |

ending at 0.00107 at iteration 199. There is a conspicuous shoulder: the red curve is flat
around 0.055–0.05 for iterations 5–15, then drops steeply from ≈ 0.041 (it. 20) to ≈ 0.025
(it. 26).

**Headline number:** superposition reaches ≈ 0.0005 within **≈ 10 iterations**; the iterative
scheme has not reached 0.001 after **200 iterations**, i.e. a factor ≳ 20 in iteration count.

Caption Fig 6 (verbatim): *"Efficient optimization: Optimizing examples iteratively leads to a
slow convergence behavior (red), whereas training in superposition (blue) leads to a drastically
faster convergence."*

Text (verbatim): *"Figure 6 shows the convergence behavior when using an iterative sample based
optimization scheme as presented ín [26], compared to our proposed training in superposition and
when using the KL-divergence. Training in superposition leads to a much faster convergence
behavior."*

⚠ "Iteration" is not defined and almost certainly means something different on the two curves
(one KW/SA step over the whole superposed state vs. one step over a single sample). See §9.

---

## 4. §III.C The Iris experiment — encoding, Fig 7, Fig 8, Fig 9, the 98 % claim

### 4.1 Binary encoding protocol (verbatim, p. 6)

> "In the next experiment, we use the Iris-dataset as example to train on a dataset which is not
> expressible with simple Hadamard-encodings. Thus, state preparation is necessary. Our workflow
> is as follows: First, the input data is encoded as a higher-dimensional binary vector. Taking
> the iris dataset as a toy example, it consists of 4 dimensional data encoding sepal length,
> sepal width, petal length and petal width. A kMeans clustering on each dimension with $k = 3$
> is used on the training data. Thus, every datapoint can be encoded in a
> $4 \times 3 = 12-$dimensional binary vector which contains exactly 4 non-zero entries. Thus, a
> binary encoding is used to represent the dataset. Please note, that each binary encoding
> corresponds to a unique entry in a $2^{12}$-dimensional vector space. Thus, 12 qubits can be
> brought in a superposition to represent the entire dataset. Therefore, quantum state
> preparation can be used, as suggested in [8, 20, 21]."

(The text dump renders `$2^{12}$` as "21 2-dimensional"; the PDF shows `2¹²`.)

> "It should be noted, that state preparation of an entire dataset can be a huge effort which
> might lead to the need of alternative approaches for optimization and representation."

> "Our choice of preprocessing via classical clustering and binarization simplifies the
> classification task while encoding structure into the input. Our intention is to provide a
> uniform, hardware-friendly binary encoding which can also be used for other datasets while
> staying within the constraints of current qubit counts and connectivity. We would also like to
> clarify that this is just one approach to represent data with the main purpose of demonstrating
> our proposed concepts."

This is the *same* protocol as SNN §IV.B, quoted almost word for word — see the SNN doc.
QRAM ([7, 23]) is mentioned and explicitly excluded: *"Both concepts are not further
investigated in this work and will be part of future research."*

### 4.2 **Fig 7** — train/test loss, classic vs. prototype

`literature/figs/fig07-iris-loss.png` (3142×791 px). Two axes side by side.
Legend in both: **blue** = `Loss Train`, **red** = `Loss Test`.

Caption (verbatim): *"Train (blue) and test (red) loss during optimization using (left) the
classical perceptron model and (right) the prototype model. The prototype model shows better
generalization performance."*

Text (verbatim): *"Figure 7 shows the train and test loss during optimization using (left) the
classical perceptron model and (right) the prototype model. While both models can explain the
training data very well, the prototype model (with the same amount of parameters) on the right
shows better generalization performance."*

**Left panel (classical perceptron).** Axes: x `0 … 15` (frame x-px 42 → 1521),
y `0 … 0.25` (frame y-px 759 → 129, ticks every 0.05). Reading error **±0.0015**.

| x | train (blue) | test (red) |
|---|---|---|
| 1 | 0.2427 | 0.1933 |
| 2 | 0.1956 | 0.1603 |
| 3 | 0.1234 | 0.1056 |
| 4 | 0.0786 | 0.0639 |
| 5 | 0.0502 | 0.0528 |
| 6 | 0.0369 | 0.0456 |
| 7 | 0.0290 | 0.0353 |
| 8 | 0.0242 | 0.0371 |
| 9 | 0.0198 | 0.0329 |
| 10 | 0.0151 | **0.0290** (minimum) |
| 11 | 0.0105 | 0.0357 |
| 12 | 0.0075 | 0.0333 |
| 13 | 0.0040 | 0.0361 |
| 14 | 0.0020 | 0.0444 |
| 15 | 0.0008 | 0.0446 |

The curves cross at x ≈ 5 (both ≈ 0.051); the test loss then rises from its minimum 0.029
(x = 10) to 0.045 (x = 15) — this is the overfitting signature the paper is pointing at.

**Right panel (prototype model).** Axes: x `0 … 30` (frame x-px 1637 → 3130),
y `0 … 0.3` (frame y-px 766 → 129, ticks every 0.05). **The curves only run to x ≈ 26.**
Reading error **±0.0015**.

| x | train (blue) | test (red) |
|---|---|---|
| 1 | 0.2694 | 0.2129 |
| 2 | 0.2647 | 0.2096 |
| 3 | 0.2553 | 0.2011 |
| 4 | 0.2411 | 0.1903 |
| 5 | 0.2199 | 0.1787 |
| 6 | 0.1990 | 0.1674 |
| 7 | 0.1724 | 0.1509 |
| 8 | 0.1361 | 0.1241 |
| 9 | 0.0944 | 0.0965 |
| 10 | 0.0721 | 0.0827 |
| 11 | 0.0626 | 0.0754 |
| 12 | 0.0530 | 0.0685 |
| 13 | 0.0490 | 0.0659 |
| 14 | 0.0457 | 0.0615 |
| 15 | 0.0393 | 0.0504 |
| 16 | 0.0318 | 0.0445 |
| 17 | 0.0278 | 0.0436 |
| 18 | 0.0250 | 0.0379 |
| 19 | 0.0217 | 0.0353 |
| 20 | 0.0188 | 0.0318 |
| 21 | 0.0170 | 0.0287 |
| 22 | 0.0151 | 0.0264 |
| 23 | 0.0139 | 0.0235 |
| 24 | 0.0127 | 0.0226 |
| 25 | 0.0125 | 0.0224 |
| 26 | 0.0122 | 0.0228 |

Train and test cross at x ≈ 9 (≈ 0.095) and the test loss decreases monotonically to
≈ 0.0224 at x = 25 with a hint of an upturn (+0.0004) at x = 26 — no overfitting within the
plotted range.

⚠ The two panels have **different x ranges (15 vs 30) and different y ranges (0.25 vs 0.3)**,
and the loss function itself is never named for this experiment (Eq 4 KL? ℓ₂?). Comparing the
two panels quantitatively is therefore not licensed by the figure.

### 4.3 **Fig 8** — Iris histograms

`literature/figs/fig08-iris-hist.png` (1337×754 px). Four panels.

Caption (verbatim): *"Top: Training data of the iris dataset. Each bin is a data sample and the
amplitude encodes the class label (three class labels are available). Bottom: Performance of the
prototype model on the test dataset with amplitude encoded predicted class labels."*

Text (verbatim): *"Figure 8 shows at the top the training data of the iris dataset. Each bin of
the histogram is a dataset example and the amplitude encodes the class label. The right side
shows the ground truth histogram (to be learned by the model). As there are three class labels
in the dataset, the histogram shows three different height levels. The left side shows the
prediction of the training dataset. […] The bottom image shows on the left the predicted
amplitudes using our prototype neural network model and on the right the ground truth."*

So: **left column = model prediction, right column = ground truth; top row = training set,
bottom row = test set.**

* Top row: **15 bars**, y axis `0 … 0.1`. x labels are 12-bit strings with exactly four `1`s
  (e.g. `010010001100`, `001001001100`), matching the 4×3 encoding.
* Bottom row: **30 bars**, y axis `0 … 0.06`, same 12-bit label format.

Ground-truth levels read off (±0.0004):

| panel | levels (read) | count at each level |
|---|---|---|
| top-right (train GT) | 0.0113, 0.0443, ≥ 0.099 (clipped at the axis top) | 3, 4, 8 |
| bottom-right (test GT) | 0.0065, 0.0260, 0.0585 | 7, 12, 11 |

**Amplitude-encoding law, inferred and checked:** the three levels are in the ratio
**1 : 4 : 9**, i.e. `p ∝ (class label)²` with labels 1, 2, 3 — an *amplitude* encoding of the
label, as the caption says. Normalisation check for the test panel:
`Σ c² = 7·1 + 12·4 + 11·9 = 154`, giving `1/154 = 0.006494`, `4/154 = 0.025974`,
`9/154 = 0.058442` — matching the read 0.0065 / 0.0260 / 0.0585 to ±0.0001, and the 30 bars sum
to 1.001. Training panel: `Σ c² = 3·1 + 4·4 + 8·9 = 91`, giving `0.010989 / 0.043956 / 0.098901`
— matching the read 0.0113 / 0.0443 / (clipped) to ±0.0004.
**[INFERENCE from the figure; the paper never writes this rule down.]**

Model panels (left) show the same bars with visible deviations, e.g. top-left levels read
0.0138, 0.0197, 0.0463, 0.0498, 0.0512, 0.0931, 0.0956, 0.1(clipped); bottom-left spans
0.0083 … 0.060.

⚠ 15 training bars vs 30 test bars is *backwards* for any normal Iris split. Not explained.

### 4.4 The 98 % claim (verbatim)

> "Even though differences are notable, when applying the regressed values to the nearest class
> labels, the performance of our model is in the range of 98% accuracy. This is in accordance to
> [26]."

Nothing else is given: no split size, no seed, no repetitions, no variance. (For contrast the
SNN paper reports 95 % ± 5 for Iris with its own model — see the SNN doc — so "in accordance to
[26]" is generous.)

### 4.5 **Fig 9** — t-SNE of data vs. weights

`literature/figs/fig09a-tsne-classic.png` (993×563) and `fig09b-tsne-proto.png` (1011×563).

* **Left (classic).** Axes x `−400 … 500`, y `−400 … 500`, gridlines every 100.
  Legend: red = `Data Points`, blue = `Hidden Weights Positions`. 22 red points spread over the
  whole square; **4 blue points tightly clustered** at approximately
  `(120, 149)`, `(70, 54)`, `(176, 41)`, `(110, −37)` (read from figure, ±8).
* **Right (prototype).** Axes x `−1000 … 1000`, y `−1000 … 1000`, gridlines every 200.
  Legend: red = `Data Points`, blue = `Prototype Positions`. 20 red points; **4 blue points
  spread across the plot** at approximately `(−930, −765)`, `(−95, 490)`, `(710, 80)`,
  `(820, −175)` (read from figure, ±15).

Caption (verbatim): *"Left: TSNE-plot of the training data (red) and the weights of the hidden
layer (blue) when using the classical architecture. Right: TSNE-plot of the training data (red)
and the weights of the prototype layer (blue) when using the prototype architecture. Whereas the
hidden data reveal only limited information about the dataset, are spanning the weights of the
prototype layer much better the dataset. These weights can be easier as-signed to samples from
the dataset and used for explanation."*

⚠ The two panels use different axis scales, so "spanning the dataset better" is partly an
artefact of the plotting; and t-SNE embeddings of two different runs are not comparable.

---

## 5. §III.D Flappy Bird — **Fig 10**, **Fig 11**

Text (verbatim, p. 7–8):

> "Figure 10 shows the pipeline for training an agent to play Flappy Birds. From the image, the
> position of the bird, as well as the position of the tubes is translated to (input) qubit
> states. Based on this input, which are encoded by using RX-Gates, a quantum neural network
> decides, whether to fly upwards or downwards."

> "Our trained model plays the game perfectly, we stopped a run after the score of 450.000 which
> took 6 days. Figure 11 shows a few images of the gameplay. As model we use our
> prototype-architecture with 4 hidden neurons."

### 5.1 **Fig 10** circuit — 7 qubits

`literature/figs/fig10-flappy.png` (2605×1800 px). Left half: a game screenshot with
`FPS: 60.17`, `collide = 0.00`, score `0`, and two blue boxes marking the bird's bounding box
and the tube gap, with two black leader lines into the circuit. Right half: the circuit.

* **7 wires**, numbered 1…7.
* Wires **1, 2**: the two inputs, each with a single un-controlled `RX` at the start:
  * wire 1: `RX(1.9537)`
  * wire 2: `RX(2.1962)`
  These are the *state encoding* (bird position / tube position), not learnable weights.
* Wires **3, 4, 5, 6**: the **4 hidden neurons**.
* Four `Pro…` composite blocks, each with **3 ports**: ports 1, 2 → wires 1, 2 and port 3 →
  wire `2 + k`, k = 1…4.
* Wire **7**: output neuron — classic (Fig-1) form:
  * `RX(-1.4277)` (bias, at the very start of the wire),
  * `CRX(-1.0422)` control wire 3,
  * `CRX(-0.9988)` control wire 4,
  * `CRX(-1.2622)` control wire 5,
  * `CRX(-0.91002)` control wire 6.
* The last `CRX` box is highlighted in blue and a blue ⇅ arrow sits to its right — the measured
  output of wire 7 is the up/down decision.

All eight numbers above are printed at legible size; read from figure with the **last digit
uncertain by ±1**.

2 inputs + 4 hidden + 1 output = 7 qubits ✓ consistent with "4 hidden neurons".

Caption Fig 10 (verbatim): *"Training an agent to play Flappy Birds. From the image, the
position of the bird, as well as the position of the tubes is translated to qubit states. The
quantum neural network decides, whether to fly upwards or downwards."*

### 5.2 **Fig 11** — gameplay frames

`literature/figs/fig11-flappy-frames.png` (1046×746 px). A 2 × 5 grid of 10 game screenshots,
scores in reading order `0, 1, 2, 3, 3, 4, 5, 6, 6, 7`; each carries an `FPS: …` readout
(values 20.93 … 72.49) and `collide = 0.00`. Caption: *"Example frames of the Flappy Birds
agent."*

### 5.3 What the input angles do (and do not) tell us

`RX(1.9537)` and `RX(2.1962)`. Under `P(|1⟩) = sin²(θ/2)` these encode
`s = sin²(θ/2) = 0.6866` and `0.7927`; under `θ = π·s` they encode `s = 0.6218` and `0.6991`.
Both are admissible. The naive `θ = 2s` is **excluded**: it would need `s = 1.098 > 1`.
See §6.4 for the (unsuccessful) attempt to pin this down from the ECG figure, and the SNN doc
§3 for the convention SNN actually uses (`θ = 2·asin(√a)`, verified numerically there).

---

## 6. §III.D ECG denoising — **Fig 12**, **Fig 13**, **Table I**, Eqs (6)–(8)

Text (verbatim, p. 8):

> "Motivated from future applications in direct combination of quantum sensors and AI driven
> quantum signal processing, we show a small use case in the final experiment: Figure 12 shows
> the concept for a sliding window based signal denoising task. We follow an architecture similar
> to Figure 10. Thus, five consecutive (noisy) samples at time point $t$,
> $[s(t-2) \dots s(t+2)]$ are used to predict the value $s(t)$. As time-series data an ECG-signal
> use case has been used. The goal is to denoise a signal overlaid with stochastic noise."

Caption Fig 12 (verbatim): *"Concept for sliding window based denoising an ECG-signal using a
Prototype Quantum neural network. A small window of 5 samples is used as input for the neural
network and the smoothed value of the middle position is estimated."*

### 6.1 **Fig 12** circuit — 9 qubits, 3 prototype blocks

`literature/figs/fig12-ecg-arch.png` (3077×2003 px). Composite figure: full noisy signal
(bottom left), zoomed window (top left), circuit (right), denoised output zoom (far right).

* **9 wires**, numbered 1…9.
* Wires **1–5**: the five window samples `s(t−2) … s(t+2)`, each with one un-controlled `RX`
  inside a blue highlight box:

  | wire | `RX` angle as printed |
  |---|---|
  | 1 | `0.54454` |
  | 2 | `0.78147` |
  | 3 | `0.51643` |
  | 4 | `0.70226` |
  | 5 | `0.78199` |

  (read from figure; last digit uncertain by ±1. Wire 4 could also read `0.70225`;
  wire 5 could also read `0.78190`.)
* Wires **6, 7, 8**: the **3 hidden / prototype neurons**. Three `Pro…` blocks, each with
  6 ports: 1–5 → wires 1–5, port 6 → wire `5 + k`, k = 1…3.
* Wire **9**: output neuron, classic form:
  * `RX(-1.0211)` (bias),
  * `CRX(-1.0925)` control wire 6,
  * `CRX(-0.97304)` control wire 7,
  * `CRX(-1.0781)` control wire 8 (highlighted in blue; an arrow leads from it to the denoised
    signal inset).

5 inputs + 3 hidden + 1 output = 9 qubits. This corresponds to the `#LS,PT = 3` row of Table I.

### 6.2 The zoomed window (top-left inset of Fig 12)

Axes calibration, measured from the tick marks:
* y ticks `0.05, 0.10, …, 0.50` at rows 709.5 … 19.0 (76.8 px per 0.05).
* x ticks every 10 samples, 16.1 px per sample; the labelled ticks are `100` (px 1308) and
  `110` (px 1469).
* The blue shaded window spans px 580 … 687, i.e. sample indices **≈ 54.8 … 61.4**.

The red curve inside the window is drawn in a desaturated red (blue overlay). Its vertices were
recovered by least-squares fitting each straight segment and intersecting neighbours — the
intersections land at sample indices 55.04, 56.05, 57.01, 58.03, 59.03, 60.05, confirming the
16.1 px/sample calibration to better than 0.05 samples:

| sample n | value (read from figure, **±0.004**) |
|---|---|
| 55 | 0.2108 |
| 56 | 0.3344 |
| 57 | 0.3329 |
| 58 | 0.3631 |
| 59 | 0.2597 |
| 60 | 0.3144 |

Five black leader lines start on the curve. Their left endpoints (detected as connected black
components) are at pixel `(633, 234)`, `(619, 274)`, `(605, 281)`, `(652, 390)`, `(578, 477)`,
i.e. they attach to samples **58, 57, 56, 59, 55** — the first five samples of the window,
`n = 55 … 59`, values `{0.2108, 0.3344, 0.3329, 0.3631, 0.2597}`.

### 6.3 Signal statistics from Fig 13

`literature/figs/fig13-ecg-result.png` (1651×475 px). Single axes, frame x-px 78.5 → 1625.5,
y-px 406.5 → 8.5.

* **x axis** `t`, ticks `0, 200, …, 2000`. **Plotted data run t ≈ 0 … 1800**; the axis extends
  to 2000 with no data beyond ≈ 1801.
* **y axis** `amplitude`, ticks `0, 0.2, 0.4, 0.6, 0.8, 1`.
* Legend: **red** `Noisy Input`, **green** `Ground truth`, **blue** `PT-QNN-Denoise`.

Read values (**±0.005** unless noted):

| quantity | value |
|---|---|
| signal **length** | ≈ 1800 samples (t = 0 … 1800) |
| noisy input, full range | 0.052 … 0.954 |
| ground truth, full range | 0.117 … 0.903 |
| denoised (blue), full range | 0.097 … 0.883 |
| noise, peak-to-peak (mean vertical spread of the red trace per column, minus the ≈ 0.013 line width) | **≈ 0.094**, i.e. amplitude **≈ ±0.047** (±0.01) |
| ground-truth line spread (line width only) | 0.020 |

Ground-truth local maxima (t, amplitude), read to ±10 in t and ±0.006 in amplitude:

| t | amplitude | note |
|---|---|---|
| 80 | 0.343 | small hump |
| **370** | **0.866** | large beat 1 |
| 732 | 0.378 | small hump |
| 797 | 0.338 | small hump |
| 1003 | 0.356 | small hump |
| **1269** | **0.853** | large beat 2 |
| 1573 | 0.334 | small hump |

**Number of beats:** **2** large beats (t ≈ 370 and t ≈ 1269, separation ≈ 899 samples) plus
5 smaller humps in the 1800-sample record. The waveform is a smooth quasi-periodic envelope
rather than a QRS-shaped ECG. Troughs sit at ≈ 0.13 … 0.20.

Caption Fig 13 (verbatim): *"Denoising an ECG-signal. The noisy input signal is shown in red,
the ground truth signal is shown in green and the denoising result of our Prototype quantum
neural network is shown in blue (best viewed in color and zoom in)."*

### 6.4 The value → angle encoding: what Fig 12 does and does not settle

Task: decide between `θ = π·s`, `θ = 2·asin(√s)`, and others, using the five printed angles and
the five read-off signal values.

**Result: the five Fig-12 angles are inconsistent with *any* monotone encoding of the five
values in the shaded window.** Proof: samples 56 and 57 have essentially the same value
(0.3344 vs 0.3329, Δ = 0.0015 ± 0.006, i.e. indistinguishable), while sample 58 is clearly
higher (0.3631). Any monotone `f` must therefore send {56, 57} to a near-degenerate angle pair
and 58 to a strictly larger angle. The angle multiset is
`{0.51643, 0.54454, 0.70226, 0.78147, 0.78199}` — the only near-degenerate pair is
`(0.78147, 0.78199)`, so `f(58)` would have to exceed 0.78199 while the remaining angles are
0.51643, 0.54454, 0.70226 — a contradiction. The sorted-to-sorted (monotone) pairing gives
`0.3329 → 0.70226` and `0.3344 → 0.78147`, i.e. `Δθ = 0.079` for `Δs = 0.0015`, which no smooth
map can produce given the other pairs. Likewise the *shape* of the two sequences disagrees:
values go up, flat, up, down; angles (wire order) go up, down, up, up.

**Conclusion:** the circuit screenshot in Fig 12 was taken at a *different* window position from
the one shaded in the plot. **[INFERENCE, but the argument above is tight.]** Do not use Fig 12
to calibrate the encoding.

What *can* be said:

* `θ = 2·s` is **excluded** by Fig 10 (`2.1962/2 = 1.098 > 1`).
* `θ = π·s` implies, for Fig 12's angles, `s ∈ {0.1644, 0.1733, 0.2235, 0.2488, 0.2489}` —
  values that sit in the baseline band of the Fig 13 signal, so it is not excluded.
* `θ = 2·asin(√s)` implies `s ∈ {0.0652, 0.0723, 0.1183, 0.1451, 0.1452}` — near the bottom of
  the signal's range (min 0.052), so it is not excluded either.
* **The prerequisite paper settles it:** SNN Fig 4 prints `CRX(1.369)` labelled `[0.4]` and
  `CRX(1.7722)` labelled `[0.6]`, and `2·asin(√0.4) = 1.36944`, `2·asin(√0.6) = 1.77215`.
  So SNN's convention is **`θ = 2·asin(√a)`**, i.e. `P(|1⟩) = sin²(θ/2) = a`. See
  `docs/physics/rosenhahn_osborne_hirche_snn.md` §3. PTQNN never restates it.
  **Recommended `[ASSUMPTION]` for code: `θ = 2·asin(√s)` with `s` normalised to [0, 1].**

### 6.5 Eqs (6)–(8): PSNR and SSIM

> "The Peak Signal-to-Noise Ratio (PSNR) is defined as:"

```latex
\mathrm{PSNR} = 10 \log_{10}\!\left( \frac{R^2}{\mathrm{MSE}(s_1, s_2)} \right)
\tag{6}
```

```latex
= 20 \log_{10}\!\left( \frac{R}{\sqrt{\mathrm{MSE}(s_1, s_2)}} \right)
\tag{7}
```

> "with MSE denoting the Mean Squared Error and $R$ denoting the maximum fluctuation in the
> signal. The Structural Similarity (SSIM) index is defined as"

```latex
\mathrm{SSIM}(s_1, s_2) =
\frac{(2 \mu_{s_1}\mu_{s_2} + C_1)(2\sigma_{s_1 s_2} + C_2)}
     {(\mu_{s_1}^2 + \mu_{s_1}^2 + C_1)(\sigma_{s_1}^2 + \sigma_{s_1}^2 + C_2)}
\tag{8}
```

> "where $\mu$ and $\sigma$ denote the mean, variance and covariance of the signals and $C$ are
> (small) stabilizer constants to avoid division by zero."

⚠ **Eq (8) as printed is wrong**: the denominator repeats the `s₁` subscript in both factors.
The standard SSIM denominator is `(μ_{s₁}² + μ_{s₂}² + C₁)(σ_{s₁}² + σ_{s₂}² + C₂)`. Use the
standard form; record the paper's typo. `C₁`, `C₂` and `R` are never given numerically, and the
SSIM window/σ is never specified.

### 6.6 **Table I** (reproduced in full, verbatim)

Caption (verbatim): *"Evaluation of reconstruction quality for noisy ECG signals for different
amount of prototype or latent space layers. CQNN denotes the classical quantum neural network
with 1-3 neurons as latent space dimension in the hidden layer and PTQNN denotes the prototype
model with 1-3 prototype layers. Measured are the Peak signal to noise ration (PSNR) and the
structural similarity index (SSIM). Already a few prototypes or neurons are sufficient to
denoise the ECG-signal significantly."*

|            | CQNN PSNR ↑ | CQNN SSIM ↑ | PTQNN (ours) PSNR ↑ | PTQNN (ours) SSIM ↑ |
|------------|-------------|-------------|---------------------|---------------------|
| **#LS,PT = 1** | 32.72   | 0.90        | 32.9                | 0.87                |
| **#LS,PT = 2** | 38.20   | 0.90        | 40.00               | 0.93                |
| **#LS,PT = 3** | 39.65    | 0.92        | 40.04               | 0.93                |

(The column header is `#LS,PT`; entries are transcribed exactly, including the inconsistent
significant figures `32.9` vs `32.72` and `0.87` vs `0.90`.)

Accompanying text (verbatim):

> "Table I shows that already a few prototypes are sufficient to denoise the ECG-signal
> significantly. Our model outer-performs the classical perceptron, but it should be noted that
> both models solve the task well."

⚠ Note the row `#LS,PT = 1`: PTQNN's SSIM (0.87) is *worse* than CQNN's (0.90) while its PSNR is
marginally better — so "outer-performs" is not uniform.

---

## 7. §IV Summary (verbatim highlights)

> "In contrast to conventional perceptron approaches that rely solely on weighted summation, it
> is well known that such models are susceptible to overfitting and limited interpretability.
> Instead, our method leverages learnable prototypes and XOR-type feature comparisons to enable
> both zero- and one-valued input features to actively contribute to neuron activation."

> "The second main contribution of this work lies in the proposed quantum optimization strategy,
> which encodes the complete training dataset as a superposed quantum state, thus permitting
> simultaneous, global parameter updates by directly minimizing the divergence between target and
> model distributions."

Funding: BMFTR KISSKI (01IS22093C), QUICS (13N17418), Quantum Valley Lower Saxony, EXC-2122
PhoenixD, EXC-2123 Quantum Frontiers. The acknowledgement thanks *"Tobias Osborne from the
Institute for Theoretical Physics (ITP) at the Leibniz University of Hannover, Germany for
fruitful discussions and hints in this project."*

---

## 8. Cross-figure invariants worth testing against

Collected here because these are the checkable facts a reimplementation must reproduce:

1. Fig 2 gate list (§2.1) — six gates, no plain bias on the output wire, shared `θ`.
2. Bit-count target `p(x) = (popcount(x) + 1) / 112` (§3.6), confirmed to ±0.0002 on all 32 bins.
3. Fig 4: 12 qubits = 5 + 6 + 1; Fig 5: 11 qubits = 5 + 5 + 1 (§3.4, §3.5).
4. Fig 10: 7 qubits = 2 + 4 + 1 (§5.1). Fig 12: 9 qubits = 5 + 3 + 1 (§6.1).
5. The output neuron in *every* architecture figure (4, 5, 10, 12) is a **classic** Fig-1
   neuron — un-controlled bias `RX` followed by one `CRX` per hidden neuron — **not** a
   prototype neuron. Only the hidden layer uses `Pro…` blocks.
6. Iris label amplitude encoding `p ∝ label²`, normalised (§4.3).
7. Fig 6: superposition converges by iteration ≈ 10, iterative does not reach 1e-3 by 200.

### 8.1 Which software produced the figures — inference

Neither paper names its simulator. However:

* The bar colour in Fig 8 is exactly `RGB(0, 114, 189)` = MATLAB's default
  `[0 0.4470 0.7410]`; Figs 4/5 use `RGB(17, 113, 190)` and `RGB(112, 170, 216)` from the same
  family.
* The circuit renderings in Figs 4, 5, 10, 12 (numbered wires on the left, grey `RX` boxes with
  the numeric angle typeset *underneath* the gate name, composite gates drawn as tall boxes with
  numbered ports and an ellipsis-truncated label `Pro…`) are the exact output style of the
  **MATLAB Support Package for Quantum Computing** (`quantumCircuit` / `compositeGate` +
  `plot`).
* Figs 3, 6, 7, 9, 13 are MATLAB `surf` / `plot` with default jet / lines colormaps and MATLAB
  tick typography.

**[INFERENCE, high confidence: the experiments were run and plotted in MATLAB, not
Qiskit/PennyLane.]** This is *not* stated anywhere in either paper.

---

## 9. What the paper does not specify

Everything in this section is a gap that a reimplementation must fill with an `[ASSUMPTION]`
(see `PLAN.md §0`).

**Model definition**

1. **The prototype neuron's unitary is never written down.** Fig 2 is the only definition.
   In particular: whether the CRX angle `θ` really is shared across all inputs of one neuron
   (both boxes print `θ`, and §III.C's "with the same amount of parameters" requires it) or
   whether the shared symbol is just sloppy drawing. §III.B says the model *"does require an
   increase of optimization parameters"*, which contradicts §III.C's *"with the same amount of
   parameters"*. **Unresolved contradiction inside the paper.**
2. Sign/direction convention: whether `RX(φ)` means `exp(-iφX/2)` (Qiskit/MATLAB) or
   `exp(-iφX)`. Never stated; every angle in the paper is only meaningful modulo this.
3. How a data value is mapped to a rotation angle (see §6.4). PTQNN never states it; SNN states
   `asin(√aᵢ)` in words and prints `2·asin(√a)` in its Fig 4.
4. Whether the prototype angles `φᵢ` are unconstrained reals or restricted (e.g. to `{0, π}` for
   binary prototypes).
5. How the 5 input bits are prepared in Fig 4 (the shaded box has no gates); presumably `X`
   gates, but the figure is silent.
6. What the output "amplitude encoding" means precisely — whether the model distribution is
   `P(inputs = x, out = 1)` renormalised over `x` (as `CLAUDE.md` assumes) or `P(out = 1 | x)`.
   The paper only says *"The output neuron gives a probability of measurement. It yields an
   amplitude encoding of the desired bit count."*

**Optimiser**

7. **No hyper-parameters at all.** `aₙ`, `cₙ` (Kiefer–Wolfowitz), `k`, the initial temperature
   `T₀`, the annealing schedule `Tₜ`, the neighbourhood/perturbation distribution for `x̂ₜ`,
   the number of KW directions per step, stopping criterion, parameter initialisation, RNG seed
   — none are given, in either paper.
8. What one "iteration" is on each curve of Fig 6 (they are almost certainly different units for
   the red and blue curves; the comparison is therefore not apples-to-apples as plotted).
9. Number of measurement shots, or whether probabilities are computed exactly from the state
   vector. Both papers write losses as sums over exact probabilities, which implies exact
   simulation, but never say so.
10. Which simulator / language (see §8.1 — MATLAB is inferred from the figure style, not stated).

**Data and protocol**

11. **ECG data source.** No dataset name, no sampling rate, no citation, no preprocessing.
    Fig 13's `t` axis is in samples, not seconds. The noise model ("stochastic noise") is not
    specified — not even Gaussian vs. uniform, nor its σ (we measure peak-to-peak ≈ 0.094).
12. `R`, `C₁`, `C₂` and the SSIM window for Eqs (6)–(8).
13. What "#LS,PT" counts exactly for CQNN vs. PTQNN in Table I (latent-space *neurons* vs.
    prototype *layers* — the caption uses both words), and whether the two models are
    parameter-matched.
14. Iris: train/test split sizes and procedure, whether kMeans is refit per fold, how ties are
    broken in the binarisation, how many repetitions the "98 %" is averaged over, and its
    variance. Fig 8 shows 15 training bars and 30 test bars, which is not explained.
15. Which loss is plotted in Fig 7 (KL of Eq 4? ℓ₂?), and why the two panels use different x and
    y ranges.
16. **Flappy Bird: the entire RL protocol.** No reward, no episode/rollout definition, no
    exploration schedule, no game implementation or physics constants, no description of how the
    bird/tube pixel positions are normalised to `[0, 1]`, no training-time or sample budget
    (only "score of 450.000 which took 6 days" for *inference*), no mention of how many hidden
    neurons vs. the 4 in Fig 10 were tried.
17. The state-preparation circuit for the Iris superposition is only cited ([8, 20, 21]), never
    given; its gate cost is not counted anywhere.

**Editorial defects to be aware of when quoting**

18. `"the former work XY"` appears twice (§II.B, §II.C) and `"As shown in XY"` once — unresolved
    `\cite` placeholders. From context all three mean reference [26] (= the SNN paper).
19. `"Figure shows 2 the core idea"` (§III.B) — transposed words.
20. Fig 3's caption says the mismatched-pattern activation "is zero"; the plotted surface bottoms
    out at ≈ 0.2 (read from figure, ±0.05) and its colour bar starts at 0.3.
21. Fig 3's caption describes prototype `|01⟩` with qubits named `q₀, q₁`; the body text
    describes prototype `[1 0]` with qubits named `q₁, q₂`. Same object, two incompatible
    namings.
22. Eq (8) SSIM denominator typo (§6.5).
23. Fig 4 (6 hidden neurons) vs Fig 5 (5 hidden neurons) for the same experiment, unremarked.
24. References [28] and [29] are the same paper (Snell, Swersky, Zemel, *Prototypical networks
    for few-shot learning*), cited twice under different keys.
