# Ground truth: Rosenhahn, Osborne & Hirche, *Stochastic Neural Networks for Quantum Devices*

Pin tag: **`SNN`**. arXiv:2602.22241v2 [quant-ph], 13 Aug 2026 (dated "August 14, 2026"),
19 pages. Source: `literature/pdf/2602.22241.pdf`, text dump `literature/2602.22241.txt`,
extracted figure rasters `literature/snn-figs/*.png`. Authors: Bodo Rosenhahn¹,
Tobias J. Osborne², Christoph Hirche¹ (¹ tnt/L3S, ² ITP/L3S, Leibniz Universität Hannover).

This is reference `[26]` of **PTQNN** (`docs/physics/rosenhahn_hirche_ptqnn.md`); PTQNN's
"classic perceptron" (its Fig 1) *is* the neuron defined here, and PTQNN's optimiser and Iris
binarisation are inherited verbatim from this paper. This distillation only covers what PTQNN
depends on, plus the noise analysis. Conventions are as in the PTQNN doc (verbatim text in
quotes; figure read-offs carry an explicit reading error).

---

## 1. §II.A — the classical perceptron, Eq (1)

Same as PTQNN Eq (1), same wording:

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

---

## 2. §II.B — the stochastic artificial neuron and its activation probability

Verbatim (p. 3):

> "Stochastic artificial neurons are perceptron models that incorporate a random component into
> their activity, which distinguishes them from deterministic perceptrons. Instead of providing a
> fixed output for a given input, stochastic artificial neurons are activating based on a
> probability."

> "A stochastic artificial neuron can be formulated as a scalar product of the input values and
> (learnable) weights with an added bias. The score gives the probability for the activation of
> the neuron,"

```latex
p(\sigma_j = 1) = p\!\left( \sum_i \omega_i a_i + b \right)
```

(*unnumbered*.)

> "In this work, we assume binary input values of qubits which means that $a_i \in \{0, 1\}$. To
> this end, the probabilistic artificial neuron can be seen as a modified version of the
> perceptron model, presented in [15]. The core differences are the probabilistic nature of
> activation, the absence of ancilla qubits, and no need for a repeat-until-success (RUS)
> circuit. Since the sine function is already nonlinear, our model shares similarities with
> Radial Basis Function networks [57]."

---

## 3. §III — the quantum perceptron: the angle ↔ probability convention

### 3.1 The stated rule (verbatim, §III, p. 4)

> "The general idea about modeling the behavior of a perceptron with probabilistic activation on
> a quantum computer is shown in Figure 3: The (binary) inputs are manipulating the activation
> probability of the perceptron by using RX-Gates (for the bias) and Controlled RX-Gates. The aim
> is to use input qubits to increase the probability of the perceptron activation by a value
> $a_i$. **To translate this probability score to an appropriate rotation we simply use
> $asin(\sqrt{a_i})$ as angular measure for the rotation.**" (emphasis ours)

Caption Fig 3 (verbatim):

> "Concept of a quantum perceptron with probabilistic activation. The bias is represented as an
> RX-Gate and the binary input qubits $a_i$ increment the probability of neuron excitation by
> their angular components $\omega_i$ using a controlled gate. To ensure a linear and additive
> behavior of the angles, directly mapping to the probabilities, the asin-function is used. The
> right image visualizes the qubit activation on the Bloch sphere (just the real part is shown)
> and the effect of rotating the qubits, based on the bias and CRX-gates. Please note, that for
> complex input signals, the model can be extended with CRY- and CRZ-gates in a straightforward
> way."

### 3.2 **Fig 3** circuit (vector figure, p. 5)

3 wires. Top two are the inputs, drawn with a bracket labelled `[a₁; a₂]`. Bottom wire is the
neuron, output labelled `o₁`. Gates, left to right:

1. `RX(ω₀)` on the bottom wire — the bias, un-controlled.
2. `CRX(ω₁)` — control = `a₁` (top wire), target = bottom wire.
3. `CRX(ω₂)` — control = `a₂` (middle wire), target = bottom wire.

To the right of the circuit is a Bloch-sphere sketch (real half-plane only, `|0⟩` and `|1⟩`
poles) with the three cumulative rotation levels marked `ω₀, ω₁, ω₂` stacked toward `|1⟩`.

**This circuit is identical to PTQNN Fig 1**, with `(ω₀, ω₁, ω₂) ≡ (θ, φ₁, φ₂)`.

### 3.3 **Fig 4** pins the convention numerically — `θ = 2·asin(√a)`

Fig 4 (raster, p. 5) draws the same 3-qubit circuit with the *numbers filled in*: inputs
labelled `[0/1; 0/1]`, and the two controlled gates printed as

* `CRX(1.369)` (control = **first/top** input), annotated underneath `[0.4]`,
* `CRX(1.7722)` (control = **second/middle** input), annotated underneath `[0.6]`,

output labelled `o₁`. There is **no bias `RX`** drawn in Fig 4 (i.e. `ω₀ = 0`).

Check:

```
2·asin(√0.4) = 2 × 0.6847192 = 1.3694384   → printed 1.369
2·asin(√0.6) = 2 × 0.8860771 = 1.7721542   → printed 1.7722
```

**Therefore the convention is `θ = 2·asin(√a)`, i.e. with `RX(θ) = exp(-iθX/2)` one has
`P(|1⟩) = sin²(θ/2) = a`.** The text's "we simply use `asin(√aᵢ)`" is shorthand for the
*half*-angle. This is the single most load-bearing fact in the paper and it is only recoverable
from the figure. **[Verified numerically to 5 digits; not an inference.]**

The additivity claim then holds exactly, because `asin(√a) + asin(√(1−a)) = π/2`:
`1.369 + 1.7722 = 3.1412 ≈ π`, so both inputs active ⇒ `P = sin²(π/2) = 1`.

### 3.4 Fig 4's four histograms (read from figure)

Below the circuit, four small bar plots, y axis `Probability` `0 … 1` with ticks every 0.2,
x axis the 3-bit measured basis state (order `a₁ a₂ o₁`); each has an inset legend naming the
input:

| legend | bars (state : probability) |
|---|---|
| `Input: 00` | `000` : 1.0 |
| `Input: 01` | `010` : 0.4, `011` : 0.6 |
| `Input: 10` | `100` : 0.6, `101` : 0.4 |
| `Input: 11` | `111` : 1.0 |

Read to **±0.02**; all values sit exactly on the 0.4/0.6/1.0 gridlines.

Caption Fig 4 (verbatim): *"Example activations for a perceptron with two binary inputs. The
weights are set to [0.4 0.6]. The bottom histograms demonstrate the additive increase of the
activation probability."*

Body text (verbatim, p. 3–4):

> "For the first qubit we set as activation probability the value 0.6 and for the second one the
> value 0.4. The combination of all possible states $(00, 10, 01, 11)$ leads to the observation
> probabilities shown in the histograms below: If no qubit is activated, the probability to
> measure 0 is one, the latter qubit raises the score to 0.6 and if only the front qubit (the
> upper one) is activated, the probability of the perceptron to be activated is 0.4. Finally, if
> both qubits are 1, the perceptron activates with probability 1.0."

⚠ The body text's first sentence ("first qubit … 0.6 … second one … 0.4") **contradicts the
figure and the caption**, where the first/upper control carries 0.4 and the second carries 0.6.
The rest of the same paragraph ("if only the front qubit (the upper one) is activated, the
probability … is 0.4") agrees with the figure. Trust the figure: **upper input ↦ 0.4, lower
input ↦ 0.6**.

### 3.5 Expressiveness: Eq (2), the XOR

> "Please note that cumulative values greater than 1 lead to a decrease of the activation
> probability. This allows the perceptron to be more expressive than a classical perceptron with
> e.g. a sigmoid activation. For example, the weights $(\omega_1 = 1, \omega_2 = 1)$ will lead to
> the implementation of a xor-function,"

```latex
\begin{aligned}
|00\rangle &\to |0\rangle \\
|01\rangle &\to |1\rangle \\
|10\rangle &\to |1\rangle \\
|11\rangle &\to |0\rangle .
\end{aligned}
\tag{2}
```

With `ω = 1` the angle is `2·asin(1) = π` per gate, so two active inputs give `2π` and
`sin²(π) = 0` — exactly the "cumulative values greater than 1 lead to a decrease" statement.

> "In our experiments we mainly use binary encodings and each perceptron is realized by
> CRX-gates. Please note, that for the case of complex input data, e.g. stemming from a quantum
> sensor, the CRX-gates can easily be extended to CRY- and CRZ-gates for quantum signal
> analysis."

---

## 4. §III.A — optimisation: Kiefer–Wolfowitz inside simulated annealing

### 4.1 Kiefer–Wolfowitz (unnumbered)

> "The Kiefer–Wolfowitz algorithm was introduced in 1952 [40] and was motivated by the
> Robbins–Monro algorithm [61]. The algorithm has been presented as a method which can
> stochastically estimate the maximum (or minimum) of a function. Let $M(x)$ be a function which
> has (wlog) a maximum at the point $\theta$. It is further assumed that $M(x)$ is reasonably
> smooth. The structure of the algorithm follows a gradient-like method, with the iterates being
> generated as"

```latex
x_{n+1} = x_n + a_n \left( \frac{M(x_n + c_n) - M(x_n - c_n)}{2 c_n} \right)
```

> "where $M(x_n + c_n)$ and $M(x_n - c_n)$ are independent. Thus, the gradient of $M(x)$ is
> approximated by a central difference method with a damping factor $a_n$ slightly shifting the
> solution towards an optimum."

(Identical display to PTQNN §II.C.)

### 4.2 Simulated annealing, Eq (3)

> "Our method works in a simulated annealing scheme: Simulated Annealing (SA) is a probabilistic
> technique to approximating the optimum of a given function [43]. […] As an optimization scheme
> the algorithm works iteratively with respect to time $t$ given a state $x_t$, similar to the
> Kiefer-Wolfowitz algorithm. At each step, the simulated annealing heuristic samples a
> neighboring state $\hat{x}_t$ of the current state $x_t$, **e.g. based on an approximate
> gradient**. Then a probabilistic decision is made to decide whether to move to the new state
> $x_{t+1} = \hat{x}_t$ or to remain in the former state $x_{t+1} = x_t$. The probability of
> making the transition from the current state $x_t$ to the new state $\hat{x}_t$ is defined by
> an acceptance probability function $P(e(x_t), e(\hat{x}_t), T)$. The function $e(x)$ evaluates
> the energy of this state, which is in our case **the fitness score given by the optimization
> task (e.g. the $\ell_2$-loss)**. The parameter $T$ is a time-dependent variable dictating the
> behavior of the stochastic process according to a cooling scheme or annealing schedule. The $P$
> function is typically chosen in such a way that the probability of accepting an uphill move
> decreases with time and it decreases as the difference $e(\hat{x}_t) - e(x_t)$ increases."
> (emphasis ours)

```latex
P\big(e(x_t), e(\hat{x}_t), T\big) \propto
\exp\left( - \frac{e(\hat{x}_t) - e(x_t)}{kT} \right)
\tag{3}
```

> "with $k$ a damping factor $k > 0$. Thus, our optimization is based on a local stochastic
> search scheme which is no strict gradient descent. Please note that several neural networks
> have special requirements on the networks, such as shared weights or symmetry properties. Using
> stochastic search, as proposed here, can ensure certain constraints during optimization."

**Stated hyper-parameters: none.** `aₙ`, `cₙ`, `k`, `T₀`, the cooling schedule, the
neighbourhood distribution, the iteration budget and the initialisation are all unspecified,
here and in PTQNN. (One initialisation *is* given, but only for the RBM/AE capacity experiment
of §IV.E.1: *"All involved weights are initialized with zero. Thus, the models start their
optimization with the same loss."*)

### 4.3 Parameter-shift rules — the RX rule (unnumbered) and **Eq (4)**

> "We will compare this approach to standard gradient descent based optimization using auto
> differentiation. An alternative realization of gradient descent can be based on the parameter
> shift rule [78]. If the gate $G$ has exactly two distinct eigenvalues $\pm r$ (which applies to
> all standard Pauli rotation gates such as $R_x$, $R_y$, $R_z$ with $r = \frac{1}{2}$), the
> exact derivation is given as"

```latex
\frac{\partial f(\theta)}{\partial \theta}
= r \left[ f\!\left(\theta + \frac{\pi}{4r}\right) - f\!\left(\theta - \frac{\pi}{4r}\right) \right] .
```

(*unnumbered*. With `r = 1/2` this is the familiar `½[f(θ+π/2) − f(θ−π/2)]`.)

> "For the controlled rotation gates, four eigenvalues exist and the exact derivation of a CRX
> gate is given as"

```latex
\frac{\partial f(\theta)}{\partial \theta}
= \frac{\sqrt{2}+1}{4\sqrt{2}}
  \left[ f\!\left(\theta + \frac{\pi}{4}\right) - f\!\left(\theta - \frac{\pi}{4}\right) \right]
- \frac{\sqrt{2}-1}{4\sqrt{2}}
  \left[ f\!\left(\theta + \frac{3\pi}{4}\right) - f\!\left(\theta - \frac{3\pi}{4}\right) \right] .
\tag{4}
```

(Transcribed from the typeset PDF, p. 6 — the text dump mangles the fractions.)

> "It should be noted that the parameter shift rule requires to execute the quantum circuit 2,
> resp. 4 times. Thus, optimization using the parameter shift rule can become compute intensive
> but the parameter shift rule is appropriate when analytic gradients are feasible and
> simulation/runtime is not prohibitive."

Note the coefficients: `(√2+1)/(4√2) = 0.426777`, `(√2−1)/(4√2) = 0.073223`; they sum to 1/2, as
they must. **The paper describes the parameter-shift rule but does not use it for its results** —
see §4.2 and PTQNN §II.C ("our implementation is solely based on stochastic optimization").
It is used only for the "gd" baseline column of Table I, which the text says uses
*"a plain vanilla gradient descent approach"* with *"autodiff"*.

---

## 5. §IV.A — noise analysis of the quantum perceptron

### 5.1 Setup and Eq (5)

> "To get a better understanding of the stability of the perceptron under gate noise, especially
> amplitude damping, we assume a simple perceptron realizing a xor-decision (see equation 2).
> This perceptron takes two input values of $|q_0\rangle$ and $|q_1\rangle$ and the output qubit
> $|q_2\rangle$ is realized by two CRX-gates performing a controlled $\pi/2$-rotation. Figure 5
> summarizes the circuit. The qubits are parametrized as"

```latex
|q_i\rangle = \alpha_i |0\rangle + (1 - \alpha_i) |1\rangle
\tag{5}
```

⚠ Two problems with this passage:

* **Eq (5) is not normalised.** `|α|² + |1−α|² = 1` only at `α ∈ {0, 1}`. It is presumably meant
  as `√α|0⟩ + √(1−α)|1⟩` (which would make the Fig 6/Fig 7 axes `q₀, q₁ ∈ [0,1]` read as
  *probabilities*, consistent with §III). **Treat Eq (5) as a typo.**
* **The text says "a controlled π/2-rotation" but Fig 5 draws `RX(π)`.** `π` is the correct
  value for the XOR of Eq (2) (`ω = 1` ⇒ `θ = 2·asin(1) = π`, and two active inputs give `2π`
  ⇒ `P = 0`). "π/2" is presumably the half-angle appearing in the matrix. **Trust the figure:
  `θ = π` per CRX.**

### 5.2 **Fig 5** circuit (vector figure, p. 7) — the noise-insertion points

3 wires: `|q₀⟩`, `|q₁⟩` (inputs), `|q₂⟩` (output). Gate sequence, left to right:

1. `CRX(π)`: control `q₀`, target `q₂` (drawn as an `RX(π)` box on `q₂` with a control dot
   on `q₀`).
2. **`L₁`** — a red starburst spanning **all three wires**, i.e. a 3-qubit noise channel.
3. `CRX(π)`: control `q₁`, target `q₂`.
4. **`L₂`** — a second red starburst spanning all three wires.
5. A measurement meter on `q₂`.

Caption (verbatim): *"Setup for measuring the impact of noisy gates for the implementation of an
artificial neuron realizing a xor-decision."*

So `L₁` and `L₂` are the *same* channel applied after the first and after the second CRX
respectively; they are not different noise *types*. Text (verbatim): *"We assume, that this
effect can happen after each CRX-gate in the circuit. Thus, noise after the first gate will
accumulate to the noise of the second gate."*

### 5.3 The Kraus model — Eqs (6), (7), (8)

> "To model energy relaxation effects in the three-qubit system, we employ an amplitude damping
> channel [56]. The standard single-qubit amplitude damping process can be characterized by a set
> of Kraus operators $\{E_0, E_1\}$, defined as:"

```latex
E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \qquad
E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix},
\tag{6}
```

> "where $\gamma \in [0, 1]$ represents the damping probability (or decay rate) of a single qubit
> from the excited state $|1\rangle$ to the ground state $|0\rangle$."

> "Now, the noise mechanism during each gate operation is simulated. The joint noise action is
> represented using $2^3$ Kraus opreators"

```latex
N_{i,j,k} = E_i \otimes E_j \otimes E_k, \qquad i, j, k \in \{0, 1\}
\tag{7}
```

> "Under this noise model, the transformation of the three-qubit density matrix $\rho$ is given
> by the quantum channel:"

```latex
\mathcal{N}(\rho) = \sum_{i,j,k \in \{0,1\}} N_{i,j,k}\, \rho\, N_{i,j,k}^{\dagger}
\tag{8}
```

So: **independent single-qubit amplitude damping with the same `γ` on all three qubits,
applied twice (once after each CRX)**. `Σ N† N = I` follows from `E₀†E₀ + E₁†E₁ = I` on each
factor.

### 5.4 Results — Fig 6 and Fig 7

> "To measure the stability, the trace distance between the unitary of the noisy circuit and the
> noiseless circuit is used. Figure 7 shows the trace distance for
> $\gamma \in \{0, 0.1, 0.2, 0.3\}$. Since amplitude decay is irrelevant for $|q_0\rangle$ and
> $|q_1\rangle$ at zero, the trace distance is very small here, but when both qubits are close to
> 1, the noise has an increased impact. Since the amount of gates scale linear with the feature
> dimension it is obvious that for larger systems, the noise will have a major impact on the
> quality of the results. Thus, error correction needs to be considered when realizing the
> circuits in today's available hardware."

⚠ "the trace distance between the **unitary** of the noisy circuit and the noiseless circuit" —
the noisy circuit has no unitary. What is actually plotted must be the trace distance between
the two *output states* (or, less likely, between Choi matrices). Not resolved by the paper.

**Fig 6** (raster, p. 7): 3-D surface. x = `q₀` (0 … 1), y = `q₁` (0 … 1), z = `q₂` (0 … 1,
ticks 0, 0.5, 1), colour bar 0 … 1 (jet). No gate noise. Caption (verbatim): *"Probability of
activating $|q_2\rangle$ (Fig 5) for varying $|q_0\rangle$ and $|q_1\rangle$. Thresholding the
probabilities leads to the desired xor pattern for non-binary inputs (no noise on gates)."*
The thresholded pattern is stated in the body text as

```
|q0⟩ ; |q1⟩ ∈ [0…0.5 ; 0…0.5]  → 0
|q0⟩ ; |q1⟩ ∈ [0.5…1 ; 0…0.5]  → 1
|q0⟩ ; |q1⟩ ∈ [0…0.5 ; 0.5…1]  → 1
|q0⟩ ; |q1⟩ ∈ [0.5…1 ; 0.5…1]  → 0
```

**Fig 7** (raster, p. 7): 3-D surface, z axis labelled `Trace Distance` with ticks
`0, 0.1, …, 0.9`; x = `q₀` (0 … 1), y = `q₁` (0 … 1); jet colour bar with ticks `0 … 0.7`.
Four sheets are drawn, one per `γ`. Read-off at the far corner `q₀ = q₁ = 1`, where each sheet
is maximal (calibration: 450 px per unit z, sheet-vs-floor pixel differences at the corner):

| γ | max trace distance at `q₀ = q₁ = 1` |
|---|---|
| 0.0 | 0 (flat sheet, exactly on the floor) |
| 0.1 | **≈ 0.40** |
| 0.2 | **≈ 0.65** |
| 0.3 | **≈ 0.80** |

Reading error **±0.03**. All four sheets meet at 0 at the corner `q₀ = q₁ = 0`, as the text
says. The `γ` ↦ sheet assignment is by ordering (larger γ ⇒ larger distance); the figure has no
legend. **[INFERENCE, but unambiguous given the text.]**

---

## 6. §IV.B — shallow networks, the Iris binarisation protocol, Table I

### 6.1 The circuit (Fig 8) — verbatim

> "Figure 8 demonstrates the implementation of a shallow neural network on a quantum device based
> on our earlier defined stochastic perceptrons. The shown model consists of 4 qubits as input
> which are connected to three hidden artificial neurons by using CRX-Gates. The hidden neurons
> are three additional qubits. The RX-Gate itself is the bias. Afterwards, the three qubits
> representing the neural activity are connected to two qubits acting as output neurons. The
> (C)RX-Gates contain with the rotation angles $\omega_i$ the learnable parameters."

Caption (verbatim): *"Implementation of a shallow neural network on a quantum device. This model
consists of 4 qubits as input, three hidden artificial neurons and two output neurons. The
(C)RX-Gates contain the learnable parameters."*

So the qubit count is `#inputs + #hidden + #outputs` (4 + 3 + 2 = 9 here); one qubit per output
*class*.

### 6.2 The binarisation protocol (verbatim, p. 8)

> "To model a classification task using a quantum circuit, first the data is encoded as a
> higher-dimensional binary vector. Taking the iris dataset as a toy example, it consists of 4
> dimensional data encoding sepal length, sepal width, petal length and petal width. **After
> separating training and test data**, a kMeans clustering on each dimension with $k = 3$ is used
> **on the training data**. Thus, every datapoint can be encoded in a
> $4 \times 3 = 12-$dimensional binary vector which contains exactly 4 non-zero entries. **For
> the given cluster centers, the same can be done with the test data.** Thus, a binary encoding
> is used to represent the datasets." (emphasis ours)

This is strictly more informative than PTQNN's copy of the same paragraph: the split happens
**before** clustering, kMeans is fit on **train only**, and test points are assigned to the
train-fitted centres. Use this version.

> "Our choice of preprocessing via classical clustering and binarization simplifies the
> classification task while encoding structure into the input. Our intention is to provide a
> uniform, hardware-friendly binary encoding across heterogeneous datasets and to stay within the
> constraints of current qubit counts and connectivity."

Datasets (verbatim):

> "for the experiments the classical iris, wine, zoo and mnist datasets were used. The datasets
> present multi-class classification tasks, with three categories for the iris and wine datasets
> and seven categories for the zoo dataset. The datasets are all available at the UCI repository
> [22]. Additionally, we performed experiments on a subset of the well-known MNIST dataset [18]
> for digit classification. Here, we restricted the model to five classes (the digits
> $0 \dots 4$) and used 4500 (random) samples for training and 500 for testing."

### 6.3 **Table I** (reproduced in full, verbatim)

Caption (verbatim): *"Test datasets overview and performance. The column gd indicates the
performance when using a plain vanilla gradient descent approach for optimization. Please note
the high variance of the results, due to local minima."*

| Dataset | Inp. | Perc. | qubits | Classes | Acc (gd) | Acc ours |
|---------|------|-------|--------|---------|----------|----------|
| Iris    | 12   | 4     | 19     | 3       | 84 % (± 21) | **95 % (± 5)** |
| Zoo     | 6    | 5     | 18     | 7       | 72 % (± 22) | **87 % (± 5)** |
| wine    | 12   | 3     | 18     | 3       | 75 % (± 5)  | **95 % (± 4)** |
| mnist5  | 20   | 8     | 38     | 5       | 75 % (± 28) | **99 % (± 5)** |

Arithmetic check `qubits = Inp. + Perc. + Classes`: Iris 12+4+3 = 19 ✓, Zoo 6+5+7 = 18 ✓,
wine 12+3+3 = 18 ✓, **mnist5 20+8+5 = 33 ≠ 38** ✗. The mnist5 row needs 5 extra qubits — most
likely a second hidden layer (8 + 5), but this is never stated. Flag as unresolved.

Accompanying text (verbatim):

> "Table I summarizes the four example datasets and the overall performance. It contains the
> used qubits to represent the problem as quantum code, the amount of target classes and the
> achieved accuracy with gradient descent using autodiff (gd) and our optimized quantum circuits
> (ours). Please note the large variance of gradient descent, which results in a low average
> score due to local minima. Thus, when starting from a random seed, the optimizer often
> converges to good results comparable to our approach, but frequently the optimization fails
> completely. […] For this experiment, we did not perform data augmentation, dropout or other
> common methods to increase performance of gradient descent. The combination of gradient descent
> with common measures to improve performance leads to similar results as our model."

**Fig 10** (Iris convergence) caption (verbatim): *"Convergence of the optimization algorithm
during training the iris dataset. Shown is the loss decrease in blue and the accuracy (of the
test data) in red. In this run, the algorithm could perfectly classify all test samples."*
No axis numbers are quoted here; the figure is small and its ticks are not needed by PTQNN.

⚠ The `±` numbers are not defined (std over runs? over folds?), the number of repetitions is not
given, and the train/test split fractions are given **only** for mnist5.

---

## 7. Shots, measurement protocol, and simulation tooling

### 7.1 Shots / measurement

**Neither paper states a shot count anywhere.** The available evidence points to *exact*
state-vector simulation with analytically computed probabilities:

* The loss for the RBM/autoencoder capacity experiments is defined (§IV.E.1, verbatim):
  > "To determine this, **we simulate the quantum circuit**. For a target $T$ we compute
  > $\sum_i (\mathrm{prob}(i) * \mathrm{xor}(\mathrm{state}(i), T)$, with xor denoting the
  > Hadamard distance between the observed variable and the target $T$, multiplied with the
  > observation likelihood. Thus, if the state(i) is equal to $T$ and observed with a probability
  > 1, the score is zero."
  This is a sum over *all* basis states weighted by exact probabilities — not an estimator from
  samples.
* Fig 4's histograms sit exactly on 0.4 / 0.6 / 1.0 with no sampling scatter.
* PTQNN's KL (its Eq 4) is a sum over all 32 outcomes; its Fig 5 model histogram sums to
  0.9995 ± 0.0003 across 32 bins, again with no visible shot noise.
* The word "shots" does not occur in either text dump.

**[INFERENCE: exact probabilities from a full state-vector/density-matrix simulation, no
sampling. Mark any shot count in our code `[ASSUMPTION]`.]**

The noise analysis (§5) is explicitly density-matrix based (Eq 8 acts on `ρ`), so at least that
part uses a 3-qubit density matrix.

### 7.2 Simulator / language

**Not stated in either paper.** No software section, no code-availability statement, no GitHub
URL, no mention of Qiskit, PennyLane, Cirq, MATLAB, PyTorch or NumPy anywhere in either text.
The only software-adjacent citation is reference [58] (Picard, *torch.manual_seed(3407) is all
you need*), cited for the local-minima argument, not for tooling.

**[INFERENCE, high confidence: MATLAB.]** Evidence, all from PTQNN's figures (the SNN figures
are the same house style):

* PTQNN Fig 8's bar fill is exactly `RGB(0, 114, 189)` = MATLAB's default axes colour
  `[0 0.4470 0.7410]`; Fig 5's bars are `RGB(17, 113, 190)` / `RGB(112, 170, 216)`.
* PTQNN Figs 4/5/10/12 are rendered in the exact style of the **MATLAB Support Package for
  Quantum Computing** (`quantumCircuit` … `plot`): wires numbered on the left, grey gate boxes
  labelled `RX` with the numeric angle typeset in small print *underneath*, and composite gates
  drawn as tall multi-port boxes with an ellipsis-truncated name (`Pro…`) and numbered ports.
* SNN Figs 6, 7, 10, 15–19 and PTQNN Figs 3, 6, 7, 9, 13 are MATLAB `surf` / `plot` with the
  default jet and `lines` colormaps and MATLAB tick typography.

That package's `RX` gate is `exp(-iθX/2)`, which is consistent with the `θ = 2·asin(√a)`
convention verified in §3.3.

---

## 8. Other architectures (one line each, for completeness)

PTQNN does not depend on these, but they constrain the neuron model:

* **§IV.C Hopfield** (Fig 11): one recurrent layer, `ω_{i,j} = ω_{j,i}`, `ω_{i,i} = 0` enforced
  by cutting the self-connection (marked with red boxes in the figure). *"In our experiments the
  model typically converges after three to five iterations."* Pattern demo is 9-dimensional
  (3 × 3), two memorised patterns.
* **§IV.D RBM** (Fig 14): bipartite, symmetric weights, weight sharing between encode and
  decode (red connections). Iris demo: 12 input qubits → 2 latent → 12.
* **§IV.E Autoencoder** (Fig 16): same circuit as the RBM with the weight sharing removed.
* **§IV.E.1** capacity study on all 2³ binary codes and on the 5 × 6 dataset of **Eq (9)**:

```latex
D = \begin{pmatrix}
1 & 1 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 & 0 & 1 \\
1 & 1 & 0 & 1 & 1 & 0
\end{pmatrix}
\tag{9}
```

  *"All involved weights are initialized with zero."* Both models need a 3-dimensional latent
  space to reproduce the 2³ code exactly.
* **§IV.F CNN** (Fig 20): convolution realised by weight sharing + connection cutting
  (Toeplitz structure); bars-and-stripes task on a 3 × 3 patch, *"only 8 positive patterns and
  $2^9 - 8$ negative patterns"*, solved with *"the Kiefer–Wolfowitz algorithm and preferencial
  sampling [20] … when using majority voting."*
* **§IV.G Grover/GenAI** (Figs 22–25): the frozen trained network plus its inverse is used as a
  Grover oracle, preceded by a Hadamard superposition and followed by the diffusion operator.

---

## 9. What this paper does not specify (relevant to PTQNN)

1. **No optimiser hyper-parameters** — `aₙ`, `cₙ`, `k`, `T₀`, cooling schedule, perturbation
   distribution, iteration budget, stopping criterion, seeds, number of repeats behind the
   `±` figures in Table I. (Only the RBM/AE experiment's zero initialisation is stated.)
2. **No shot count / measurement protocol** (see §7.1 — exact simulation is inferred).
3. **No simulator, language, or code release** (see §7.2 — MATLAB is inferred).
4. The rotation convention `RX(θ) = exp(-iθX/2)` is never written; it is only recoverable from
   the numbers in Fig 4 (§3.3).
5. Eq (5) is not a normalised state (§5.1).
6. "trace distance between the unitary of the noisy circuit and the noiseless circuit" is not
   a well-defined quantity as written (§5.4).
7. The text/figure conflict on the Fig 4 weight ordering (§3.4) and on `π` vs `π/2` in Fig 5
   (§5.1).
8. Table I's mnist5 qubit count is inconsistent with the other three rows (§6.3); the network
   depth for mnist5 is not given.
9. How the Zoo dataset's 16 features become `Inp. = 6` — the `k = 3` per-dimension recipe would
   give a multiple of 3 (e.g. 48), not 6. Unexplained.
10. Train/test split fractions for Iris, wine and zoo (given only for mnist5).
11. Whether the noise channel of §5 is applied in any of the *training* experiments (it appears
    to be a standalone stability study; nothing else in the paper is noisy).
