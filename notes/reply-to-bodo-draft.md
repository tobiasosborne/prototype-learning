Dear Bodo,

good questions - several land where the report says too little. Figure numbers are ours.

Fig 10: the prototype wins under every preset (KL 0.0007 on H2-1 to 0.0062 on Garnet, classic 0.0014-0.0084), but the right panel is the point: noise displaces both equally (KL(clean||noisy) 0.0021 vs 0.0022, TV 0.005-0.039). Same gate counts, so the prototype's edge is its noiseless fit, not robustness.

Figs 11 and 12: one sentence each, agreed. Fig 11 only shows linear response (Fez scaled by s: KL ~ s^1.7, TV ~ s^0.95). Fig 12 belongs in the shots section: the bias of a KL from N shots, 31.8/N, which sets the shot budget and explains why sampled training stalls.

Fig 13: yes, with one caveat. Noise in the training loop recovers the noisy KL by x158 (prototype) and x361 (classic), but those parameters are worse noiselessly: the optimiser folds the channel into the angles rather than finding robust ones. Calibration per machine, then, not robustness.

Fig 14: not quite - all seven machines play perfectly (110/110 pipes, 20 held-out games), 1.5-4 % of frames flipping. Fez is not the most tolerant, just the one I stress-tested (fullest calibration); the right panel scales its errors by s: 60 % perfect games at s=2, none at s=4 - two-fold headroom.

Fig 16: indicative only - 8 ECG windows instead of 898, 16 Iris trajectories instead of 2000; the Iris variance is sampling, not physics. The asymmetry is real, though: ECG reads one expectation value, Iris a conditional distribution over 4096 strings, directly exposed to probability leaking out of its 20-code support. P1 > C1 is not readable off it either - the prototype cells even gain 3.8 dB under Fez, a small-sample artefact; Fig 10 carries that claim.

Figs 17/18: the shading in 17 is routing - hatched = gates spent on SWAPs, three each: 82 % of bit-count's 327 two-qubit gates on heavy-hex, against 60 all-to-all. In 18 the stacked bars are -log F by error source. Hardware analysis, yes - and that is the point: it settles Quantinuum vs Garnet. Flappy reaches F ~ 0.92 (H2-1), 0.83 (Forte), 0.63 (Fez), 0.24 (Garnet); bit-count 0.81/0.64/0.48/0.008. Garnet is the cheapest job we costed and buys nothing: ions first.

Shots: report, not optimise, with two results: Wald's sequential rule wins every game at 2191 shots per frame against 10,000 fixed (factor 4.6); and the plug-in bias above, which shows the loss, not the optimiser, limits sampled training.

Costs: nothing was paid, nothing ran on hardware. Every number is a simulator run under vendor calibration data; the cost table is what their price lists imply. Say an agent did it, yes, but precisely: no quantum computer was used - the text should read "simulators matched to calibration data".

Iris: your policy is implemented exactly - codes with two labels can be dropped as well as merged; we report both, and with dropping we quote accuracy on the filtered test set and on all test flowers, since the filter reads the test labels. The 20-seed sweep turned up a trap in the read-out: rounding to the nearest label needs one overall scale (the weights 1:4:9 are fixed only up to a constant), and fitted on the evaluated set it makes a flower's label depend on the other flowers. On one seed, dropping three test codes moved it by a class step and accuracy from 94 % to 39 %. Fitted once on the training set it is stable: over 20 seeds both networks reach 94 % on all test flowers (prototype 94.3 +- 2.8, classic 93.9 +- 2.7; 96.4 % on the filtered set with dropping) - below your 98 %, and no accuracy edge either way.

Iterative scheme: the KL ~ 0.15 stall was our measurement, not the scheme. We graded sample-wise training with the superposition observable; the prototype block does not restore the input register, so P(x, out=1) is not 2^-n P(out=1|x). Graded per input, with one elementary KW step per iteration rather than a 36-coordinate sweep, it converges - consistent with your 1000 times more iterations. With one KW step (one coordinate, one random input, four objective evaluations) per iteration, four seeds of six reach a per-input KL below 1e-3 after 12k, 23k, 62k and 73k iterations (the paper's endpoint is about 1e-3); a fifth gets to 1.4e-3, one plateaus at 0.02. The superposition KL of those same networks never drops below 0.035 and ends between 0.07 and 0.53, which is exactly the old 'stall'. What remains assumed is only the unit of your x-axis; if one plotted iteration is a sweep over all weights, our 1e5 steps are your 120.

t-SNE: our mapping already used the inverse pattern cos^2(phi/2). The picture is replaced by a covering cost of the training codes by the decoded prototype patterns, against a random null and the best 4-subset, plus an encoding-free score from the neurons' actual firing on each training code (coverage: does every code have a neuron that responds; selectivity: does each neuron respond to some codes and not others). We fixed the pass criterion before running 20 seeds under both policies. Result: selectivity is real and large - the prototype layer's median is 18 standard deviations above its random-init null, the classic layer's 2 (p < 1e-5) - but coverage goes the other way (0.6 vs 2.2: classic neurons fire on more of the data), and the decoded angle patterns cover the training codes no better than random ones, in either the direct or your inverse convention. So trained prototype neurons are specialised detectors but do not tile the data; by our criterion Fig 9 does not reproduce, and the report says so.

ECG: on your recording, first 1800 samples (the Fig 13 window), same protocol as before (8 restarts per cell, best by training error), PSNR in dB for 1/2/3 hidden neurons: prototype 36.8 / 37.5 / 36.9 against your 32.9 / 40.0 / 40.0; classic 34.0 / 36.3 / 38.0 against your 32.7 / 38.2 / 39.7. So the one-neuron advantage is there and larger than yours, the two- and three-neuron prototype cells fall 2.5 to 3 dB short, and the order at three neurons flips. The full 7700-sample trace gives the same shape: prototype 37.8 / 37.8 / 36.3, classic 34.9 / 37.7 / 38.2 - so it is not the window. Its input PSNR is 27.5 dB and the noise std 0.035, against our 0.02 (32 dB in) - so Table I's 32.9 dB for one neuron is the +5 dB gain we measured, from a lower start.

Overleaf: a drop-in folder is ready - preamble snippet, main-text subsection, compiling appendix. Your pasted appendix uses macros your preamble lacks (\KL, \Pout, \cref, booktabs, siunitx), so it will not compile; the drop-in fixes that. One query: l. 387 has 4,500,000 in ten days, the submission 450,000 in six - which? On authorship, an acknowledgement is entirely fine by me.

Shall we talk next week? [SLOTS] both suit me.

Best wishes,
Tobias
