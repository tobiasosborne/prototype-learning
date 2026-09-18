# Agentic reproducibility report — drop-in files
**Upload flat, next to `main.tex`:** `ptqnn-agentic-preamble.tex`, `sec-reproducibility.tex`,
`app-agentic.tex`, and the two PNGs marked NEW in `figures-needed.txt`
(`fig_bitcount_convergence.png`, `fig_flappy_shots_programme.png`; the other 11 are already there).
Do **not** upload `standalone.tex` / `standalone.pdf` / `refs.bib` — they are only the local test build.
**Three edits to `main.tex`:** (1) after `\usepackage{adjustbox}` (l. 15) insert
`\input{ptqnn-agentic-preamble}`; (2) replace l. 446–457 — the `figure*` with `PTAgentComp.png`
**and** the subsection "Reproducibility Report Using Agentic AI" — with `\input{sec-reproducibility}`
(the scoreboard is now a real LaTeX table, so `PTAgentComp.png` is unused); (3) replace l. 474 to
just before `\end{document}` (the whole pasted appendix) with `\appendix` then `\input{app-agentic}`.
Append the last three entries of `refs.bib` (`wald1945`, `moody2001mitbih`, `gleinig2021sparse`) to
`sample.bib`; the other two keys are already in it. The new text does not name the agent systems —
re-insert "fable, codex, opencode, DeepSeek V4" if you want them named. Optional: adding `floatfix`
to the `revtex4-2` class options silences two harmless "float is stuck" placement warnings.
**Two things only you can decide.**
*Wording.* Abstract and contribution 4 say "solely based on the .pdf file of this submission". The
reimplementation also used the companion paper (arXiv:2602.22241) for the optimiser and the input
angle encoding, plus the public Iris/MIT-BIH data; and the noise, hardware and shot-noise work is an
extension, not a reproduction. Suggested contribution 4 (it also fills `\footnote{add link here}`):
"We reproduced our results with an agentic AI clean-room reimplementation, given only the text and
figures of this submission, the companion paper~\cite{rosenhahn2026stochasticneuralnetworksquantum}
and the public Iris and MIT-BIH data — no source code. It is available at
\url{https://github.com/tobiasosborne/prototype-learning}." The same wording fixes the abstract.
*Flappy score.* l. 387 says 4.500.000 in 10 days; the appendix you pasted quoted 450,000 in six days
from the submission. The new appendix quotes neither — which is right?
