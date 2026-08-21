# literature/ — local only

Everything in this directory except this file is gitignored (copyright). The repository's ground
truth is the pair of distillations in `docs/physics/`, which quote every sentence, equation and
figure reading the code pins to; the PDFs are needed only to re-check a pin.

* `pdf/QPrototypeLearning-5.pdf` — Rosenhahn & Hirche, *Prototype Learning for Quantum Neural
  Networks* (PTQNN). Pins `[PTQNN §/Fig/Eq]`.
* `pdf/2602.22241.pdf` — Rosenhahn, Osborne & Hirche, *Stochastic Neural Networks for Quantum
  Devices*, arXiv:2602.22241 (SNN). Pins `[SNN §/Fig/Table]`. Source of the KW–SA optimiser, the
  angle ↔ probability convention (`θ = 2·asin(√s)`, Fig 4) and the Iris binarisation protocol.
* `*.txt` — text extractions of the two PDFs.
* `figs/` — crops of PTQNN Fig 4–13 used for reading off bar heights and landmarks
  (`docs/physics/rosenhahn_hirche_ptqnn.md` records each reading with its error).
* `snn-figs/` — crops of SNN figures.

Re-obtaining: arXiv:2602.22241 is public; the PTQNN manuscript is the authors' draft v5 and is
not redistributed here.
