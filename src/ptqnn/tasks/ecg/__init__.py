"""ECG denoising ``[PTQNN §III.D, Fig 12, 13, Table I]``.

"five consecutive (noisy) samples at time point t, ``[s(t−2) … s(t+2)]`` are used to predict the
value ``s(t)``. As time-series data an ECG-signal use case has been used. The goal is to denoise
a signal overlaid with stochastic noise."

:mod:`~ptqnn.tasks.ecg.data` builds (or downloads) the signal and cuts the sliding windows,
:mod:`~ptqnn.tasks.ecg.model` is the encoding, forward pass and the PSNR/SSIM metrics of
Eq 6–8, :mod:`~ptqnn.tasks.ecg.train` runs KW–SA and reproduces Table I.
"""
