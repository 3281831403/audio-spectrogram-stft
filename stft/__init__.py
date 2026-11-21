"""Audio spectrogram and STFT package.

Public API for the exercise lives in :mod:`stft.spectrogram`.
"""

from .spectrogram import (
    StftError,
    frame,
    magnitude_db,
    mel_filterbank,
    rfft,
    spectrogram,
    stft_batch,
    to_mel,
    window,
)

__all__ = [
    "StftError",
    "window",
    "frame",
    "rfft",
    "magnitude_db",
    "spectrogram",
    "mel_filterbank",
    "to_mel",
    "stft_batch",
]
