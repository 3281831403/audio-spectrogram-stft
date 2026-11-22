"""Core STFT and spectrogram helpers for the audio editor.

Audio is represented as a plain ``list[float]`` so the implementation only
needs the Python standard library (``math``/``cmath``) plus pytest for the
tests.  Numeric comparisons in the test suite use an absolute tolerance of
``1e-6``.
"""

from __future__ import annotations

import cmath
import math

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


class StftError(Exception):
    """Raised for invalid STFT parameters, data, or batch input."""


def _positive_int(value) -> bool:
    """True only for genuine positive integers (booleans excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def window(n: int, kind: str) -> list[float]:
    """Return a length-``n`` periodic (DFT-even) window.

    ``kind`` must be one of ``"hann"``, ``"hamming"``, or ``"blackman"``.
    The denominator is ``n`` (periodic correction), rather than ``n - 1``
    (the symmetric-window convention), and the window is endpoint-symmetric
    under periodic extension: with ``w[n] == w[0]``, ``w[k] == w[n-k]``.
    For ``k = 0 .. n-1`` the coefficients are::

        hann:     0.5 - 0.5 * cos(2*pi*k/n)
        hamming:  0.54 - 0.46 * cos(2*pi*k/n)
        blackman: 0.42 - 0.5 * cos(2*pi*k/n) + 0.08 * cos(4*pi*k/n)

    The returned values are exact to within ``1e-6``.  An unknown ``kind`` or
    a non-positive ``n`` raises :class:`StftError`.

    Args:
        n: Number of window coefficients; must be positive.
        kind: Window name, exactly ``"hann"``, ``"hamming"``, or
            ``"blackman"``.

    Returns:
        A list of ``n`` floats.
    """

    if not _positive_int(n):
        raise StftError(f"n must be a positive integer, got {n!r}")
    if kind == "hann":
        return [
            0.5 - 0.5 * math.cos(2.0 * math.pi * k / n) for k in range(n)
        ]
    if kind == "hamming":
        return [
            0.54 - 0.46 * math.cos(2.0 * math.pi * k / n) for k in range(n)
        ]
    if kind == "blackman":
        return [
            0.42
            - 0.5 * math.cos(2.0 * math.pi * k / n)
            + 0.08 * math.cos(4.0 * math.pi * k / n)
            for k in range(n)
        ]
    raise StftError(f"unknown window kind: {kind!r}")


def frame(samples: list[float], n_fft: int, hop: int) -> list[list[float]]:
    """Split ``samples`` into fixed-size, zero-padded frames.

    Every returned frame has exactly ``n_fft`` floats.  For non-empty audio
    with ``len(samples) >= n_fft``, the frame count is::

        1 + (len(samples) - n_fft) // hop

    and frame ``i`` starts at sample ``i * hop``.  For non-empty audio shorter
    than one frame, return one frame padded on the right with zeros; an empty
    input returns ``[]``.  Padding is performed with ``0.0`` and never changes
    the input list.

    ``hop`` must be positive and ``n_fft`` must be positive; otherwise raise
    :class:`StftError` before doing any work.  In particular, ``hop=0`` must
    never enter a loop.

    Args:
        samples: Audio samples as floats.
        n_fft: Frame length (also the transform size).
        hop: Frame advance in samples; must be greater than zero.

    Returns:
        A list of ``n_fft``-long float lists, in time order.
    """

    if not _positive_int(n_fft):
        raise StftError(f"n_fft must be a positive integer, got {n_fft!r}")
    if not _positive_int(hop):
        # hop=0 would otherwise loop forever in the framing loop.
        raise StftError(f"hop must be a positive integer, got {hop!r}")

    total = len(samples)
    if total == 0:
        return []

    # A non-empty signal shorter than one frame yields exactly one padded
    # frame.  Otherwise the count follows 1 + (len - n_fft) // hop: this
    # keeps every full frame instead of dropping the tail and never
    # over-counts the way ``len // hop`` (plus one) would.
    if total < n_fft:
        frame_count = 1
    else:
        frame_count = 1 + (total - n_fft) // hop
    frames: list[list[float]] = []
    for i in range(frame_count):
        start = i * hop
        end = start + n_fft
        if end <= total:
            frames.append(list(samples[start:end]))
        else:
            # Right-pad the final partial frame with zeros; copy only the
            # available samples so the caller's list is never mutated.
            tail = list(samples[start:total])
            tail.extend([0.0] * (n_fft - len(tail)))
            frames.append(tail)
    return frames


def rfft(frame: list[float]) -> list[complex]:
    """Return the one-sided, amplitude-scaled DFT of one real frame.

    The transform length is ``n = len(frame)``.  Compute the direct DFT bins
    ``k = 0 .. n // 2`` with a ``1/n`` normalization::

        X[k] = (1/n) * sum(frame[t] * exp(-2j*pi*k*t/n) for t in range(n))

    and return exactly ``n // 2 + 1`` complex values.  To preserve the
    amplitude of a real sinusoid in the one-sided spectrum, multiply all
    non-DC bins by two: for even ``n``, bins ``1 .. n//2 - 1``; for odd ``n``,
    bins ``1 .. n//2``.  The DC bin and the even-length Nyquist bin are not
    doubled.  Consequently ``cos(2*pi*m*t/n)`` has magnitude ``1.0`` at bin
    ``m`` (within ``1e-6``) for ``0 < m < n//2``.

    An empty frame raises :class:`StftError`.

    Args:
        frame: A real-valued frame; typically produced by :func:`frame`.

    Returns:
        ``n // 2 + 1`` complex one-sided spectrum values.
    """

    n = len(frame)
    if n == 0:
        raise StftError("rfft requires a non-empty frame")

    spectrum: list[complex] = []
    for k in range(n // 2 + 1):
        total = 0.0j
        for t, sample in enumerate(frame):
            total += sample * cmath.exp(-2.0j * math.pi * k * t / n)
        value = total / n
        # One-sided amplitude correction: energy in the discarded conjugate
        # half is folded back in.  DC (and even-length Nyquist) stay as-is.
        if k != 0 and not (n % 2 == 0 and k == n // 2):
            value *= 2.0
        spectrum.append(value)
    return spectrum


def magnitude_db(spec: list[complex], floor_db: float) -> list[float]:
    """Convert complex amplitudes to dBFS and clamp them at ``floor_db``.

    Each value is ``20 * log10(abs(value))``, not ``10 * log10(...)``.
    The result for every bin is clamped to at least ``floor_db``; a zero bin
    maps exactly to ``floor_db`` instead of producing ``-inf`` or ``NaN``.
    ``floor_db`` must be finite.  Results agree to within ``1e-6``.

    Args:
        spec: Complex one-sided spectrum values.
        floor_db: Finite lower bound in dB.

    Returns:
        A list of finite floats, one per input bin.
    """

    if not math.isfinite(floor_db):
        raise StftError(f"floor_db must be finite, got {floor_db!r}")

    result: list[float] = []
    for value in spec:
        magnitude = abs(value)
        if magnitude <= 0.0:
            result.append(floor_db)
            continue
        db = 20.0 * math.log10(magnitude)
        # Clamp so log(0)-style values can never leak -inf into the view.
        result.append(db if db >= floor_db else floor_db)
    return result


def spectrogram(
    samples: list[float],
    n_fft: int,
    hop: int,
    kind: str,
) -> list[list[float]]:
    """Compute a magnitude spectrogram with shape ``time x frequency``.

    The audio is split with :func:`frame`, each frame is multiplied by the
    periodic :func:`window` named by ``kind``, and :func:`rfft` followed by
    :func:`magnitude_db` produces ``n_fft // 2 + 1`` frequency bins per frame.
    The dB floor is ``-120.0``.  An empty input returns ``[]``.

    ``n_fft`` and ``hop`` must be positive and ``kind`` must be known;
    otherwise raise :class:`StftError`.  In particular, ``hop=0`` must raise
    immediately rather than loop forever.

    Args:
        samples: Audio samples as floats.
        n_fft: Frame/transform length; must be positive.
        hop: Frame advance; must be greater than zero.
        kind: Window name accepted by :func:`window`.

    Returns:
        Rows in time order; each row has ``n_fft // 2 + 1`` dB values.
    """

    # Validate everything up front; window() also rejects an unknown kind.
    win = window(n_fft, kind)
    if not _positive_int(hop):
        raise StftError(f"hop must be a positive integer, got {hop!r}")

    if len(samples) == 0:
        return []

    db_floor = -120.0
    rows: list[list[float]] = []
    for one_frame in frame(samples, n_fft, hop):
        # A fresh windowed buffer per frame keeps frames isolated.
        windowed = [sample * coeff for sample, coeff in zip(one_frame, win)]
        rows.append(magnitude_db(rfft(windowed), db_floor))
    return rows


def _hz_to_mel(freq: float) -> float:
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> list[list[float]]:
    """Build a peak-normalized triangular mel filterbank.

    Frequencies use the mel mapping ``m(f) = 2595 * log10(1 + f/700)``.
    Place ``n_mels + 2`` edge frequencies uniformly in mel space from
    ``fmin`` to ``min(fmax, sample_rate / 2)``; thus the bank is clamped to
    Nyquist and never indexes past the ``n_fft // 2 + 1`` FFT bins.  Filter
    ``m`` rises linearly from zero at edge ``m`` to one at edge ``m + 1`` and
    falls linearly to zero at edge ``m + 2``.

    Adjacent triangles share only a zero-valued edge, so no bin receives more
    than one unit of summed filter gain: for every FFT bin ``b``,
    ``sum(bank[m][b] for m) <= 1 + 1e-6``.  Each row has exactly
    ``n_fft // 2 + 1`` non-negative weights.  A degenerate triangle that
    contains no FFT bin is returned as an all-zero row rather than raising.

    ``n_mels`` and ``n_fft`` must be positive, ``sample_rate`` must be
    positive, and ``0 <= fmin < min(fmax, sample_rate/2)``; otherwise raise
    :class:`StftError`.

    Args:
        n_mels: Number of triangular filters (rows).
        n_fft: FFT size used by the spectrogram.
        sample_rate: Sampling rate in Hz.
        fmin: Lowest filter edge in Hz.
        fmax: Requested highest filter edge in Hz; clamped to Nyquist.

    Returns:
        ``n_mels`` rows, each with ``n_fft // 2 + 1`` float weights.
    """

    if not _positive_int(n_mels):
        raise StftError(f"n_mels must be a positive integer, got {n_mels!r}")
    if not _positive_int(n_fft):
        raise StftError(f"n_fft must be a positive integer, got {n_fft!r}")
    if not _positive_int(sample_rate):
        raise StftError(
            f"sample_rate must be a positive integer, got {sample_rate!r}"
        )
    if not isinstance(fmin, (int, float)) or not math.isfinite(fmin):
        raise StftError(f"fmin must be finite, got {fmin!r}")
    if not isinstance(fmax, (int, float)) or not math.isfinite(fmax):
        raise StftError(f"fmax must be finite, got {fmax!r}")

    nyquist = sample_rate / 2.0
    upper = min(fmax, nyquist)
    if fmin < 0.0 or not (fmin < upper):
        raise StftError(
            "require 0 <= fmin < min(fmax, sample_rate/2); "
            f"got fmin={fmin!r}, fmax={fmax!r}, nyquist={nyquist!r}"
        )

    n_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(upper)
    # n_mels + 2 edges, uniformly spaced in mel space.
    edges_hz = [
        _mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1))
        for i in range(n_mels + 2)
    ]

    bank: list[list[float]] = []
    for m in range(n_mels):
        left, center, right = edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]
        row = [0.0] * n_bins
        for b in range(n_bins):
            # FFT bin center frequency in Hz; b never exceeds Nyquist because
            # the highest edge was clamped to sample_rate / 2.
            freq = b * sample_rate / n_fft
            if left <= freq <= center and center > left:
                weight = (freq - left) / (center - left)
            elif center < freq <= right and right > center:
                weight = (right - freq) / (right - center)
            else:
                weight = 0.0
            # Negative zero would break the non-negative weight contract.
            row[b] = max(0.0, weight)
        bank.append(row)
    return bank


def to_mel(spec: list[list[float]], bank: list[list[float]]) -> list[list[float]]:
    """Project a time-major spectrogram onto a mel filterbank.

    ``spec`` has shape ``time x frequency`` (``T x F``) and ``bank`` has shape
    ``n_mels x F`` (``M x F``).  The result is ``M x T``: row count equals
    ``n_mels`` and column count equals the number of time frames.  Entry
    ``result[m][t]`` is ``sum(bank[m][f] * spec[t][f] for f in range(F))``,
    a plain matrix multiplication with the transposed spectrogram.

    A dimension mismatch raises :class:`StftError`.  Empty inputs are valid:
    an empty spectrogram produces ``n_mels`` empty rows.

    Args:
        spec: Time-major magnitude/power spectrogram.
        bank: Mel filterbank with one row per mel band.

    Returns:
        A mel spectrogram with ``len(bank)`` rows and ``len(spec)`` columns.
    """

    n_mels = len(bank)
    n_frames = len(spec)
    n_bins = len(bank[0]) if n_mels > 0 else 0

    for row in bank:
        if len(row) != n_bins:
            raise StftError("ragged mel filterbank: rows have differing widths")
    for frame_row in spec:
        if len(frame_row) != n_bins:
            raise StftError(
                f"spectrogram width {len(frame_row)} does not match "
                f"filterbank width {n_bins}"
            )

    result: list[list[float]] = []
    for m in range(n_mels):
        weights = bank[m]
        mel_row: list[float] = []
        for t in range(n_frames):
            frame_row = spec[t]
            mel_row.append(
                sum(weights[f] * frame_row[f] for f in range(n_bins))
            )
        result.append(mel_row)
    return result


def stft_batch(
    tracks: list[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    """Compute spectrograms for several tracks in one call.

    ``tracks`` is a list of mappings.  Each mapping must contain ``"id"`` (a
    hashable track identifier) and ``"samples"`` (a ``list[float]``).
    ``config`` must contain positive ``"n_fft"`` and ``"hop"`` integers and a
    ``"kind"`` accepted by :func:`window`.

    The return value is::

        {
            "order": [track_id, ...],          # exactly the input order
            "spectrograms": [spec, ...],       # spectrograms[i] belongs to order[i]
        }

    The whole batch is validated before any result is built.  If any track or
    the config is invalid, raise :class:`StftError` and return nothing; a
    failed batch must not expose a partial result or leave caller-owned data
    modified.  Every track is computed with its own frames/window/spectrum
    buffers, so no spectral data is shared or overwritten between tracks.
    An empty ``tracks`` list returns empty ``order`` and ``spectrograms``.

    Args:
        tracks: Track mappings with ``"id"`` and ``"samples"``.
        config: Batch parameters ``"n_fft"``, ``"hop"``, and ``"kind"``.

    Returns:
        A dict with ordered ids and their corresponding spectrograms.
    """

    if not isinstance(config, dict):
        raise StftError(f"config must be a mapping, got {type(config).__name__}")
    n_fft = config.get("n_fft")
    hop = config.get("hop")
    kind = config.get("kind")
    if not _positive_int(n_fft):
        raise StftError(
            f"config n_fft must be a positive integer, got {n_fft!r}"
        )
    if not _positive_int(hop):
        raise StftError(f"config hop must be a positive integer, got {hop!r}")
    # Reject unknown window kinds before touching any track data.
    window(n_fft, kind)

    if not isinstance(tracks, list):
        raise StftError(
            f"tracks must be a list, got {type(tracks).__name__}"
        )

    validated: list[tuple[object, list[float]]] = []
    for index, track in enumerate(tracks):
        if not isinstance(track, dict):
            raise StftError(f"track {index} must be a mapping")
        if "id" not in track:
            raise StftError(f"track {index} is missing 'id'")
        if "samples" not in track:
            raise StftError(f"track {index} is missing 'samples'")
        track_id = track["id"]
        try:
            hash(track_id)
        except TypeError as exc:
            raise StftError(
                f"track {index} id must be hashable"
            ) from exc
        samples = track["samples"]
        if not isinstance(samples, list):
            raise StftError(
                f"track {index} samples must be a list, got "
                f"{type(samples).__name__}"
            )
        validated.append((track_id, samples))

    # All inputs are valid; only now do we build results.  Each spectrogram()
    # call allocates its own frames/window/spectrum buffers, so tracks can
    # never share or overwrite each other's spectral data.
    order: list[object] = []
    spectrograms: list[list[list[float]]] = []
    for track_id, samples in validated:
        order.append(track_id)
        spectrograms.append(spectrogram(samples, n_fft, hop, kind))

    return {"order": order, "spectrograms": spectrograms}
