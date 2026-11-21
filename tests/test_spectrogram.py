"""Red tests for the audio spectrogram/STFT exercise.

Every test exercises one or more public stubs.  The initial repository must
report exactly 15 failures, all from ``NotImplementedError``; no collection,
import, or syntax errors are allowed.
"""

import math

import pytest

from stft.spectrogram import (
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

TOL = 1e-6


def _assert_vector_close(actual, expected, tol=TOL):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert abs(got - want) <= tol


def _assert_matrix_close(actual, expected, tol=TOL):
    assert len(actual) == len(expected)
    for got_row, want_row in zip(actual, expected):
        _assert_vector_close(got_row, want_row, tol)


def test_window_uses_periodic_correction():
    n = 8
    for kind in ("hann", "hamming", "blackman"):
        got = window(n, kind)
        expected = []
        for k in range(n):
            phase = 2.0 * math.pi * k / n
            if kind == "hann":
                value = 0.5 - 0.5 * math.cos(phase)
            elif kind == "hamming":
                value = 0.54 - 0.46 * math.cos(phase)
            else:
                value = (
                    0.42
                    - 0.5 * math.cos(phase)
                    + 0.08 * math.cos(2.0 * phase)
                )
            expected.append(value)
        _assert_vector_close(got, expected)


def test_frame_pads_tail_with_zeros():
    short = frame([1.0, 2.0, 3.0], 4, 2)
    _assert_matrix_close(short, [[1.0, 2.0, 3.0, 0.0]])

    samples = [float(i) for i in range(8)]
    frames = frame(samples, 4, 2)
    assert len(frames) == 3
    _assert_vector_close(frames[-1], [4.0, 5.0, 6.0, 7.0])


def test_frame_count_uses_correct_formula():
    samples = [float(i) for i in range(8)]
    frames = frame(samples, 4, 2)
    assert len(frames) == 1 + (len(samples) - 4) // 2
    _assert_matrix_close(
        frames,
        [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
    )


def test_rfft_magnitude_is_not_6db_low():
    n = 8
    bin_index = 2
    samples = [
        math.cos(2.0 * math.pi * bin_index * t / n) for t in range(n)
    ]
    spec = rfft(samples)
    assert len(spec) == n // 2 + 1
    assert abs(spec[bin_index]) == pytest.approx(1.0, abs=TOL)
    for index, value in enumerate(spec):
        if index != bin_index:
            assert abs(value) <= TOL


def test_magnitude_db_uses_twenty_log10():
    db = magnitude_db([1.0 + 0.0j, 0.1 + 0.0j, 0.01 + 0.0j], -120.0)
    _assert_vector_close(db, [0.0, -20.0, -40.0])


def test_magnitude_db_clamps_floor():
    db = magnitude_db([0.0 + 0.0j, 1e-9 + 0.0j, 1e-3 + 0.0j], -100.0)
    _assert_vector_close(db, [-100.0, -100.0, -60.0])
    assert all(math.isfinite(value) for value in db)


def test_mel_filterbank_has_no_double_gain():
    bank = mel_filterbank(3, 8, 8000, 0.0, 4000.0)
    assert len(bank) == 3
    for row in bank:
        assert len(row) == 8 // 2 + 1
        assert all(weight >= 0.0 for weight in row)
        assert max(row) > 0.0
    for bin_index in range(8 // 2 + 1):
        total = sum(row[bin_index] for row in bank)
        assert total <= 1.0 + TOL


def test_mel_filterbank_clamps_to_nyquist():
    bank = mel_filterbank(4, 16, 8000, 0.0, 12000.0)
    assert len(bank) == 4
    for row in bank:
        assert len(row) == 16 // 2 + 1
        assert all(math.isfinite(weight) for weight in row)
        assert all(weight >= 0.0 for weight in row)
        assert max(row) > 0.0


def test_stft_batch_does_not_share_spectrum_buffer():
    config = {"n_fft": 4, "hop": 4, "kind": "hann"}
    tracks = [
        {"id": "silence", "samples": [0.0] * 8},
        {"id": "impulse", "samples": [0.0, 1.0] + [0.0] * 6},
    ]
    batch = stft_batch(tracks, config)
    assert batch["order"] == ["silence", "impulse"]
    expected_silence = spectrogram(tracks[0]["samples"], 4, 4, "hann")
    expected_impulse = spectrogram(tracks[1]["samples"], 4, 4, "hann")
    _assert_matrix_close(batch["spectrograms"][0], expected_silence)
    _assert_matrix_close(batch["spectrograms"][1], expected_impulse)
    assert batch["spectrograms"][0] != batch["spectrograms"][1]


def test_frame_rejects_zero_hop():
    with pytest.raises(StftError):
        frame([0.0] * 8, 8, 0)


def test_pure_sine_peaks_at_expected_bin():
    n_fft = 64
    bin_index = 8
    samples = [
        math.cos(2.0 * math.pi * bin_index * t / n_fft)
        for t in range(n_fft)
    ]
    spec = spectrogram(samples, n_fft, n_fft, "hann")
    assert len(spec) == 1
    assert len(spec[0]) == n_fft // 2 + 1
    peak = max(range(len(spec[0])), key=lambda index: spec[0][index])
    assert peak == bin_index
    assert spec[0][peak] > -10.0

    bank = mel_filterbank(4, n_fft, 8000, 0.0, 4000.0)
    mel = to_mel(spec, bank)
    assert len(mel) == len(bank)
    assert len(mel[0]) == len(spec)
    assert all(math.isfinite(value) for row in mel for value in row)


def test_hop_equal_n_fft_has_no_overlap():
    samples = [float(i) for i in range(8)]
    frames = frame(samples, 4, 4)
    _assert_matrix_close(
        frames,
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
    )


def test_spectrogram_rejects_zero_hop():
    with pytest.raises(StftError):
        spectrogram([0.0] * 16, 8, 0, "hann")


def test_empty_audio_returns_empty():
    assert frame([], 8, 2) == []
    assert spectrogram([], 8, 2, "hann") == []


def test_stft_batch_preserves_track_order():
    config = {"n_fft": 4, "hop": 4, "kind": "hann"}
    tracks = [
        {"id": "z", "samples": [0.0] * 8},
        {"id": "a", "samples": [1.0] + [0.0] * 7},
        {"id": "m", "samples": [0.0] * 7 + [1.0]},
    ]
    batch = stft_batch(tracks, config)
    assert batch["order"] == ["z", "a", "m"]
    assert len(batch["spectrograms"]) == 3
    assert [len(item) for item in batch["spectrograms"]] == [2, 2, 2]
