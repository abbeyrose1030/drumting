"""Find a metronome click mixed into a drum stem, then lift it back out.

The click is a short pulse that lands on a steady beat and matches itself
almost exactly from beat to beat. Drum hits are noisier, so they stay.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _quarter_period(bpm: float) -> float:
    return 60.0 / float(bpm)


def _bar_seconds(bpm: float, time_signature: str) -> float:
    num_s, den_s = str(time_signature).split("/")
    numerator, denominator = int(num_s), int(den_s)
    return numerator * (4.0 / denominator) * _quarter_period(bpm)


def _highpass(audio: np.ndarray, sample_rate: int, cutoff: float = 1800.0) -> np.ndarray:
    sos = butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def _pulse_start(audio: np.ndarray, hit_time: float, sample_rate: int) -> int | None:
    """Start the window a couple of milliseconds before the local peak."""
    center = int(round(hit_time * sample_rate))
    radius = int(0.008 * sample_rate)
    left = max(0, center - radius)
    right = min(len(audio), center + radius)
    if right <= left:
        return None
    peak = left + int(np.argmax(np.abs(audio[left:right])))
    return max(0, peak - int(0.002 * sample_rate))


def _is_short_pulse(audio: np.ndarray, hit_time: float, sample_rate: int) -> bool:
    """A click dies within a few dozen milliseconds. A hat or snare does not."""
    start = _pulse_start(audio, hit_time, sample_rate)
    if start is None:
        return False
    early_end = start + int(0.015 * sample_rate)
    late_start = start + int(0.045 * sample_rate)
    late_end = start + int(0.09 * sample_rate)
    if late_end > len(audio) or early_end <= start:
        return False
    early = float(np.sqrt(np.mean(audio[start:early_end] ** 2))) + 1e-9
    late = float(np.sqrt(np.mean(audio[late_start:late_end] ** 2)))
    return late / early < 0.35


def _snippets(audio: np.ndarray, times: list[float], sample_rate: int, window: int) -> tuple[np.ndarray, list[float]]:
    kept_audio = []
    kept_times = []
    for hit_time in times:
        start = _pulse_start(audio, hit_time, sample_rate)
        if start is None or start + window > len(audio):
            continue
        chunk = audio[start : start + window]
        if np.max(np.abs(chunk)) < 1e-4:
            continue
        kept_audio.append(chunk)
        kept_times.append(float(hit_time))
    if not kept_audio:
        return np.empty((0, window), dtype=np.float32), []
    return np.stack(kept_audio), kept_times


def _spectral_units(snippets: np.ndarray) -> np.ndarray:
    """Phase-invariant fingerprint, so a few milliseconds of jitter still match."""
    window = np.hanning(snippets.shape[1]).astype(np.float32)
    magnitude = np.log1p(np.abs(np.fft.rfft(snippets * window, axis=1)))
    centered = magnitude - magnitude.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return centered / norms


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - np.mean(left)
    right = right - np.mean(right)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)


def _candidate_periods(bpm_hint: float | None, audio: np.ndarray, sample_rate: int) -> list[float]:
    if bpm_hint:
        quarter = _quarter_period(bpm_hint)
    else:
        import librosa

        estimate = librosa.feature.tempo(y=audio, sr=sample_rate)
        bpm = float(np.atleast_1d(estimate).astype(float)[0])
        if not np.isfinite(bpm) or bpm <= 0:
            bpm = 120.0
        quarter = _quarter_period(bpm)
    # A click is usually on the quarter, sometimes on the eighth or the half.
    return [quarter, quarter / 2.0, quarter * 2.0]


def _matching_grid(times: np.ndarray, anchor: float, period: float, tolerance: float = 0.02) -> np.ndarray:
    steps = np.round((times - anchor) / period)
    error = np.abs(times - (anchor + steps * period))
    return error <= tolerance


def _best_grid(audio_hi: np.ndarray, sample_rate: int, period: float, duration: float) -> tuple[list[float], float]:
    """Pick the phase where the same short pulse repeats on this spacing."""
    import librosa

    onset_times = librosa.onset.onset_detect(
        y=audio_hi,
        sr=sample_rate,
        hop_length=128,
        units="time",
        delta=0.02,
        wait=1,
        backtrack=True,
    )
    onset_times = np.asarray(onset_times, dtype=float)
    if onset_times.size < 6:
        return [], -1.0

    window = int(0.025 * sample_rate)
    snippets, times = _snippets(audio_hi, onset_times.tolist(), sample_rate, window)
    if len(times) < 6:
        return [], -1.0

    times_arr = np.asarray(times, dtype=float)
    short = [index for index, hit_time in enumerate(times) if _is_short_pulse(audio_hi, hit_time, sample_rate)]
    if len(short) < 6:
        return [], -1.0
    snippets = snippets[short]
    times_arr = times_arr[short]
    unit = _spectral_units(snippets)

    best_rank = -1.0
    best_times: list[float] = []
    # Each detected pulse is a possible downbeat. The click is the one the others resemble.
    for index, anchor in enumerate(times_arr):
        hits = _matching_grid(times_arr, float(anchor), period)
        hit_count = int(np.count_nonzero(hits))
        if hit_count < 6:
            continue
        hit_index = np.flatnonzero(hits)
        scores = unit[hit_index] @ unit[index]
        strong = scores >= 0.85
        strong_count = int(np.count_nonzero(strong))
        # A snare on top of the click hides some beats. Enough clean pulses still reveal the grid.
        if strong_count < 6:
            continue
        expected = max(1.0, duration / period)
        coverage = strong_count / expected
        if coverage < 0.3:
            continue
        rank = float(strong_count)
        if rank > best_rank:
            chosen = hit_index[strong]
            best_times = [float(value) for value in times_arr[chosen]]
            best_rank = rank
    return best_times, best_rank


def _fold_bpm(period: float) -> float:
    bpm = 60.0 / period
    while bpm < 70:
        bpm *= 2.0
    while bpm > 180:
        bpm /= 2.0
    return bpm


def find_count_in(audio: np.ndarray, sample_rate: int) -> list[float]:
    """The run of identical, evenly spaced clicks before the drums come in.

    Returns their times, or [] when the stem does not open with one."""
    import librosa

    pad = int(0.1 * sample_rate)
    padded = np.concatenate([np.zeros(pad, dtype=np.float32), audio[: int(40 * sample_rate)].astype(np.float32)])
    onsets = librosa.onset.onset_detect(y=padded, sr=sample_rate, hop_length=256, units="time", backtrack=False, delta=0.1)
    onsets = [float(t) - 0.1 for t in onsets if t >= 0.08]
    if len(onsets) < 4:
        return []

    window = int(0.025 * sample_rate)
    snippets, times = _snippets(padded, [t + 0.1 for t in onsets[:40]], sample_rate, window)
    if len(times) < 4:
        return []
    times = [t - 0.1 for t in times]
    units = _spectral_units(snippets)
    short = [_is_short_pulse(padded, t + 0.1, sample_rate) for t in times]

    run = [0]
    period = None
    for index in range(1, len(times)):
        # Beat 1 is often an accented click, so it only roughly matches the others.
        if not short[index] or max(float(units[index] @ units[j]) for j in run) < 0.7:
            break
        gap = times[index] - times[run[-1]]
        if period is None:
            if not 0.25 <= gap <= 1.6:
                break
            period = gap
        elif abs(gap - period) > 0.03 * period:
            break
        run.append(index)
    if not short[0] or len(run) < 3:
        return []
    return [round(times[i], 5) for i in run]


def _grid_click(origin: float, period: float, template_times: list[float], duration: float, bpm_hint: float | None, time_signature: str, count_in: bool) -> dict:
    if bpm_hint:
        written_bpm = float(bpm_hint)
    else:
        written_bpm = _fold_bpm(period)
    quarter = _quarter_period(written_bpm)
    # An accented click can match itself only on every other beat, so the grid steps
    # at least once per quarter. Removal skips any grid point that holds no click.
    step = min(period, quarter) if bpm_hint else period
    click_times = []
    cursor = origin
    while cursor < duration - 0.02:
        click_times.append(round(cursor, 5))
        cursor += step
    return {
        "found": True,
        "tempo": round(written_bpm, 2),
        "period": period,
        "quarter": quarter,
        "origin": round(origin, 5),
        "template_times": [round(hit, 5) for hit in template_times],
        "click_times": click_times,
        "time_signature": time_signature,
        "count_in": count_in,
    }


def click_from_count_in(count_in: list[float], duration: float, bpm_hint: float | None, time_signature: str) -> dict:
    """A count-in starts on beat 1 and sets the tempo, so it fixes the whole grid."""
    period = (count_in[-1] - count_in[0]) / (len(count_in) - 1)
    if bpm_hint:
        quarter = _quarter_period(bpm_hint)
        # Some count-ins click on the half note or the eighth. A typed tempo that fits none
        # of those is a typo, and the count-in wins.
        if not any(abs(period - k * quarter) <= 0.04 * k * quarter for k in (0.5, 1.0, 2.0)):
            bpm_hint = None
    return _grid_click(count_in[0], period, count_in, duration, bpm_hint, time_signature, True)


def find_click(
    audio: np.ndarray,
    sample_rate: int,
    bpm_hint: float | None = None,
    time_signature: str = "4/4",
    count_in: list[float] | None = None,
) -> dict | None:
    """Locate a mixed-in click. Returns None when the stem has no steady click."""
    duration = len(audio) / sample_rate
    if duration < 2.0:
        return None
    if count_in:
        return click_from_count_in(count_in, duration, bpm_hint, time_signature)

    high = _highpass(audio, sample_rate)
    peak = float(np.max(np.abs(high))) or 1.0
    high = high / peak

    found = []
    quarter_hint = _quarter_period(bpm_hint) if bpm_hint else None
    for period in _candidate_periods(bpm_hint, audio, sample_rate):
        times, rank = _best_grid(high, sample_rate, period, duration)
        if rank < 6 or len(times) < 6:
            continue
        # When the tempo is known, a quarter-note click outranks a half-note one that only sees every other beat.
        preference = 1.0
        if quarter_hint:
            preference = 1.0 / (1.0 + abs(period - quarter_hint) / quarter_hint)
        found.append((rank * preference, period, times))

    if not found:
        return None
    _score, best_period, best_times = max(found, key=lambda item: item[0])

    if best_period is None:
        return None
    # The clean pulses give the phase. Fill every beat, including ones a drum hit covered.
    return _grid_click(min(best_times), best_period, best_times, duration, bpm_hint, time_signature, False)


def remove_click(audio: np.ndarray, sample_rate: int, click: dict) -> tuple[np.ndarray, int]:
    """Subtract the repeating click waveform. Drum hits that do not match it stay."""
    window = int(0.03 * sample_rate)
    high = _highpass(audio, sample_rate)
    template_times = click.get("template_times") or click["click_times"]
    snippets, times = _snippets(high, template_times, sample_rate, window)
    if len(times) < 4:
        return audio, 0

    # Subtract only the bright click. A kick underneath is not in this high-passed template.
    template = np.median(snippets, axis=0)
    template = template - np.mean(template)

    cleaned = audio.copy()
    removed = 0
    for hit_time in click["click_times"]:
        start = _pulse_start(high, hit_time, sample_rate)
        if start is None or start + window > len(cleaned):
            continue
        segment = cleaned[start : start + window]
        high_segment = high[start : start + window]
        likeness = _correlation(high_segment, template)
        if likeness < 0.55:
            continue
        centered = high_segment - np.mean(high_segment)
        scale = float(np.dot(centered, template) / (np.dot(template, template) + 1e-9))
        if scale <= 0:
            continue
        cleaned[start : start + window] = segment - (scale * template)
        removed += 1

    np.clip(cleaned, -1.0, 1.0, out=cleaned)
    return cleaned, removed


def drop_click_residuals(audio: np.ndarray, sample_rate: int, onsets: list[float], click_times: list[float]) -> list[float]:
    """Drop a click that left a tiny transient. Keep the onset when a drum is still there."""
    peak = float(np.max(np.abs(audio))) + 1e-9
    kept = []
    window = int(0.08 * sample_rate)
    for onset in onsets:
        if not any(abs(onset - hit_time) <= 0.03 for hit_time in click_times):
            kept.append(onset)
            continue
        start = max(0, int(onset * sample_rate))
        segment = audio[start : start + window]
        if segment.size and float(np.max(np.abs(segment))) >= peak * 0.18:
            kept.append(onset)
    return kept


def align_to_click(events: list[dict], click: dict) -> list[dict]:
    """Put beat 1 on the first click, and drop a click-only count-in."""
    origin = float(click["origin"])
    bpm = float(click["tempo"])
    bar = _bar_seconds(bpm, click.get("time_signature") or "4/4")
    shifted = []
    for event in events:
        hit_time = float(event["time_sec"]) - origin
        if hit_time >= -0.03:
            shifted.append({**event, "time_sec": round(max(0.0, hit_time), 6)})
    if not shifted or bar <= 0:
        return shifted

    first = min(event["time_sec"] for event in shifted)
    skip_bars = 0
    while first >= (skip_bars + 1) * bar - 0.05:
        skip_bars += 1
    if skip_bars <= 0:
        return shifted
    cut = skip_bars * bar
    aligned = []
    for event in shifted:
        hit_time = event["time_sec"] - cut
        if hit_time >= -0.03:
            aligned.append({**event, "time_sec": round(max(0.0, hit_time), 6)})
    return aligned
