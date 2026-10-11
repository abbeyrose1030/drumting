"""Line the hits up with the bar, so beat 1 of the chart is beat 1 of the song."""

from __future__ import annotations

import math

from app.engrave import parse_time_signature


def _sixteenth(bpm: float) -> float:
    return 60.0 / float(bpm) / 4.0


def grid_phase(events: list[dict], bpm: float) -> float:
    """Where the sixteenth-note grid sits, in seconds, from the average lean of every hit."""
    step = _sixteenth(bpm)
    if not events:
        return 0.0
    sin_sum = cos_sum = 0.0
    for event in events:
        angle = 2.0 * math.pi * (float(event["time_sec"]) % step) / step
        sin_sum += math.sin(angle)
        cos_sum += math.cos(angle)
    if abs(sin_sum) + abs(cos_sum) < 1e-9:
        return 0.0
    angle = math.atan2(sin_sum, cos_sum) % (2.0 * math.pi)
    return angle / (2.0 * math.pi) * step


def _downbeat_score(ticks: list[tuple[int, list[str]]], shift: int, measure: int, beat: int, backbeats: set[int]) -> float:
    score = 0.0
    for tick, names in ticks:
        pos = (tick - shift) % measure
        if pos == 0:
            if "kick" in names:
                score += 3.0
            if "crash" in names:
                score += 3.0
            if "snare" in names:
                score -= 1.0
        elif pos in backbeats and "snare" in names:
            score += 2.0
        elif pos % beat == 0 and "kick" in names:
            score += 0.5
    return score


def align_to_downbeat(events: list[dict], bpm: float, time_signature: str, phase_known: bool = False) -> list[dict]:
    """Shift hits so the strongest downbeat lands on beat 1, and drop leading empty bars.

    With a click the sub-beat phase is already exact, so only whole beats are tried."""
    if not events:
        return events
    numerator, denominator = parse_time_signature(time_signature)
    step = _sixteenth(bpm)
    measure_ql = numerator * (4.0 / denominator)
    measure = int(round(measure_ql * 4))
    beat = 6 if denominator == 8 and numerator % 3 == 0 else (2 if denominator == 8 else int(round(16 / denominator)))
    backbeats = {beat, 3 * beat} if (numerator, denominator) == (4, 4) else set()

    phase = 0.0 if phase_known else grid_phase(events, bpm)
    ticks = [(int(round((float(e["time_sec"]) - phase) / step)), list(e.get("instruments") or [])) for e in events]
    candidates = range(0, measure, beat) if phase_known else range(measure)
    best = max(candidates, key=lambda shift: (_downbeat_score(ticks, shift, measure, beat, backbeats), -shift))

    offset = phase + best * step
    bar = measure * step
    first = min(float(e["time_sec"]) for e in events)
    while offset > first + step / 2:
        offset -= bar
    while offset + bar <= first + step / 2:
        offset += bar
    return [{**event, "time_sec": round(max(0.0, float(event["time_sec"]) - offset), 6)} for event in events]
