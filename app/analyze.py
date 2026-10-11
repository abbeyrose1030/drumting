"""Find the hits in an isolated drum stem. Rudiment mode still uses DrumScript's classifier."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "DrumScript"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


def _clicks_per_bar(click: dict | None, time_signature: str) -> int:
    if click is None:
        return 1
    try:
        num_s, den_s = str(time_signature).split("/")
        bar = int(num_s) * (4.0 / int(den_s)) * float(click["quarter"])
    except (ValueError, KeyError, ZeroDivisionError):
        return 4
    spacing = click["click_times"][1] - click["click_times"][0] if len(click["click_times"]) > 1 else bar
    return max(1, int(round(bar / spacing)))


def analyze(
    audio_path: str,
    rudiment: bool = False,
    bpm: float | None = None,
    time_signature: str = "4/4",
    use_click: bool = True,
) -> dict:
    """Return tempo and classified hits. Audio must already be drums only."""
    import numpy as np

    from app.click_track import (
        align_to_click,
        click_from_count_in,
        drop_click_residuals,
        find_click,
        find_count_in,
        remove_click,
    )
    from app.detect import detect_hits
    from app.grid import align_to_downbeat
    from drumscript.audio_processor.audio_loader import load_audio, normalise_audio
    from drumscript.audio_processor.onset_detector import detect_onsets
    from drumscript.audio_processor.tempo_detector import estimate_tempo
    from drumscript.drum_classifier.classify import classify_rudiment_events
    from drumscript.notation_generator.constants import SAMPLE_RATE

    audio, sample_rate = load_audio(str(audio_path), sr=SAMPLE_RATE)
    audio = normalise_audio(audio)
    if audio.size == 0:
        raise ValueError("That audio file is empty.")

    duration = len(audio) / sample_rate
    count_in = [] if rudiment else find_count_in(audio, sample_rate)
    if count_in:
        print(f"Count-in of {len(count_in)} clicks from {count_in[0]:.3f}s")

    click = None
    clicks_removed = 0
    if use_click and not rudiment:
        click = find_click(audio, sample_rate, bpm_hint=bpm, time_signature=time_signature, count_in=count_in)
        if click is not None:
            audio, clicks_removed = remove_click(audio, sample_rate, click)
            print(f"Click grid at {click['tempo']} BPM, {clicks_removed} clicks removed, downbeat at {click['origin']:.3f}s")
    # Without a click to remove, a count-in still fixes the tempo and where beat 1 falls.
    grid = click
    if grid is None and count_in:
        grid = click_from_count_in(count_in, duration, bpm, time_signature)

    if bpm:
        tempo = float(bpm)
    elif grid is not None:
        tempo = float(grid["tempo"])
    else:
        raw_tempo = estimate_tempo(audio, sample_rate)
        tempo = float(np.atleast_1d(raw_tempo).astype(float)[0])
        if not np.isfinite(tempo) or tempo <= 0:
            tempo = 120.0

    if rudiment:
        onsets = [float(onset) for onset in detect_onsets(audio, sample_rate)]
        classified = classify_rudiment_events(audio, sample_rate, onsets)
    else:
        click_times = None if click is None else click["click_times"]
        classified = detect_hits(
            audio, sample_rate, click_times, _clicks_per_bar(click, time_signature), count_in if click else None
        )
        if click is not None:
            times = [event["time_sec"] for event in classified]
            kept = set(drop_click_residuals(audio, sample_rate, times, click["click_times"]))
            classified = [event for event in classified if event["time_sec"] in kept]
        onsets = [event["time_sec"] for event in classified]

    events = []
    for event in classified:
        instruments = []
        for name in event.get("instruments") or []:
            if name not in instruments:
                instruments.append(name)
        events.append({"time_sec": float(event["time_sec"]), "instruments": instruments})

    if count_in:
        events = [e for e in events if min(abs(e["time_sec"] - t) for t in count_in) > 0.03]
    if grid is not None:
        events = align_to_click(events, grid)
    if not rudiment and not count_in:
        events = align_to_downbeat(events, tempo, time_signature, phase_known=click is not None)

    return {
        "tempo": round(tempo, 2),
        "events": events,
        "onset_count": len(onsets),
        "duration_sec": round(len(audio) / sample_rate, 3),
        "click_found": click is not None,
        "clicks_removed": clicks_removed,
        "grid_origin": None if grid is None else grid["origin"],
        "count_in": len(count_in),
    }
