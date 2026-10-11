"""Find kick, snare, tom, and cymbal hits in a drum stem.

Each instrument gets its own detector, so a kick under a hi-hat is two
hits, not one blended guess. Only the energy that arrives at a hit counts.
The ring of the last hit cannot fake a new one.
"""

from __future__ import annotations

import numpy as np

SR = 44100
HOP = 256
N_FFT = 2048
N_MELS = 112
FMIN = 30.0
FMAX = 16000.0
FRAME = HOP / SR


def _ms(value: float) -> int:
    return max(1, int(round(value / 1000.0 / FRAME)))


def _templates(freqs: np.ndarray) -> tuple[list[str], np.ndarray]:
    lf = np.log2(freqs)

    def bump(center: float, width_oct: float, amp: float = 1.0) -> np.ndarray:
        return amp * np.exp(-0.5 * ((lf - np.log2(center)) / width_oct) ** 2)

    def ramp(lo: float, hi: float) -> np.ndarray:
        return np.clip((lf - np.log2(lo)) / (np.log2(hi) - np.log2(lo)), 0.0, 1.0)

    shapes = {
        "kick": bump(60, 0.45) + bump(2500, 1.0, 0.03),
        "snare": bump(200, 0.5) + bump(3500, 1.3, 0.35),
        "hat": ramp(4000, 9000) + bump(800, 1.0, 0.01),
        "cymbal": bump(5000, 1.2) + bump(800, 0.8, 0.05),
    }
    names = list(shapes)
    w = np.stack([shapes[name] / shapes[name].sum() for name in names], axis=1)
    return names, w + 1e-6


def _nmf(v: np.ndarray, w0: np.ndarray, fixed: int = 0, iterations: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """KL-NMF that lets each template fit this kit, but only halfway.
    The last `fixed` templates were measured from this recording and stay as they are."""
    rng = np.random.default_rng(0)
    w = w0.copy()
    h = rng.random((w.shape[1], v.shape[1])).astype(np.float32) * float(v.mean())
    ones = np.ones_like(v)
    base = w0 / w0.sum(axis=0, keepdims=True)
    free = w.shape[1] - fixed
    for step in range(iterations):
        wh = w @ h + 1e-9
        h *= (w.T @ (v / wh)) / (w.T @ ones + 1e-9)
        if 20 <= step < 80:
            wh = w @ h + 1e-9
            fitted = w * ((v / wh) @ h.T) / (ones @ h.T + 1e-9)
            fitted /= fitted.sum(axis=0, keepdims=True) + 1e-12
            w[:, :free] = 0.5 * base[:, :free] + 0.5 * fitted[:, :free]
    return w, h


def _click_templates(
    mel: np.ndarray, click_frames: np.ndarray, cycle: int, lone_frames: np.ndarray | None = None
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """What one leftover click adds to the spectrum, learned from the clicks that land alone.

    Metronomes accent beat 1, so each beat of the bar gets its own template. Returns
    (template, every click frame on that beat, the lone ones) per beat. A count-in is
    known to be clicks alone, so when one is given it is the whole reference."""
    if lone_frames is not None and lone_frames.size:
        rises = []
        for frame in lone_frames:
            if 4 <= frame < mel.shape[1] - 4:
                before = mel[:, frame - 4 : frame - 1].mean(axis=1)
                after = mel[:, frame - 1 : frame + 4].max(axis=1)
                rises.append(np.maximum(after - before, 0.0))
        if rises:
            template = np.median(np.stack(rises), axis=0)
            if template.sum() > 0:
                return [(template / template.sum() + 1e-6, np.asarray(click_frames), np.asarray(lone_frames))]
    cycle = max(1, cycle)
    if click_frames.size < 8 * cycle:
        cycle = 1
    groups = []
    for phase in range(cycle):
        rises = []
        used = []
        for frame in click_frames[phase::cycle]:
            if frame < 4 or frame + 4 >= mel.shape[1]:
                continue
            before = mel[:, frame - 4 : frame - 1].mean(axis=1)
            after = mel[:, frame - 1 : frame + 4].max(axis=1)
            rises.append(np.maximum(after - before, 0.0))
            used.append(frame)
        if len(rises) < 4:
            continue
        rises = np.stack(rises)
        energy = rises.sum(axis=1)
        alone = energy <= np.percentile(energy, 25)
        template = np.median(rises[alone], axis=0)
        if template.sum() <= 0:
            continue
        used = np.asarray(used)
        groups.append((template / template.sum() + 1e-6, used, used[alone]))
    return groups


def _peaks(curve: np.ndarray, lookback: int, spacing: int) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima, scored by how far the curve jumped to get there."""
    n = curve.size
    if n < 3:
        return np.empty(0, int), np.empty(0)
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    local_max = maximum_filter1d(curve, size=2 * spacing + 1, mode="nearest")
    trough = minimum_filter1d(curve, size=lookback + 1, origin=lookback // 2, mode="nearest")
    rise = curve - trough
    idx = np.flatnonzero((curve >= local_max) & (rise > 0))
    if idx.size == 0:
        return idx, np.empty(0)
    keep = [idx[0]]
    for frame in idx[1:]:
        if frame - keep[-1] >= spacing:
            keep.append(frame)
        elif rise[frame] > rise[keep[-1]]:
            keep[-1] = frame
    keep = np.asarray(keep)
    return keep, rise[keep]


def _adaptive_keep(frames: np.ndarray, heights: np.ndarray, factor: float, window_sec: float = 8.0) -> np.ndarray:
    """Keep peaks that stand out for their part of the song, so quiet verses still count."""
    if frames.size == 0:
        return frames
    global_ref = float(np.percentile(heights, 90))
    times = frames * FRAME
    keep = np.zeros(frames.size, dtype=bool)
    for index, when in enumerate(times):
        near = heights[np.abs(times - when) <= window_sec / 2]
        local_ref = float(np.percentile(near, 90)) if near.size >= 4 else global_ref
        ref = max(local_ref, 0.4 * global_ref)
        keep[index] = heights[index] >= factor * ref
    return frames[keep]


def _band(spec: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    rows = (freqs >= lo) & (freqs < hi)
    return spec[rows].sum(axis=0)


def _db(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(values + 1e-10)


def _near(frames: np.ndarray, frame: int, radius: int) -> bool:
    if frames.size == 0:
        return False
    pos = np.searchsorted(frames, frame)
    for candidate in (pos - 1, pos):
        if 0 <= candidate < frames.size and abs(int(frames[candidate]) - frame) <= radius:
            return True
    return False


def _sustain(high_db: np.ndarray, frame: int, until_ms: float) -> float:
    """How far the top end falls after a hit, in dB. Later hits only add energy, so the
    quietest point is a fair read of the ring even when the drummer keeps playing."""
    n = high_db.size
    peak = float(np.max(high_db[frame : min(n, frame + _ms(15) + 1)]))
    lo = min(n - 1, frame + _ms(50))
    hi = min(n, frame + _ms(until_ms))
    if hi - lo < 2:
        return 40.0
    return peak - float(np.min(high_db[lo:hi]))


def _wash_rise(high_db: np.ndarray, frame: int) -> tuple[float, float]:
    """The cymbal wash that hangs between hits, after this one, and how much this hit lifted it.
    A hat played into a crash that is still ringing does not lift it."""
    n = high_db.size
    before = high_db[max(0, frame - _ms(300)) : max(1, frame - _ms(20))]
    after = high_db[min(n - 1, frame + _ms(50)) : min(n, frame + _ms(250))]
    if before.size < 2 or after.size < 2:
        return -120.0, 0.0
    floor_after = float(np.min(after))
    return floor_after, floor_after - float(np.min(before))


def _metal_kind(high_db: np.ndarray, wide_db: np.ndarray, frame: int, loud_ref: float, top: float) -> str:
    """Closed hats choke fast. Open hats ring until the foot closes them. Crashes ring loud and long."""
    short_drop = _sustain(high_db, frame, 150)
    long_drop = _sustain(high_db, frame, 400)
    peak = float(np.max(high_db[frame : min(high_db.size, frame + _ms(15) + 1)]))
    wide_peak = float(np.max(wide_db[frame : min(wide_db.size, frame + _ms(15) + 1)]))
    wash, lift = _wash_rise(high_db, frame)
    # Under a snare the crash hides behind the snare's peak, but its wash still shows.
    if peak >= top - 8.0 and wash >= top - 22.0 and lift >= 10.0:
        return "crash"
    if long_drop < 12.0 and lift >= 6.0:
        return "crash" if wide_peak >= loud_ref else "ride"
    if short_drop < 10.0 and lift >= 3.0:
        return "hi_hat_open"
    return "hi_hat_closed"


def _stream_fills(metal: np.ndarray, frame: int) -> bool:
    """A hat pattern that runs right through this hit, at an even spacing."""
    pos = np.searchsorted(metal, frame)
    if pos == 0 or pos >= metal.size:
        return False
    before = metal[pos - 1]
    after = metal[pos]
    gap_before = frame - before
    gap_after = after - frame
    if gap_before <= 0 or gap_after <= 0 or max(gap_before, gap_after) > _ms(600):
        return False
    ratio = gap_before / gap_after
    if not 0.8 <= ratio <= 1.25:
        return False
    if pos >= 2:
        prior = before - metal[pos - 2]
        if abs(prior - (gap_before + gap_after) / 2) > 0.25 * prior:
            return False
    return True


def _tom_names(pitches: list[float]) -> list[str]:
    """Split tom hits into floor, mid, and high by where their pitch clusters."""
    if not pitches:
        return []
    values = np.log2(np.asarray(pitches))
    order = np.sort(values)
    gaps = np.diff(order)
    splits = [order[i] for i in np.argsort(gaps)[::-1][:2] if gaps[i] > 0.25]
    splits.sort()
    labels = []
    for value in values:
        group = sum(value > edge for edge in splits)
        labels.append(group)
    groups = sorted(set(labels))
    if len(groups) == 1:
        mid = float(np.median(pitches))
        name = "low_tom" if mid < 110 else ("mid_tom" if mid < 170 else "high_tom")
        return [name] * len(pitches)
    names = ["low_tom", "mid_tom", "high_tom"] if len(groups) == 3 else ["low_tom", "high_tom"]
    lookup = {group: names[i] for i, group in enumerate(groups)}
    return [lookup[label] for label in labels]


def detect_hits(
    audio: np.ndarray,
    sample_rate: int,
    click_times: list[float] | None = None,
    click_cycle: int = 4,
    count_in: list[float] | None = None,
) -> list[dict]:
    """Return [{time_sec, instruments}] for one drum stem.

    Pass click_times when a metronome was mixed in, and click_cycle for how many clicks
    make a bar. Whatever is left of the click gets its own templates, so it cannot pass
    for a snare or a hat. Pass the count-in when there is one: those clicks have no
    drums under them, so they show best what the click alone sounds like."""
    import librosa

    if sample_rate != SR:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SR)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return []
    audio = audio / peak

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0
    ).astype(np.float32)
    freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    names, w0 = _templates(freqs)
    click_frames = np.asarray([int(round(t / FRAME)) for t in click_times or []], dtype=int)
    lone_frames = np.asarray([int(round(t / FRAME)) for t in count_in or []], dtype=int)
    click_groups = _click_templates(mel, click_frames, click_cycle, lone_frames) if click_frames.size else []
    for index, (template, _all, _lone) in enumerate(click_groups):
        names.append(f"click{index}")
        w0 = np.concatenate([w0, template[:, None]], axis=1)
    _w, act = _nmf(mel + 1e-9, w0.astype(np.float32), fixed=len(click_groups))
    act = {name: act[i] for i, name in enumerate(names)}

    power = mel**2
    high = _band(power, freqs, 6000, 16000)
    high_db = _db(high)
    wide_db = _db(_band(power, freqs, 2500, 16000))
    noise = _band(mel, freqs, 2000, 6000)
    body = _band(mel, freqs, 70, 400)

    from scipy.ndimage import uniform_filter1d

    smooth = {name: uniform_filter1d(curve, size=3) for name, curve in act.items()}

    def audible(frames: np.ndarray, heights: np.ndarray, curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Peaks in near-silence are rounding noise, however much they stand out locally."""
        floor = float(np.percentile(curve, 99.9)) * 10 ** (-45.0 / 20.0)
        keep = curve[frames] >= floor
        return frames[keep], heights[keep]

    kick_frames, kick_h = audible(*_peaks(smooth["kick"], _ms(60), _ms(60)), smooth["kick"])
    kick_frames = _adaptive_keep(kick_frames, kick_h, 0.22)
    snare_frames, snare_h = audible(*_peaks(smooth["snare"], _ms(50), _ms(55)), smooth["snare"])
    snare_frames = _adaptive_keep(snare_frames, snare_h, 0.2)

    high_flux = np.maximum(0.0, high_db[2:] - high_db[:-2])
    high_flux = np.concatenate([[0.0, 0.0], high_flux])
    level_floor = max(float(np.percentile(high_db, 95)) - 45.0, float(np.percentile(high_db, 99.9)) - 55.0)
    high_level = np.clip(high_db - level_floor, 0.0, None)
    metal_curve = uniform_filter1d(high_flux * high_level, size=2)
    metal_frames, metal_h = _peaks(metal_curve, _ms(12), _ms(55))
    # Leakage from a low drum is not a cymbal: the top end has to be loud next to the whole kit.
    loudest = float(np.percentile(_db(power.sum(axis=0)), 99.9))
    peak_high = np.asarray([high_db[f : f + _ms(15) + 1].max() for f in metal_frames]) if metal_frames.size else np.empty(0)
    keep = peak_high >= loudest - 70.0
    metal_frames, metal_h = metal_frames[keep], metal_h[keep]
    metal_frames = _adaptive_keep(metal_frames, metal_h, 0.12)

    # The NMF peak can land a frame or two after the attack. Pull each hit to its sharpest rise.
    def attack(frames: np.ndarray, curve: np.ndarray) -> np.ndarray:
        rise = np.diff(curve, prepend=curve[:1])
        out = []
        for frame in frames:
            lo = max(0, frame - _ms(25))
            out.append(lo + int(np.argmax(rise[lo : frame + 1])))
        return np.asarray(out, dtype=int)

    kick_frames = attack(kick_frames, smooth["kick"])
    snare_frames = attack(snare_frames, smooth["snare"])

    def rise_at(curve: np.ndarray, frame: int) -> float:
        lo = max(0, frame - _ms(40))
        hi = min(curve.size, frame + _ms(20) + 1)
        return float(curve[frame:hi].max() - curve[lo : frame + 1].min())

    # On a click, a hit has to stand well clear of what a click alone leaves behind on that beat.
    # Hits that only just clear it are marked, and kept only when there is playing around them.
    marginal: set[int] = set()
    if click_groups:
        click_sorted = np.sort(click_frames)

        def clear_of_click(frames: np.ndarray, curve: np.ndarray) -> np.ndarray:
            limits = []
            for _template, every, lone in click_groups:
                residue = np.asarray([rise_at(curve, int(f)) for f in lone])
                limit = max(3.0 * float(np.median(residue)), 1.5 * float(np.percentile(residue, 90)))
                limits.append((np.sort(every), limit))
            kept = []
            for frame in frames:
                frame = int(frame)
                limit = next((value for every, value in limits if _near(every, frame, _ms(20))), None)
                if limit is None and _near(click_sorted, frame, _ms(20)):
                    limit = max(value for _every, value in limits)
                if limit is None:
                    kept.append(frame)
                    continue
                rise = rise_at(curve, frame)
                if rise > limit:
                    kept.append(frame)
                    if rise < 4.0 * limit:
                        marginal.add(frame)
            return np.asarray(kept, dtype=int)

        kick_frames = clear_of_click(kick_frames, smooth["kick"])
        snare_frames = clear_of_click(snare_frames, smooth["snare"])
        metal_frames = clear_of_click(metal_frames, metal_curve)

    # A kick spills a little into the snare template. Learn how much from this song's kicks,
    # then only keep a snare on a kick when it clearly rises above that spill.

    if kick_frames.size >= 4 and snare_frames.size:
        spill = np.median([rise_at(smooth["snare"], int(f)) / (rise_at(smooth["kick"], int(f)) + 1e-9) for f in kick_frames])
        kept = []
        for frame in snare_frames:
            frame = int(frame)
            if _near(kick_frames, frame, _ms(35)):
                ratio = rise_at(smooth["snare"], frame) / (rise_at(smooth["kick"], frame) + 1e-9)
                if ratio < 3.0 * spill:
                    continue
            kept.append(frame)
        snare_frames = np.asarray(kept, dtype=int)

    # Snare wires hiss. A tom is the same body hit without the hiss.
    snare_ratio = []
    pitches = []
    for frame in snare_frames:
        lo, hi = frame, min(mel.shape[1], frame + _ms(40))
        snare_ratio.append(float(noise[lo:hi].sum() / (body[lo:hi].sum() + 1e-9)))
        rows = (freqs >= 60) & (freqs <= 400)
        spectrum = mel[rows, lo:hi].sum(axis=1)
        pitches.append(float(freqs[rows][int(np.argmax(spectrum))]))
    snare_ratio = np.asarray(snare_ratio)
    is_tom = np.zeros(snare_frames.size, dtype=bool)
    if snare_frames.size >= 8:
        typical = float(np.median(snare_ratio))
        is_tom = snare_ratio < 0.3 * typical

    tom_labels = _tom_names([pitches[i] for i in np.flatnonzero(is_tom)])
    tom_lookup = dict(zip(np.flatnonzero(is_tom).tolist(), tom_labels))

    hits: list[tuple[int, str]] = []
    for frame in kick_frames:
        hits.append((int(frame), "kick"))
    for index, frame in enumerate(snare_frames):
        hits.append((int(frame), tom_lookup.get(index, "snare")))

    # A snare hit is bright too. Only call a hat on top of it when the hat pattern runs through it.
    skin_radius = _ms(35)
    loud_ref = float(np.percentile(wide_db, 99)) - 6.0
    top = float(np.percentile(high_db, 99))
    stream = [frame for frame in metal_frames if not _near(snare_frames, int(frame), skin_radius)]
    stream = np.asarray(stream, dtype=int)
    for frame in metal_frames:
        frame = int(frame)
        kind = _metal_kind(high_db, wide_db, frame, loud_ref, top)
        on_snare = _near(snare_frames, frame, skin_radius)
        if on_snare and kind in ("hi_hat_closed", "hi_hat_open"):
            if not _stream_fills(stream, frame):
                continue
            # The snare's own ring keeps the top end up, so a hat under it reads as open.
            kind = "hi_hat_closed"
        hits.append((frame, kind))

    hits = [hit for hit in hits if hit[0] >= _ms(30)]
    if marginal:
        hits = _drop_lone_clicks(hits, marginal)
    hits.sort()
    return _merge(hits)


def _drop_lone_clicks(hits: list[tuple[int, str]], marginal: set[int]) -> list[tuple[int, str]]:
    """A hit that barely beat the click, with no clear playing around it, is the click itself."""
    anchors = np.sort(np.asarray([frame for frame, _name in hits if frame not in marginal], dtype=int))
    reach = _ms(1500)
    return [hit for hit in hits if hit[0] not in marginal or _near(anchors, hit[0], reach)]


def _merge(hits: list[tuple[int, str]]) -> list[dict]:
    """Hits within 30 ms are one stroke of the kit."""
    radius = _ms(30)
    events: list[dict] = []
    for frame, name in hits:
        if events and frame - events[-1]["_frame"] <= radius:
            if name not in events[-1]["instruments"]:
                events[-1]["instruments"].append(name)
            continue
        events.append({"_frame": frame, "time_sec": round(frame * FRAME, 5), "instruments": [name]})
    for event in events:
        event.pop("_frame")
        if "crash" in event["instruments"]:
            event["instruments"] = [n for n in event["instruments"] if n not in ("hi_hat_closed", "hi_hat_open")]
        if "hi_hat_open" in event["instruments"] and "hi_hat_closed" in event["instruments"]:
            event["instruments"].remove("hi_hat_closed")
    return events
