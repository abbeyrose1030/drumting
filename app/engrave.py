"""Turn classified drum hits into treble-clef drum-set notation."""

from __future__ import annotations

import math
import shutil
import threading
from pathlib import Path

# Verovio picks its font folder from the thread that first imports it.
# Requests run on worker threads, so later charts were pointed at a
# leftover installer path and the fonts could not be found.
_ENGRAVE_LOCK = threading.Lock()

# Staff positions are the usual drum-set layout, written on a treble staff.
# Middle line is B4. These are the same lines and spaces DrumScript uses,
# one octave higher so they sit correctly under a treble clef.
STAFF = {
    "kick": ("F4", "normal"),
    "low_tom": ("A4", "normal"),
    "snare": ("C5", "normal"),
    "mid_tom": ("D5", "normal"),
    "high_tom": ("E5", "normal"),
    "ride": ("F5", "x"),
    "hi_hat_closed": ("G5", "x"),
    "hi_hat_open": ("G5", "circle-x"),
    "crash": ("A5", "x"),
    "unknown": ("B4", "triangle"),
}

UP_INSTRUMENTS = {"hi_hat_closed", "hi_hat_open", "ride", "crash"}

MIDI_PITCH = {
    "kick": 36,
    "snare": 38,
    "low_tom": 45,
    "mid_tom": 47,
    "high_tom": 50,
    "hi_hat_closed": 42,
    "hi_hat_open": 46,
    "ride": 51,
    "crash": 49,
}

DISPLAY_NAME = {
    "kick": "Kick",
    "snare": "Snare",
    "low_tom": "Floor tom",
    "mid_tom": "Mid tom",
    "high_tom": "High tom",
    "hi_hat_closed": "Closed hi-hat",
    "hi_hat_open": "Open hi-hat",
    "ride": "Ride",
    "crash": "Crash",
    "unknown": "Unidentified",
}


def parse_time_signature(time_signature: str) -> tuple[int, int]:
    try:
        num_s, den_s = str(time_signature).strip().split("/")
        num, den = int(num_s), int(den_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("Time signature must look like 4/4 or 6/8.") from exc
    if den not in (2, 4, 8, 16) or not 1 <= num <= 16:
        raise ValueError("Use a time signature such as 4/4, 3/4, 6/8, 5/4, or 7/8.")
    return num, den


def beat_ticks(numerator: int, denominator: int, ticks_per_quarter: int) -> int:
    """Length of one beat in grid ticks. Compound meters use a dotted beat."""
    if denominator == 8 and numerator % 3 == 0:
        return int(ticks_per_quarter * 3 / 2)
    if denominator == 8:
        return max(1, ticks_per_quarter // 2)
    if denominator == 2:
        return ticks_per_quarter * 2
    if denominator == 16:
        return max(1, ticks_per_quarter // 4)
    return ticks_per_quarter


def _duration_sizes(ticks_per_quarter: int) -> list[int]:
    raw = [
        ticks_per_quarter * 4,
        ticks_per_quarter * 3,
        ticks_per_quarter * 2,
        int(round(ticks_per_quarter * 1.5)),
        ticks_per_quarter,
        int(round(ticks_per_quarter * 0.75)),
        ticks_per_quarter // 2,
        ticks_per_quarter // 4,
    ]
    sizes: list[int] = []
    for size in raw:
        if size >= 1 and size not in sizes:
            sizes.append(size)
    return sizes


def _can_use(cursor: int, dur: int, remaining: int, beat: int) -> bool:
    if dur < 1 or dur > remaining:
        return False
    dotted = dur & (dur - 1) != 0
    aligned = cursor % dur == 0 or (dotted and beat > 0 and cursor % beat == 0)
    if not aligned:
        return False
    start_beat = cursor // beat
    end_beat = (cursor + dur - 1) // beat
    if start_beat != end_beat and (dur < beat or cursor % beat != 0):
        return False
    return True


def spell_ticks(start: int, length: int, beat: int, ticks_per_quarter: int) -> list[int]:
    """Split a span into note values that sit on the beat grid."""
    sizes = _duration_sizes(ticks_per_quarter)
    pieces: list[int] = []
    cursor = start
    remaining = length
    while remaining > 0:
        chosen = next((size for size in sizes if _can_use(cursor, size, remaining, beat)), None)
        if chosen is None:
            chosen = 1 if remaining >= 1 else remaining
        pieces.append(chosen)
        cursor += chosen
        remaining -= chosen
    return pieces


def attack_length(tick: int, next_tick: int, beat: int, ticks_per_quarter: int) -> int:
    """Longest single note value that starts on this hit and stays inside the beat."""
    beat_end = ((tick // beat) + 1) * beat
    room = min(next_tick, beat_end) - tick
    for dur in _duration_sizes(ticks_per_quarter):
        if _can_use(tick, dur, room, beat):
            return dur
    return 1


def voice_spans(hit_ticks: list[int], measure_ticks: int, beat: int, ticks_per_quarter: int) -> list[tuple[str, int, int]]:
    """Each hit is one notehead. The rest of the bar is rests, spelled normally."""
    hits = sorted({tick for tick in hit_ticks if 0 <= tick < measure_ticks})
    if not hits:
        return [("rest", 0, measure_ticks)]

    spans: list[tuple[str, int, int]] = []
    if hits[0] > 0:
        spans.append(("rest", 0, hits[0]))
    for index, tick in enumerate(hits):
        nxt = hits[index + 1] if index + 1 < len(hits) else measure_ticks
        note_len = attack_length(tick, nxt, beat, ticks_per_quarter)
        spans.append(("note", tick, note_len))
        cursor = tick + note_len
        if cursor < nxt:
            spans.append(("rest", cursor, nxt - cursor))
    return spans


def quantize(
    events: list[dict],
    bpm: float,
    time_signature: str = "4/4",
    subdivision: int = 16,
    shift_sixteenths: int = 0,
) -> dict:
    """Snap hit times onto a rhythmic grid and group them into measures."""
    if subdivision not in (8, 16):
        raise ValueError("Grid must be eighth notes or sixteenth notes.")
    bpm = float(bpm)
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("Tempo must be a positive BPM.")

    numerator, denominator = parse_time_signature(time_signature)
    # Always count in sixteenth-note ticks so dotted values stay whole numbers.
    # An eighth-note grid just snaps each hit onto the nearest eighth.
    ticks_per_quarter = 4
    snap = 1 if subdivision == 16 else 2

    beat = beat_ticks(numerator, denominator, ticks_per_quarter)
    measure_ql = numerator * (4.0 / denominator)
    measure_ticks = int(round(measure_ql * ticks_per_quarter))
    seconds_per_tick = (60.0 / bpm) / ticks_per_quarter

    grouped: dict[int, dict[str, dict[int, list[str]]]] = {}
    quantized_events: list[dict] = []
    dropped = 0

    for event in events:
        raw_tick = int(round(float(event["time_sec"]) / seconds_per_tick))
        raw_tick = int(round(raw_tick / snap) * snap) + int(shift_sixteenths)
        if raw_tick < 0:
            dropped += 1
            continue
        measure_index = raw_tick // measure_ticks
        tick = raw_tick % measure_ticks
        bucket = grouped.setdefault(measure_index, {"up": {}, "down": {}})
        names = []
        for name in event.get("instruments") or []:
            if name not in STAFF or name in names:
                continue
            names.append(name)
            voice = "up" if name in UP_INSTRUMENTS else "down"
            slot = bucket[voice].setdefault(tick, [])
            if name not in slot:
                slot.append(name)
        if names:
            quantized_events.append(
                {
                    "time_sec": round(raw_tick * seconds_per_tick, 6),
                    "instruments": names,
                }
            )

    last = max(grouped) if grouped else 0
    measures = []
    for index in range(last + 1):
        measures.append(grouped.get(index, {"up": {}, "down": {}}))

    quantized_events.sort(key=lambda item: item["time_sec"])
    return {
        "measures": measures,
        "quantized": quantized_events,
        "bpm": bpm,
        "time_signature": f"{numerator}/{denominator}",
        "numerator": numerator,
        "denominator": denominator,
        "ticks_per_quarter": ticks_per_quarter,
        "measure_ticks": measure_ticks,
        "beat": beat,
        "dropped": dropped,
    }


def count_instruments(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        for name in event.get("instruments") or []:
            if name in STAFF:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _elements_for_voice(tick_map: dict[int, list[str]], measure_ticks: int, beat: int, ticks_per_quarter: int, stems_up: bool):
    from music21 import chord, note

    spans = voice_spans(list(tick_map), measure_ticks, beat, ticks_per_quarter)
    elements = []
    for kind, start, length in spans:
        durations = [length] if kind == "note" else spell_ticks(start, length, beat, ticks_per_quarter)
        for dur in durations:
            quarter_length = dur / ticks_per_quarter
            if kind == "rest":
                elements.append(note.Rest(quarterLength=quarter_length))
                continue
            pitches = []
            for name in tick_map.get(start, []):
                pitch_name, head = STAFF[name]
                written = note.Note(pitch_name)
                written.notehead = head
                pitches.append(written)
            pitches.sort(key=lambda item: item.pitch.midi)
            element = pitches[0] if len(pitches) == 1 else chord.Chord(pitches)
            element.quarterLength = quarter_length
            element.stemDirection = "up" if stems_up else "down"
            elements.append(element)
    return elements


def build_score(
    events: list[dict],
    bpm: float,
    time_signature: str = "4/4",
    title: str = "Drum Notation",
    subdivision: int = 16,
    shift_sixteenths: int = 0,
    measures_per_system: int = 4,
):
    """Build a music21 score and the quantization used to draw it."""
    from music21 import clef, layout, meter, metadata, stream, tempo

    grid = quantize(events, bpm, time_signature, subdivision, shift_sixteenths)
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title or "Drum Notation"
    score.metadata.composer = "Drum set"

    part = stream.Part()
    part.partName = "Drum Set"
    part.partAbbreviation = "Drs."

    for index, measure_events in enumerate(grid["measures"]):
        measure = stream.Measure(number=index + 1)
        if index == 0:
            measure.insert(0, clef.TrebleClef())
            measure.insert(0, meter.TimeSignature(grid["time_signature"]))
            measure.insert(0, tempo.MetronomeMark(number=round(grid["bpm"])))
        elif index % max(1, measures_per_system) == 0:
            measure.insert(0, layout.SystemLayout(isNew=True))

        up_hits = measure_events["up"]
        down_hits = measure_events["down"]
        if not up_hits and not down_hits:
            from music21 import note

            measure.append(note.Rest(quarterLength=grid["measure_ticks"] / grid["ticks_per_quarter"]))
        else:
            up = stream.Voice(id="1")
            down = stream.Voice(id="2")
            for element in _elements_for_voice(
                up_hits, grid["measure_ticks"], grid["beat"], grid["ticks_per_quarter"], True
            ):
                up.append(element)
            for element in _elements_for_voice(
                down_hits, grid["measure_ticks"], grid["beat"], grid["ticks_per_quarter"], False
            ):
                down.append(element)
            measure.insert(0, up)
            measure.insert(0, down)
        part.append(measure)

    part.makeBeams(inPlace=True)
    score.insert(0, part)
    return score, grid


def write_musicxml(score, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(path))
    return path


def write_midi(quantized_events: list[dict], bpm: float, path: Path) -> Path:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    for event in quantized_events:
        start = float(event["time_sec"])
        for name in event["instruments"]:
            pitch = MIDI_PITCH.get(name)
            if pitch is None:
                continue
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.08))
    midi.instruments.append(drums)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path


def render_pages(musicxml_path: Path, svg_dir: Path, pdf_path: Path) -> list[Path]:
    """Engrave MusicXML to SVG pages and a PDF with Verovio."""
    import verovio

    svg_dir.mkdir(parents=True, exist_ok=True)
    for old in svg_dir.glob("page-*.svg"):
        old.unlink()

    font_dir = str(Path(verovio.__file__).resolve().parent / "data")
    with _ENGRAVE_LOCK:
        verovio.setDefaultResourcePath(font_dir)
        toolkit = verovio.toolkit()
        toolkit.setResourcePath(font_dir)
        pages = _engrave_svg(toolkit, musicxml_path, svg_dir)
    _write_pdf(pages, pdf_path)
    return pages


def _engrave_svg(toolkit, musicxml_path: Path, svg_dir: Path) -> list[Path]:
    toolkit.setOptions(
        {
            "scale": 45,
            "pageWidth": 2200,
            "pageHeight": 2970,
            "pageMarginLeft": 90,
            "pageMarginRight": 90,
            "pageMarginTop": 140,
            "pageMarginBottom": 90,
            "adjustPageHeight": True,
            "header": "auto",
            "footer": "none",
            "spacingStaff": 14,
            "breaks": "auto",
            "svgViewBox": True,
        }
    )
    if not toolkit.loadFile(str(musicxml_path)):
        raise RuntimeError("The notation engraver could not read the score.")

    pages = []
    for number in range(1, toolkit.getPageCount() + 1):
        page_path = svg_dir / f"page-{number}.svg"
        page_path.write_text(toolkit.renderToSVG(number), encoding="utf-8")
        pages.append(page_path)
    return pages


def _write_pdf(pages: list[Path], pdf_path: Path) -> None:
    """Turn the engraved SVG pages into one PDF. This Verovio build has no PDF export."""
    import subprocess

    pdf_path = Path(pdf_path).resolve()
    if not pages:
        pdf_path.write_bytes(b"")
        return

    # macOS Quick Look can rasterize, but a vector PDF is better when a converter exists.
    converter = shutil.which("rsvg-convert")
    if converter:
        subprocess.run(
            [converter, "-f", "pdf", "-o", str(pdf_path), *[str(page) for page in pages]],
            check=True,
        )
        return

    html_path = pdf_path.with_suffix(".html")
    sheets = []
    for page in pages:
        sheets.append(page.read_text(encoding="utf-8"))
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{margin:0;background:#fff}svg{width:100%;height:auto;page-break-after:always}</style>"
        f"</head><body>{''.join(sheets)}</body></html>",
        encoding="utf-8",
    )
    chrome = next(
        (
            candidate
            for candidate in (
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
                shutil.which("google-chrome"),
                shutil.which("chromium"),
            )
            if candidate and Path(candidate).exists()
        ),
        None,
    )
    if chrome:
        subprocess.run(
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
        html_path.unlink(missing_ok=True)
        return

    html_path.unlink(missing_ok=True)
    raise RuntimeError("Could not write a PDF from the engraved pages.")
