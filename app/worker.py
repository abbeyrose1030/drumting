"""Transcribe in a child process so an out-of-memory kill does not take the site down."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_meta(job_dir: Path, meta: dict) -> None:
    (job_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> int:
    job_dir = Path(sys.argv[1])
    audio_path = Path(sys.argv[2])
    rudiment = sys.argv[3] == "1"
    bpm = float(sys.argv[4]) if sys.argv[4] else None
    time_signature = sys.argv[5]
    use_click = sys.argv[6] == "1"
    meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
    try:
        from app.analyze import analyze
        from app.engrave import (
            build_score,
            count_instruments,
            render_pages,
            write_midi,
            write_musicxml,
        )

        result = analyze(
            str(audio_path),
            rudiment=rudiment,
            bpm=bpm,
            time_signature=time_signature,
            use_click=use_click,
        )
        (job_dir / "events.json").write_text(json.dumps(result["events"]), encoding="utf-8")
        meta.update({
            "tempo": result["tempo"],
            "click_found": result.get("click_found"),
            "clicks_removed": result.get("clicks_removed") or 0,
            "onset_count": result["onset_count"],
            "event_count": len(result["events"]),
            "duration_sec": result["duration_sec"],
        })
        print(f"  tempo {result['tempo']} BPM, {result['onset_count']} onsets, {len(result['events'])} events")
        events = result["events"]
        score, grid = build_score(
            events,
            bpm=float(meta["tempo"]),
            time_signature=meta["time_signature"],
            title=meta["title"],
            subdivision=int(meta.get("subdivision") or 16),
            shift_sixteenths=int(meta.get("shift_sixteenths") or 0),
            measures_per_system=int(meta.get("measures_per_system") or 4),
        )
        musicxml_path = job_dir / "score.musicxml"
        write_musicxml(score, musicxml_path)
        write_midi(grid["quantized"], grid["bpm"], job_dir / "score.mid")
        pages = render_pages(musicxml_path, job_dir / "pages", job_dir / "score.pdf")
        meta["page_count"] = len(pages)
        meta["measure_count"] = len(grid["measures"])
        meta["dropped"] = grid["dropped"]
        meta["counts"] = count_instruments(events)
        meta["status"] = "ready"
        _write_meta(job_dir, meta)
        return 0
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        meta["status"] = "error"
        meta["error"] = str(exc)
        _write_meta(job_dir, meta)
        return 1


if __name__ == "__main__":
    sys.exit(main())
