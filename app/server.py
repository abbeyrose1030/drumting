"""Drum-notation desk. Locally this binds to this machine only."""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
import uuid
from datetime import datetime
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analyze import analyze
from app.engrave import DISPLAY_NAME, build_score, count_instruments, render_pages, write_midi, write_musicxml

STATIC = Path(__file__).resolve().parent / "static"
OUTPUTS = ROOT / "outputs"
PORT = int(os.environ.get("PORT", "8765"))
HOST = os.environ.get("HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
JOB_RE = re.compile(r"^[A-Za-z0-9_-]+$")
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a"}


def _job_dir(job_id: str) -> Path:
    if not JOB_RE.match(job_id):
        raise ValueError("Unknown transcription.")
    path = (OUTPUTS / job_id).resolve()
    if OUTPUTS.resolve() not in path.parents:
        raise ValueError("Unknown transcription.")
    if not path.is_dir():
        raise FileNotFoundError("That transcription is no longer on disk.")
    return path


def _read_meta(job_id: str) -> dict:
    path = _job_dir(job_id) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_meta(job_dir: Path, meta: dict) -> None:
    (job_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _page_count(job_dir: Path) -> int:
    return len(list((job_dir / "pages").glob("page-*.svg")))


def engrave_job(job_dir: Path, meta: dict) -> dict:
    events = json.loads((job_dir / "events.json").read_text(encoding="utf-8"))
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
    _write_meta(job_dir, meta)
    return meta


def _public_job(meta: dict) -> dict:
    counts = meta.get("counts") or {}
    return {
        "id": meta["id"],
        "title": meta["title"],
        "filename": meta.get("filename") or "",
        "tempo": meta["tempo"],
        "time_signature": meta["time_signature"],
        "subdivision": meta.get("subdivision") or 16,
        "shift_sixteenths": meta.get("shift_sixteenths") or 0,
        "measures_per_system": meta.get("measures_per_system") or 4,
        "page_count": meta.get("page_count") or 0,
        "measure_count": meta.get("measure_count") or 0,
        "onset_count": meta.get("onset_count") or 0,
        "event_count": meta.get("event_count") or 0,
        "duration_sec": meta.get("duration_sec") or 0,
        "dropped": meta.get("dropped") or 0,
        "rudiment": bool(meta.get("rudiment")),
        "click_found": bool(meta.get("click_found")),
        "clicks_removed": meta.get("clicks_removed") or 0,
        "created": meta.get("created"),
        "counts": [
            {"id": name, "name": DISPLAY_NAME.get(name, name), "count": count}
            for name, count in counts.items()
        ],
    }


def _parse_bpm(raw) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        bpm = float(text)
    except ValueError as exc:
        raise ValueError("BPM needs to be a number.") from exc
    if bpm < 30 or bpm > 320:
        raise ValueError("BPM needs to be between 30 and 320.")
    return bpm


def list_jobs() -> list[dict]:
    if not OUTPUTS.exists():
        return []
    found = []
    for meta_path in OUTPUTS.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("id"):
            found.append(_public_job(meta))
    found.sort(key=lambda item: item.get("created") or "", reverse=True)
    return found[:20]


class Handler(BaseHTTPRequestHandler):
    server_version = "DrumNotation/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str, filename: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path in ("/", "/index.html"):
                page = (STATIC / "index.html").read_bytes()
                self._send(200, page, "text/html; charset=utf-8")
                return
            if path == "/api/jobs":
                self._json(200, {"jobs": list_jobs()})
                return
            parts = [part for part in path.split("/") if part]
            if len(parts) == 3 and parts[0] == "api" and parts[1] == "jobs":
                self._json(200, {"job": _public_job(_read_meta(parts[2]))})
                return
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs" and parts[3].startswith("page-"):
                job_dir = _job_dir(parts[2])
                page_name = Path(parts[3]).name
                if not re.fullmatch(r"page-\d+\.svg", page_name):
                    raise FileNotFoundError(page_name)
                svg = (job_dir / "pages" / page_name).read_bytes()
                self._send(200, svg, "image/svg+xml")
                return
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs" and parts[3] in {"score.pdf", "score.mid", "score.musicxml"}:
                job_dir = _job_dir(parts[2])
                meta = _read_meta(parts[2])
                target = job_dir / parts[3]
                stem = re.sub(r"[^A-Za-z0-9._-]+", "-", meta.get("title") or "drum-notation").strip("-") or "drum-notation"
                download_name = f"{stem}{target.suffix}"
                media = {
                    ".pdf": "application/pdf",
                    ".mid": "audio/midi",
                    ".musicxml": "application/vnd.recordare.musicxml+xml",
                }[target.suffix]
                self._send(200, target.read_bytes(), media, download_name)
                return
            self._json(404, {"error": "Not found."})
        except FileNotFoundError:
            self._json(404, {"error": "That file is not on disk."})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/transcribe":
                self._transcribe()
                return
            parts = [part for part in path.split("/") if part]
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs" and parts[3] == "engrave":
                self._reengrave(parts[2])
                return
            self._json(404, {"error": "Not found."})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": str(exc)})

    def _body_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _multipart(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        header = (
            f"Content-Type: {self.headers.get('Content-Type', '')}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8")
        message = BytesParser(policy=email_policy).parsebytes(header + body)
        fields: dict[str, str] = {}
        files: dict[str, tuple[str, bytes]] = {}
        parts = list(message.iter_parts()) or ([message] if message.get_content_disposition() else [])
        for part in parts:
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename:
                files[name] = (filename, payload)
            else:
                fields[name] = payload.decode("utf-8", errors="replace")
        return fields, files

    def _transcribe(self) -> None:
        fields, files = self._multipart()
        upload = files.get("audio")
        if upload is None or not upload[0] or not upload[1]:
            raise ValueError("Choose an audio file first.")
        filename, audio_bytes = upload
        suffix = Path(filename).suffix.lower()
        if suffix not in AUDIO_SUFFIXES:
            raise ValueError("Use a wav, mp3, flac, aiff, ogg, or m4a drum stem.")

        job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        job_dir = OUTPUTS / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        audio_path = job_dir / f"source{suffix}"
        audio_path.write_bytes(audio_bytes)

        title = str(fields.get("title") or Path(safe_name).stem).strip() or "Drum Notation"
        time_signature = str(fields.get("time_signature") or "4/4")
        rudiment = str(fields.get("rudiment") or "").lower() in {"1", "true", "on", "yes"}
        use_click = str(fields.get("has_click") or "").lower() in {"1", "true", "on", "yes"}
        bpm = _parse_bpm(fields.get("bpm"))
        print(f"Transcribing {safe_name} ({time_signature}, bpm={bpm}, click={use_click}, rudiment={rudiment})")
        result = analyze(
            str(audio_path),
            rudiment=rudiment,
            bpm=bpm,
            time_signature=time_signature,
            use_click=use_click,
        )
        (job_dir / "events.json").write_text(json.dumps(result["events"]), encoding="utf-8")

        meta = {
            "id": job_id,
            "title": title[:120],
            "filename": safe_name,
            "tempo": result["tempo"],
            "time_signature": time_signature,
            "subdivision": 16,
            "shift_sixteenths": 0,
            "measures_per_system": 4,
            "rudiment": rudiment,
            "click_found": result.get("click_found"),
            "clicks_removed": result.get("clicks_removed") or 0,
            "onset_count": result["onset_count"],
            "event_count": len(result["events"]),
            "duration_sec": result["duration_sec"],
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        print(f"  tempo {result['tempo']} BPM, {result['onset_count']} onsets, {len(result['events'])} events")
        _write_meta(job_dir, meta)
        meta = engrave_job(job_dir, meta)
        self._json(200, {"job": _public_job(meta)})

    def _reengrave(self, job_id: str) -> None:
        job_dir = _job_dir(job_id)
        meta = _read_meta(job_id)
        payload = self._body_json()
        if "tempo" in payload:
            tempo = float(payload["tempo"])
            if tempo < 30 or tempo > 320:
                raise ValueError("Tempo needs to be between 30 and 320 BPM.")
            meta["tempo"] = round(tempo, 2)
        if payload.get("time_signature"):
            meta["time_signature"] = str(payload["time_signature"])
        if "subdivision" in payload:
            subdivision = int(payload["subdivision"])
            if subdivision not in (8, 16):
                raise ValueError("Grid must be 8 or 16.")
            meta["subdivision"] = subdivision
        if "shift_sixteenths" in payload:
            shift = int(payload["shift_sixteenths"])
            if abs(shift) > 32:
                raise ValueError("Nudge stays within 32 sixteenth notes.")
            meta["shift_sixteenths"] = shift
        if "measures_per_system" in payload:
            per_system = int(payload["measures_per_system"])
            if per_system not in (2, 4):
                raise ValueError("Use 2 or 4 bars on a system.")
            meta["measures_per_system"] = per_system
        if payload.get("title"):
            meta["title"] = str(payload["title"]).strip()[:120] or meta["title"]
        print(f"Redrawing {job_id} at {meta['tempo']} BPM, {meta['time_signature']}")
        meta = engrave_job(job_dir, meta)
        self._json(200, {"job": _public_job(meta)})


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Drum Notation is running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
