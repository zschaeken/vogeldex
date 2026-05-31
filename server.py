"""
VogelDex BirdNET API server
----------------------------
Local:      uvicorn server:app --port 8765 --reload
Production: uvicorn server:app --host 0.0.0.0 --port $PORT
"""
import datetime, os, tempfile
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from pydub import AudioSegment

# ── ffmpeg: prefer local binary, fall back to PATH ───────────────────────────
_here   = os.path.dirname(os.path.abspath(__file__))
_ffmpeg = os.path.join(_here, "ffmpeg.exe")
if os.path.exists(_ffmpeg):
    AudioSegment.converter = _ffmpeg
    AudioSegment.ffmpeg    = _ffmpeg

# ── CORS: allow all origins in dev, restrict to frontend URL in prod ─────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="VogelDex BirdNET API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Load BirdNET model once at startup ───────────────────────────────────────
print("Loading BirdNET model…")
analyzer = Analyzer()
print("BirdNET ready ✓")


def day_to_week48(day_of_year: int) -> int:
    """Convert day-of-year (1–366) to BirdNET week scale (1–48)."""
    return max(1, min(48, round(day_of_year / 365 * 48)))


@app.get("/health")
async def health():
    return {"status": "ok", "model": "BirdNET-Analyzer"}


@app.post("/identify")
async def identify(
    audio: UploadFile = File(...),
    lat:   float = Query(50.85, description="Latitude (default: Brussels)"),
    lon:   float = Query(4.35,  description="Longitude (default: Brussels)"),
):
    week_48 = day_to_week48(datetime.date.today().timetuple().tm_yday)
    suffix  = os.path.splitext(audio.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    wav_path = tmp_path
    try:
        # Convert browser audio (webm/ogg/mp4) → 48kHz mono WAV for BirdNET
        if suffix in (".webm", ".ogg", ".mp4", ".m4a"):
            wav_path = tmp_path.replace(suffix, ".wav")
            try:
                seg = AudioSegment.from_file(tmp_path)
                seg.set_frame_rate(48000).set_channels(1).export(wav_path, format="wav")
            except Exception as e:
                print(f"[ffmpeg] conversion failed ({e}), trying raw file")
                wav_path = tmp_path

        # Run BirdNET
        recording = Recording(
            analyzer, wav_path,
            lat=lat, lon=lon,
            week_48=week_48,
            min_conf=0.05,
            overlap=1.5,
        )
        recording.analyze()
        print(f"[BirdNET] {len(recording.detections)} detections  "
              f"lat={lat:.2f} lon={lon:.2f} week48={week_48}")

        # Aggregate: max confidence + detection count per species
        best: dict[str, dict] = {}
        for det in recording.detections:
            sci  = det["scientific_name"]
            conf = det["confidence"]
            if sci not in best or conf > best[sci]["confidence"]:
                best[sci] = {
                    "scientific_name": sci,
                    "common_name":     det["common_name"],
                    "confidence":      conf,
                    "detections":      0,
                }
            best[sci]["detections"] += 1

        top = sorted(best.values(),
                     key=lambda x: (x["confidence"], x["detections"]),
                     reverse=True)[:6]

        for r in top:
            print(f"  {r['confidence']:.2f}  {r['common_name']}  "
                  f"({r['detections']} windows)")

        return {
            "results": [{
                "label":           f"{r['scientific_name']}_{r['common_name']}",
                "scientific_name": r["scientific_name"],
                "common_name":     r["common_name"],
                "confidence":      round(r["confidence"], 3),
                "detections":      r["detections"],
            } for r in top],
            "meta": {"week_48": week_48, "lat": lat, "lon": lon},
        }

    finally:
        if os.path.exists(tmp_path):  os.unlink(tmp_path)
        if wav_path != tmp_path and os.path.exists(wav_path): os.unlink(wav_path)
