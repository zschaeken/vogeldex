"""
VogelDex BirdNET + Claude API server
--------------------------------------
Local:      uvicorn server:app --port 8765 --reload
Production: uvicorn server:app --host 0.0.0.0 --port $PORT
"""
import datetime, os, tempfile, httpx
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from pydub import AudioSegment

# ── ffmpeg: prefer local binary, fall back to PATH ───────────────────────────
_here   = os.path.dirname(os.path.abspath(__file__))
_ffmpeg = os.path.join(_here, "ffmpeg.exe")
if os.path.exists(_ffmpeg):
    AudioSegment.converter = _ffmpeg
    AudioSegment.ffmpeg    = _ffmpeg

# ── Config ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ALLOWED_ORIGINS   = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="VogelDex API", version="1.0.0")
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
    return max(1, min(48, round(day_of_year / 365 * 48)))


# ════════════════════════════════════════════════════════════════
#  HEALTH
# ════════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    return {"status": "ok", "model": "BirdNET-Analyzer"}


# ════════════════════════════════════════════════════════════════
#  BIRDNET — audio identification
# ════════════════════════════════════════════════════════════════
@app.post("/identify")
async def identify(
    audio: UploadFile = File(...),
    lat:   float = Query(50.85),
    lon:   float = Query(4.35),
):
    week_48 = day_to_week48(datetime.date.today().timetuple().tm_yday)
    suffix  = os.path.splitext(audio.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    wav_path = tmp_path
    try:
        if suffix in (".webm", ".ogg", ".mp4", ".m4a"):
            wav_path = tmp_path.replace(suffix, ".wav")
            try:
                seg = AudioSegment.from_file(tmp_path)
                seg.set_frame_rate(48000).set_channels(1).export(wav_path, format="wav")
            except Exception as e:
                print(f"[ffmpeg] conversion failed ({e}), trying raw file")
                wav_path = tmp_path

        recording = Recording(
            analyzer, wav_path,
            lat=lat, lon=lon,
            week_48=week_48,
            min_conf=0.05,
            overlap=1.5,
        )
        recording.analyze()
        print(f"[BirdNET] {len(recording.detections)} detections  lat={lat:.2f} lon={lon:.2f}")

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


# ════════════════════════════════════════════════════════════════
#  CLAUDE — proxied so API key stays server-side
# ════════════════════════════════════════════════════════════════
class ClaudeRequest(BaseModel):
    system: str
    user:   str

@app.post("/claude")
async def claude_proxy(req: ClaudeRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured on server")

    payload = {
        "model":      "claude-sonnet-4-20250514",
        "max_tokens": 900,
        "system":     req.system,
        "messages":   [{"role": "user", "content": req.user}],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json=payload,
        )

    if not resp.is_success:
        raise HTTPException(resp.status_code, resp.text)

    return resp.json()
