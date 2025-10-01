# # SadTalker-Implementation/api.py
# import os
# import io
# import uuid
# import shutil
# import asyncio
# import tempfile
# import traceback
# from typing import Optional

# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
# from fastapi.middleware.cors import CORSMiddleware

# # SadTalker internals
# # Expecting repo layout:
# # SadTalker-Implementation/
# #   ├─ api.py  (this file)
# #   ├─ src/
# #   └─ checkpoints/
# import sys
# SADTALKER_ROOT = os.path.dirname(os.path.abspath(__file__))
# if SADTALKER_ROOT not in sys.path:
#     sys.path.append(SADTALKER_ROOT)

# from src.gradio_demo import SadTalker  # uses `from src...` imports internally

# # Optional helpers for audio conversion (mp3 → wav)
# # We'll try ffmpeg first; if not present, we fallback to librosa+soundfile
# def _ensure_wav(in_path: str) -> str:
#     """
#     Ensure audio is WAV (SadTalker prefers WAV). 
#     If input is already .wav, return it.
#     If not, try ffmpeg; if not available, fall back to librosa+soundfile.
#     """
#     ext = os.path.splitext(in_path)[1].lower()
#     if ext == ".wav":
#         return in_path

#     out_path = in_path + ".wav"

#     # Try ffmpeg (recommended)
#     try:
#         import subprocess
#         subprocess.run(
#             ["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", out_path],
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
#         )
#         return out_path
#     except Exception:
#         pass

#     # Fallback: librosa + soundfile
#     try:
#         import librosa, soundfile as sf
#         y, sr = librosa.load(in_path, sr=16000, mono=True)
#         sf.write(out_path, y, sr, subtype="PCM_16")
#         return out_path
#     except Exception as e:
#         raise RuntimeError(
#             f"Failed to convert audio to WAV. "
#             f"Tried ffmpeg and librosa+soundfile. Original error: {e}"
#         )

# # -----------------------------------------------------------------------------

# app = FastAPI(title="SadTalker API", description="Generate talking-head video from image + audio")

# # CORS (allow local Streamlit)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # lock down if you want
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # One global SadTalker instance
# SADTALKER: Optional[SadTalker] = None

# @app.on_event("startup")
# def _startup():
#     global SADTALKER
#     try:
#         checkpoints = os.path.join(SADTALKER_ROOT, "checkpoints")
#         config_path = os.path.join(SADTALKER_ROOT, "src", "config")
#         if not os.path.isdir(checkpoints):
#             raise RuntimeError("Missing checkpoints/ directory.")
#         if not os.path.isdir(config_path):
#             raise RuntimeError("Missing src/config directory.")

#         SADTALKER = SadTalker(checkpoint_path=checkpoints, config_path=config_path, lazy_load=True)
#         print("✅ SadTalker initialized.")
#     except Exception as e:
#         print("❌ SadTalker initialization failed:", e)
#         traceback.print_exc()
#         raise

# @app.get("/health", response_class=PlainTextResponse)
# def health():
#     return "ok"

# @app.post("/generate_video")
# async def generate_video(
#     image: UploadFile = File(..., description="Face image (jpg/png)"),
#     audio: Optional[UploadFile] = File(None, description="Speech audio (wav/mp3). If omitted, you must pass text."),
#     text: Optional[str] = Form(None, description="If provided (and audio is missing), server will TTS this text."),
#     pose_style: int = Form(0),
#     size_of_image: int = Form(256),
#     preprocess_type: str = Form("crop"),
#     still_mode: bool = Form(False),
#     enhancer: bool = Form(False),
#     batch_size: int = Form(1),
# ):
#     """
#     Generate a talking-head video. You must send either:
#       - image + audio, OR
#       - image + text (server will TTS)
#     Returns the MP4 bytes as a FileResponse.
#     """
#     try:
#         if SADTALKER is None:
#             raise HTTPException(status_code=500, detail="SadTalker not ready")

#         # Validate inputs
#         if image is None:
#             raise HTTPException(status_code=400, detail="Missing image file.")
#         if (audio is None) and (not text):
#             raise HTTPException(status_code=400, detail="Send either audio file or text for TTS.")

#         # Create temp workspace
#         workdir = os.path.join(SADTALKER_ROOT, "api_tmp")
#         os.makedirs(workdir, exist_ok=True)

#         # Save image
#         img_suffix = os.path.splitext(image.filename or "face.png")[1] or ".png"
#         img_path = os.path.join(workdir, f"{uuid.uuid4().hex}{img_suffix}")
#         with open(img_path, "wb") as f:
#             f.write(await image.read())

#         # Build audio
#         if audio is not None:
#             # Save uploaded audio
#             a_suffix = os.path.splitext(audio.filename or "audio")[1] or ".wav"
#             raw_audio_path = os.path.join(workdir, f"{uuid.uuid4().hex}{a_suffix}")
#             with open(raw_audio_path, "wb") as f:
#                 f.write(await audio.read())
#         else:
#             # TTS from text using gTTS (kept server-side for robustness)
#             # Note: gTTS outputs mp3; we'll convert to wav with _ensure_wav()
#             from gtts import gTTS
#             tmp_mp3 = os.path.join(workdir, f"{uuid.uuid4().hex}.mp3")
#             gTTS(text=text, lang="en").save(tmp_mp3)
#             raw_audio_path = tmp_mp3

#         # Ensure WAV (SadTalker pipelines generally prefer wav)
#         wav_path = _ensure_wav(raw_audio_path)

#         # Output path (SadTalker returns a path, but we want to control it)
#         out_path = os.path.join(workdir, f"{uuid.uuid4().hex}.mp4")

#         # Run SadTalker
#         # NOTE: gradio_demo.SadTalker.test expects:
#         #   (source_image, driven_audio, preprocess, still_mode, enhancer, batch_size, size_of_image, pose_style)
#         # If your local signature differs, adjust accordingly.
#         video_path = SADTALKER.test(
#             source_image=img_path,
#             driven_audio=wav_path,
#             preprocess=preprocess_type,
#             is_still_mode=still_mode if "is_still_mode" in SADTALKER.test.__code__.co_varnames else still_mode,
#             enhancer=enhancer,
#             batch_size=batch_size,
#             size_of_image=size_of_image,
#             pose_style=pose_style
#         )

#         # Some SadTalker builds already return the final path; if not, use out_path if saved by us
#         final_path = video_path if (video_path and os.path.exists(video_path)) else out_path
#         if not os.path.exists(final_path):
#             # If SadTalker wrote somewhere else, try to find any mp4 in workdir as a fallback
#             candidates = [os.path.join(workdir, f) for f in os.listdir(workdir) if f.lower().endswith(".mp4")]
#             if candidates:
#                 final_path = max(candidates, key=os.path.getmtime)

#         if not os.path.exists(final_path):
#             raise HTTPException(status_code=500, detail="SadTalker did not produce a video file.")

#         return FileResponse(final_path, media_type="video/mp4", filename="sadtalker_output.mp4")

#     except HTTPException:
#         raise
#     except Exception as e:
#         traceback.print_exc()
#         return JSONResponse(
#             status_code=500,
#             content={"error": str(e), "traceback": traceback.format_exc()}
#         )


# from fastapi import FastAPI, UploadFile, Form
# from fastapi.responses import FileResponse
# import os, sys, tempfile, uuid

# # SadTalker path setup
# SADTALKER_ROOT = os.path.dirname(os.path.abspath(__file__))
# if SADTALKER_ROOT not in sys.path:
#     sys.path.append(SADTALKER_ROOT)
# if os.path.join(SADTALKER_ROOT, "src") not in sys.path:
#     sys.path.append(os.path.join(SADTALKER_ROOT, "src"))

# from gradio_demo import SadTalker  # ✅ works with existing deps

# app = FastAPI(title="SadTalker API")

# # Lazy load
# _sadtalker = None
# def get_sadtalker():
#     global _sadtalker
#     if _sadtalker is None:
#         _sadtalker = SadTalker("checkpoints", "src/config", lazy_load=True)
#     return _sadtalker

# @app.post("/generate")
# async def generate_talking_head(
#     image: UploadFile,
#     audio: UploadFile,
#     preprocess: str = Form("crop"),
#     still_mode: bool = Form(False),
#     enhancer: bool = Form(False),
#     batch_size: int = Form(1),
#     size: int = Form(256),
#     pose_style: int = Form(0),
# ):
#     """Generate a talking-head video"""
#     try:
#         tmp_dir = tempfile.mkdtemp()
#         img_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.png")
#         audio_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.wav")
#         video_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.mp4")

#         with open(img_path, "wb") as f:
#             f.write(await image.read())
#         with open(audio_path, "wb") as f:
#             f.write(await audio.read())

#         sadtalker = get_sadtalker()
#         result = sadtalker.test(
#             source_image=img_path,
#             driven_audio=audio_path,
#             preprocess=preprocess,
#             still_mode=still_mode,
#             enhancer=enhancer,
#             batch_size=batch_size,
#             size=size,
#             pose_style=pose_style,
#         )

#         if isinstance(result, dict) and "video" in result:
#             return FileResponse(result["video"], media_type="video/mp4")

#         return {"error": "Video generation failed", "details": str(result)}

#     except Exception as e:
#         return {"error": str(e)}



# api.py
import os
import sys
import uuid
import shutil
import tempfile
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Ensure SadTalker src is importable ---
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    # gradio_demo.py sits under src/ and contains SadTalker class
    from gradio_demo import SadTalker
except Exception as e:
    raise RuntimeError(f"Failed to import SadTalker from src/gradio_demo.py: {e}")

# Optional server-side TTS (only used when 'audio' is not provided)
TTS_AVAILABLE = False
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False


app = FastAPI(title="SadTalker API", version="1.0.0")

# CORS (allow your Streamlit app to call this API from another port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded SadTalker instance
_sadtalker: Optional[SadTalker] = None


def get_sadtalker() -> SadTalker:
    global _sadtalker
    if _sadtalker is None:
        # Initialize once; uses the same paths as your app_sadtalker.py
        _sadtalker = SadTalker(
            checkpoint_path="checkpoints",
            config_path="src/config",
            lazy_load=True,
        )
    return _sadtalker


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"


@app.post("/generate_video")
async def generate_video(
    # NOTE: field names match your Streamlit code exactly
    image: UploadFile = File(...),
    audio: UploadFile | None = File(None),
    text: str | None = Form(None),

    pose_style: int = Form(0),
    size_of_image: int = Form(256),
    preprocess_type: str = Form("crop"),
    still_mode: bool = Form(False),
    enhancer: bool = Form(False),
    batch_size: int = Form(1),
):
    """
    Accepts an image (required) and either an audio file or fallback text.
    Calls SadTalker.test with positional args:
      (source_image, driven_audio, preprocess_type, is_still_mode,
       enhancer, batch_size, size_of_image, pose_style)
    Returns the generated MP4 bytes.
    """
    # Validate early
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Missing 'image' file."})

    if audio is None and (text is None or not text.strip()):
        return JSONResponse(
            status_code=400,
            content={"error": "Provide either 'audio' file or 'text' for server-side TTS."},
        )

    # Prepare temp workspace for this request
    workdir = os.path.join(tempfile.gettempdir(), f"sadtalker_api_{uuid.uuid4().hex}")
    os.makedirs(workdir, exist_ok=True)

    img_path = os.path.join(workdir, f"image{os.path.splitext(image.filename or 'img.png')[1] or '.png'}")
    aud_path = None  # set below if provided/created

    try:
        # Save image
        with open(img_path, "wb") as f:
            f.write(await image.read())

        # Save/prepare audio
        if audio is not None:
            # Use the uploaded audio as-is
            ext = os.path.splitext(audio.filename or "audio.mp3")[1] or ".mp3"
            aud_path = os.path.join(workdir, f"audio{ext}")
            with open(aud_path, "wb") as f:
                f.write(await audio.read())
        else:
            # No audio: synthesize from text (server-side TTS)
            if not TTS_AVAILABLE:
                return JSONResponse(
                    status_code=500,
                    content={"error": "No audio provided and server-side TTS (gTTS) not available."},
                )
            tts_mp3 = os.path.join(workdir, "tts.mp3")
            try:
                gTTS(text=text.strip(), lang="en").save(tts_mp3)
                aud_path = tts_mp3
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Server-side TTS failed: {e}"},
                )

        # Call SadTalker with positional args (MUST match your gradio wiring)
        st_model = get_sadtalker()
        result = st_model.test(
            img_path,           # source_image
            aud_path,           # driven_audio
            preprocess_type,    # preprocess_type
            still_mode,         # is_still_mode
            enhancer,           # enhancer
            batch_size,         # batch_size
            size_of_image,      # size_of_image
            pose_style,         # pose_style
        )

        # result is usually a path to the final mp4; handle list/tuple just in case
        if isinstance(result, (list, tuple)) and result:
            video_path = result[0]
        else:
            video_path = result

        if not isinstance(video_path, str) or not os.path.exists(video_path):
            return JSONResponse(
                status_code=500,
                content={"error": f"SadTalker returned invalid video path: {video_path!r}"},
            )

        # Return the MP4 file bytes
        # requests in Streamlit will treat this as resp.content bytes (what your code expects)
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=os.path.basename(video_path),
        )

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb},
        )
    finally:
        # Clean up temp files we created (image/audio); DO NOT remove the model's result path.
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7861, reload=True)
