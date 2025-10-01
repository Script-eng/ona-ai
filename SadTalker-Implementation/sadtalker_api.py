import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from src.gradio_demo import SadTalker

app = FastAPI()
sadtalker = SadTalker("checkpoints", "src/config", lazy_load=True)

@app.post("/talk")
async def talk(source_image: UploadFile = File(...), driven_audio: UploadFile = File(...)):
    """Generate talking head from image + audio."""
    try:
        # Save inputs
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_tmp:
            img_tmp.write(await source_image.read())
            img_path = img_tmp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_tmp:
            audio_tmp.write(await driven_audio.read())
            audio_path = audio_tmp.name

        # Run SadTalker
        video_path = sadtalker.test(
            source_image=img_path,
            driven_audio=audio_path,
            preprocess="crop",
            still=False,
            enhancer=False,
            batch_size=2,
            size=256,
            pose_style=0
        )

        return FileResponse(video_path, media_type="video/mp4", filename="output.mp4")

    except Exception as e:
        return {"error": str(e)}
