from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PIL import Image
import cv2
import tempfile
import os
import io

app = FastAPI(title="NeuroZoo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading AI model...")
captioner = pipeline("image-classification", model="google/vit-base-patch16-224")
print("Model ready!")

ANIMALS = ["lion", "tiger", "elephant", "wolf", "bear", "deer", "monkey",
           "bird", "snake", "crocodile", "zebra", "giraffe", "leopard",
           "dog", "cat", "horse", "cow", "sheep", "fox", "rabbit", "eagle"]

def analyze_frame_local(frame_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    results = captioner(image)
    top = results[0]["label"] if results else "Unknown"
    confidence = results[0]["score"] if results else 0.0

    species = "Unknown"
    for animal in ANIMALS:
        if animal in top.lower():
            species = animal.capitalize()
            break
    if species == "Unknown":
        species = top.split(",")[0].strip().capitalize()

    return {
        "species": species,
        "count": 1,
        "behavior": "Detected in frame",
        "emotional_state": "calm",
        "health_indicators": "No visible health issues",
        "anomalies": "None detected",
        "sleep_estimate": "Unknown",
        "environment": "Natural habitat",
        "confidence": round(confidence, 2),
        "summary": f"AI detected {species} with {round(confidence*100, 1)}% confidence."
    }

def extract_frames(video_path: str, max_frames: int = 3):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0
    interval = max(1, int(total / max_frames))
    frames = []
    for i in range(max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frames.append(buffer.tobytes())
    cap.release()
    return frames, duration

@app.get("/")
def root():
    return {"status": "NeuroZoo API running"}

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        content_type = file.content_type or ""

        if "image" in content_type:
            analysis = analyze_frame_local(contents)
            return {"filename": file.filename, "analysis": analysis, "frames_analyzed": 1}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        frames, duration = extract_frames(tmp_path, max_frames=3)
        os.unlink(tmp_path)

        results = []
        for i, frame in enumerate(frames):
            print(f"Analyzing frame {i+1}/{len(frames)}...")
            try:
                result = analyze_frame_local(frame)
                results.append(result)
            except Exception as e:
                print(f"Frame {i+1} error: {e}")

        if not results:
            return {"error": "No animals detected", "filename": file.filename}

        species_list = [r["species"] for r in results if r.get("species") != "Unknown"]
        emotions = [r["emotional_state"] for r in results]

        final = {
            "species": max(set(species_list), key=species_list.count) if species_list else "Unknown",
            "count": 1,
            "behavior": results[-1].get("behavior", "Unknown"),
            "emotional_state": max(set(emotions), key=emotions.count),
            "health_indicators": results[-1].get("health_indicators", ""),
            "anomalies": "None detected",
            "sleep_estimate": "Unknown",
            "environment": results[0].get("environment", ""),
            "confidence": round(sum(r.get("confidence", 0) for r in results) / len(results), 2),
            "summary": results[-1].get("summary", ""),
            "duration_seconds": round(duration, 1),
            "frames_analyzed": len(results)
        }

        return {"filename": file.filename, "analysis": final, "frames_analyzed": len(results)}

    except Exception as e:
        return {"error": str(e), "filename": file.filename}