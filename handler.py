import os
import cv2
import uuid
import base64
import runpod
import subprocess
import requests
import numpy as np
import insightface
from gfpgan import GFPGANer

# 1. Load models into GPU memory at container startup
print("Initializing InsightFace...")
app = insightface.app.FaceAnalysis(name="buffalo_l", root="/root/.insightface")
app.prepare(ctx_id=0, det_size=(640, 640))

print("Loading Inswapper ONNX...")
swapper = insightface.model_zoo.get_model(
    "/app/checkpoints/inswapper_128.onnx",
    download=False,
    provider_options=[{"device_id": 0}]
)

print("Loading GFPGAN Restorer...")
gfpgan = GFPGANer(
    model_path="/app/checkpoints/GFPGANv1.4.pth",
    upscale=1,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None
)

def fetch_image_from_url(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def match_face_by_embedding(target_faces, ref_embedding, threshold=0.45):
    best_match = None
    max_sim = -1.0
    for face in target_faces:
        sim = float(np.dot(face.normed_embedding, ref_embedding))
        if sim > max_sim:
            max_sim = sim
            best_match = face
    return best_match if max_sim >= threshold else None

def swap_single_frame(frame, mappings, pipeline="fast"):
    detected_faces = app.get(frame)
    if not detected_faces:
        return frame

    # Sort faces left-to-right
    detected_faces.sort(key=lambda x: x.bbox[0])

    for m in mappings:
        src_face = m["source_face"]
        target_face = None

        if "ref_embedding" in m and m["ref_embedding"] is not None:
            target_face = match_face_by_embedding(detected_faces, m["ref_embedding"])
        elif "target_index" in m and m["target_index"] < len(detected_faces):
            target_face = detected_faces[m["target_index"]]

        if target_face is not None:
            frame = swapper.get(frame, target_face, src_face, paste_back=True)
            if pipeline in ["gfpgan", "codeformer"]:
                _, _, frame = gfpgan.enhance(frame, has_aligned=False, only_center_face=True, paste_back=True)

    return frame

def handler(event):
    job_input = event.get("input", {})
    media_type = job_input.get("type", "video")  # "image" or "video"
    pipeline = job_input.get("pipeline", "fast")  # "fast", "gfpgan", "codeformer"
    pairs = job_input.get("pairs", [])
    target_url = job_input.get("target_url")

    if not target_url or not pairs:
        return {"error": "Missing 'target_url' or 'pairs' list in input."}

    # Prepare face embeddings
    active_mappings = []
    for p in pairs:
        src_img = fetch_image_from_url(p["source_url"])
        src_faces = app.get(src_img)
        if not src_faces:
            continue
        mapping = {"source_face": src_faces[0]}

        if "reference_target_url" in p and p["reference_target_url"]:
            ref_img = fetch_image_from_url(p["reference_target_url"])
            ref_faces = app.get(ref_img)
            if ref_faces:
                mapping["ref_embedding"] = ref_faces[0].normed_embedding
        elif "target_index" in p:
            mapping["target_index"] = int(p["target_index"])
        else:
            mapping["target_index"] = 0

        active_mappings.append(mapping)

    temp_id = uuid.uuid4().hex[:8]

    # --- Mode 1 & 2: Single or Multiple Person Photo Swap ---
    if media_type == "image":
        target_img = fetch_image_from_url(target_url)
        result = swap_single_frame(target_img, active_mappings, pipeline)
        _, buffer = cv2.imencode(".jpg", result)
        return {
            "status": "success",
            "media_type": "image",
            "base64": base64.b64encode(buffer).decode("utf-8")
        }

    # --- Mode 3 & 4: Single or Multiple Person Video Swap ---
    tmp_in = f"/tmp/in_{temp_id}.mp4"
    tmp_swapped = f"/tmp/swapped_{temp_id}.mp4"
    tmp_out = f"/tmp/out_{temp_id}.mp4"

    try:
        with open(tmp_in, "wb") as f:
            f.write(requests.get(target_url, timeout=60).content)

        cap = cv2.VideoCapture(tmp_in)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(tmp_swapped, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = swap_single_frame(frame, active_mappings, pipeline)
            writer.write(frame)

        cap.release()
        writer.release()

        # Remux original audio
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_swapped, "-i", tmp_in,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0?", "-shortest", tmp_out
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        with open(tmp_out, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return {"status": "success", "media_type": "video", "base64": encoded}

    finally:
        for p in [tmp_in, tmp_swapped, tmp_out]:
            if os.path.exists(p):
                os.remove(p)

runpod.serverless.start({"handler": handler})
