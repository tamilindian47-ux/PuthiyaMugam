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
from sklearn.cluster import DBSCAN

# Warm load InsightFace and GFPGAN
app = insightface.app.FaceAnalysis(name="buffalo_l", root="/root/.insightface")
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model(
    "/app/checkpoints/inswapper_128.onnx",
    download=False,
    provider_options=[{"device_id": 0}]
)

gfpgan = GFPGANer(
    model_path="/app/checkpoints/GFPGANv1.4.pth",
    upscale=1,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None
)

def resolve_input_to_file(data_or_url, out_path):
    """Handles both URL and Base64 uploads."""
    if data_or_url.startswith("http://") or data_or_url.startswith("https://"):
        resp = requests.get(data_or_url, timeout=60)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
    else:
        # Base64 string
        if "," in data_or_url:
            data_or_url = data_or_url.split(",", 1)[1]
        raw_bytes = base64.b64decode(data_or_url)
        with open(out_path, "wb") as f:
            f.write(raw_bytes)

def decode_image(data_or_url):
    temp = f"/tmp/img_{uuid.uuid4().hex[:6]}.jpg"
    try:
        resolve_input_to_file(data_or_url, temp)
        return cv2.imread(temp)
    finally:
        if os.path.exists(temp):
            os.remove(temp)

# --- ACTION 1: DETECT & CLUSTER UNIQUE PEOPLE IN VIDEO/IMAGE ---
def detect_unique_people(media_path, is_video=True):
    detections = []  # list of dicts: {'crop': img, 'embedding': vec, 'score': area}

    if not is_video:
        frame = cv2.imread(media_path)
        faces = app.get(frame)
        for f in faces:
            box = f.bbox.astype(int)
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
            crop = frame[y1:y2, x1:x2]
            detections.append({'crop': crop, 'embedding': f.normed_embedding, 'score': (x2-x1)*(y2-y1)})
    else:
        cap = cv2.VideoCapture(media_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        step = max(1, total_frames // 30)  # sample ~30 representative frames across video
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                faces = app.get(frame)
                for f in faces:
                    box = f.bbox.astype(int)
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                    if (x2 - x1) > 50 and (y2 - y1) > 50:
                        crop = frame[y1:y2, x1:x2]
                        detections.append({'crop': crop, 'embedding': f.normed_embedding, 'score': (x2-x1)*(y2-y1)})
            frame_idx += 1
        cap.release()

    if not detections:
        return []

    # Cluster identities using DBSCAN on normalized cosine distance
    embeddings = np.array([d['embedding'] for d in detections])
    # metric='cosine', eps=0.45 (same person usually < 0.45 cosine distance)
    clustering = DBSCAN(eps=0.42, min_samples=1, metric='cosine').fit(embeddings)
    labels = clustering.labels_

    unique_characters = []
    unique_labels = [l for l in set(labels) if l != -1]
    
    for l in unique_labels:
        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == l]
        # Best crop has largest area/highest detection quality
        best_idx = max(cluster_indices, key=lambda idx: detections[idx]['score'])
        # Centroid identity embedding
        cluster_embeddings = embeddings[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        # Encode thumbnail to JPEG base64
        _, buf = cv2.imencode(".jpg", detections[best_idx]['crop'])
        thumb_b64 = base64.b64encode(buf).decode("utf-8")

        unique_characters.append({
            "character_id": int(l),
            "thumbnail_base64": thumb_b64,
            "embedding": centroid.tolist()
        })

    return unique_characters

# --- ACTION 2: PERFORM PERSISTENT FACE SWAP ---
def execute_swap(media_path, is_video, swap_targets, pipeline="fast"):
    """
    swap_targets schema:
    [
      { "target_embedding": [...512 floats...], "source_face": FaceObj }
    ]
    """
    if not is_video:
        frame = cv2.imread(media_path)
        faces = app.get(frame)
        for face in faces:
            for target in swap_targets:
                sim = float(np.dot(face.normed_embedding, target["target_embedding"]))
                if sim >= 0.40:
                    frame = swapper.get(frame, face, target["source_face"], paste_back=True)
                    if pipeline in ["gfpgan", "codeformer"]:
                        _, _, frame = gfpgan.enhance(frame, has_aligned=False, only_center_face=True, paste_back=True)
                    break
        _, out_buf = cv2.imencode(".jpg", frame)
        return {"status": "success", "media_type": "image", "base64": base64.b64encode(out_buf).decode("utf-8")}

    # Video Mode
    temp_id = uuid.uuid4().hex[:8]
    tmp_swapped = f"/tmp/swap_{temp_id}.mp4"
    tmp_final = f"/tmp/final_{temp_id}.mp4"

    cap = cv2.VideoCapture(media_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(tmp_swapped, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        for face in faces:
            for target in swap_targets:
                # Cosine similarity matching
                sim = float(np.dot(face.normed_embedding, target["target_embedding"]))
                if sim >= 0.40:
                    frame = swapper.get(frame, face, target["source_face"], paste_back=True)
                    if pipeline in ["gfpgan", "codeformer"]:
                        _, _, frame = gfpgan.enhance(frame, has_aligned=False, only_center_face=True, paste_back=True)
                    break
        writer.write(frame)

    cap.release()
    writer.release()

    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_swapped, "-i", media_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0?", "-shortest", tmp_final
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    with open(tmp_final, "rb") as f:
        vid_b64 = base64.b64encode(f.read()).decode("utf-8")

    for fpath in [tmp_swapped, tmp_final]:
        if os.path.exists(fpath):
            os.remove(fpath)

    return {"status": "success", "media_type": "video", "base64": vid_b64}

# --- RUNPOD MAIN ROUTER ---
def handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action", "swap")  # "detect" or "swap"
    media_data = job_input.get("media")  # URL or base64 string
    is_video = job_input.get("is_video", True)

    if not media_data:
        return {"error": "Missing 'media' payload."}

    req_id = uuid.uuid4().hex[:8]
    ext = "mp4" if is_video else "jpg"
    tmp_media = f"/tmp/input_{req_id}.{ext}"

    try:
        resolve_input_to_file(media_data, tmp_media)

        # 1. Step 1: Detect unique persons
        if action == "detect":
            characters = detect_unique_people(tmp_media, is_video=is_video)
            return {"status": "success", "characters": characters}

        # 2. Step 2: Swap matched identities
        elif action == "swap":
            pipeline = job_input.get("pipeline", "fast")
            swaps = job_input.get("swaps", [])  # list of { target_embedding: [...], new_face: url/b64 }

            swap_targets = []
            for s in swaps:
                src_img = decode_image(s["new_face"])
                src_faces = app.get(src_img)
                if src_faces:
                    swap_targets.append({
                        "target_embedding": np.array(s["target_embedding"], dtype=np.float32),
                        "source_face": src_faces[0]
                    })

            return execute_swap(tmp_media, is_video, swap_targets, pipeline=pipeline)

        return {"error": f"Unknown action '{action}'"}

    except Exception as e:
        return {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(tmp_media):
            os.remove(tmp_media)

runpod.serverless.start({"handler": handler})
