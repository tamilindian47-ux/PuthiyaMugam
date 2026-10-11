import os
import sys
import cv2
import uuid
import base64
import traceback
import subprocess
import requests
import numpy as np
import runpod

print("=== Starting Worker Initialization ===")
sys.stdout.flush()

try:
    import insightface
    from gfpgan import GFPGANer

    # Detect execution providers safely
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    print(f"Available ONNX providers: {available_providers}")
    
    selected_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
    print(f"Using providers: {selected_providers}")

    # 1. Warm load detector
    print("Loading buffalo_l face detector...")
    face_app = insightface.app.FaceAnalysis(name="buffalo_l", root="/root/.insightface", providers=selected_providers)
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # 2. Warm load Inswapper
    print("Loading Inswapper ONNX model...")
    model_path = "/app/checkpoints/inswapper_128.onnx"
    try:
        from insightface.model_zoo.inswapper import INSwapper
        swapper = INSwapper(model_file=model_path, session_options=None)
    except Exception as e:
        print(f"Loading via model_zoo fallback: {e}")
        swapper = insightface.model_zoo.get_model(model_path, download=False)

    # 3. Warm load GFPGAN
    print("Loading GFPGAN Restorer...")
    gfpgan = GFPGANer(
        model_path="/app/checkpoints/GFPGANv1.4.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None
    )
    print("=== All Models Loaded Successfully! ===")
    sys.stdout.flush()

except Exception as init_err:
    print("FATAL CONTAINER INITIALIZATION ERROR:")
    traceback.print_exc()
    sys.stdout.flush()
    raise init_err


def resolve_input_to_file(data_or_url, out_path):
    if data_or_url.startswith("http://") or data_or_url.startswith("https://"):
        resp = requests.get(data_or_url, timeout=60)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
    else:
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


# Pure NumPy Identity Clustering (No external sklearn dependency)
def cluster_embeddings_numpy(detections, threshold=0.42):
    clusters = []
    for i, d in enumerate(detections):
        vec = d['embedding']
        matched = False
        for c in clusters:
            centroid = np.mean([detections[idx]['embedding'] for idx in c], axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            if float(np.dot(vec, centroid)) >= (1.0 - threshold):
                c.append(i)
                matched = True
                break
        if not matched:
            clusters.append([i])

    unique_characters = []
    for l, c_indices in enumerate(clusters):
        best_idx = max(c_indices, key=lambda idx: detections[idx]['score'])
        cluster_vecs = [detections[idx]['embedding'] for idx in c_indices]
        centroid = np.mean(cluster_vecs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        _, buf = cv2.imencode(".jpg", detections[best_idx]['crop'])
        thumb_b64 = base64.b64encode(buf).decode("utf-8")

        unique_characters.append({
            "character_id": int(l),
            "thumbnail_base64": thumb_b64,
            "embedding": centroid.tolist()
        })
    return unique_characters


def detect_unique_people(media_path, is_video=True):
    detections = []
    if not is_video:
        frame = cv2.imread(media_path)
        faces = face_app.get(frame)
        for f in faces:
            box = f.bbox.astype(int)
            x1, y1 = max(0, box[0]), max(0, box[1])
            x2, y2 = min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
            crop = frame[y1:y2, x1:x2]
            detections.append({'crop': crop, 'embedding': f.normed_embedding, 'score': (x2-x1)*(y2-y1)})
    else:
        cap = cv2.VideoCapture(media_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        step = max(1, total_frames // 30)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                faces = face_app.get(frame)
                for f in faces:
                    box = f.bbox.astype(int)
                    x1, y1 = max(0, box[0]), max(0, box[1])
                    x2, y2 = min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                    if (x2 - x1) > 40 and (y2 - y1) > 40:
                        crop = frame[y1:y2, x1:x2]
                        detections.append({'crop': crop, 'embedding': f.normed_embedding, 'score': (x2-x1)*(y2-y1)})
            frame_idx += 1
        cap.release()

    if not detections:
        return []

    return cluster_embeddings_numpy(detections)


def execute_swap(media_path, is_video, swap_targets, pipeline="fast"):
    if not is_video:
        frame = cv2.imread(media_path)
        faces = face_app.get(frame)
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
        faces = face_app.get(frame)
        for face in faces:
            for target in swap_targets:
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


def handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action", "swap")
    media_data = job_input.get("media")
    is_video = job_input.get("is_video", True)

    if not media_data:
        return {"error": "Missing 'media' payload."}

    req_id = uuid.uuid4().hex[:8]
    ext = "mp4" if is_video else "jpg"
    tmp_media = f"/tmp/input_{req_id}.{ext}"

    try:
        resolve_input_to_file(media_data, tmp_media)

        if action == "detect":
            characters = detect_unique_people(tmp_media, is_video=is_video)
            return {"status": "success", "characters": characters}

        elif action == "swap":
            pipeline = job_input.get("pipeline", "fast")
            swaps = job_input.get("swaps", [])

            swap_targets = []
            for s in swaps:
                src_img = decode_image(s["new_face"])
                src_faces = face_app.get(src_img)
                if src_faces:
                    swap_targets.append({
                        "target_embedding": np.array(s["target_embedding"], dtype=np.float32),
                        "source_face": src_faces[0]
                    })

            return execute_swap(tmp_media, is_video, swap_targets, pipeline=pipeline)

        return {"error": f"Unknown action '{action}'"}

    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(tmp_media):
            os.remove(tmp_media)


runpod.serverless.start({"handler": handler})
