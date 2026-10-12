import os
import sys
import cv2
import uuid
import base64
import traceback
import subprocess
import requests
import numpy as np
import onnxruntime as ort
import insightface
import runpod

print("=== Starting Worker Initialization ===")
sys.stdout.flush()

try:
    try:
        ort.preload_dlls()
    except Exception as preload_err:
        print(f"preload_dlls warning: {preload_err}")

    available_providers = ort.get_available_providers()
    print(f"Available ONNX providers detected: {available_providers}")

    if 'CUDAExecutionProvider' not in available_providers:
        raise RuntimeError(
            f"FATAL: CUDAExecutionProvider was not found by ONNX Runtime! Detected: {available_providers}"
        )

    providers = ['CUDAExecutionProvider']

    face_app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root="/root/.insightface",
        allowed_modules=['detection', 'recognition'],
        providers=providers
    )
    face_app.prepare(ctx_id=0, det_thresh=0.35, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model(
        "/app/checkpoints/inswapper_128.onnx",
        download=False,
        providers=providers
    )

    occ_session = ort.InferenceSession(
        "/app/checkpoints/face_occluder.onnx",
        providers=providers
    )

    for name, m in face_app.models.items():
        active = m.session.get_providers()
        print(f"Model '{name}' active providers: {active}")
        if 'CUDAExecutionProvider' not in active:
            raise RuntimeError(f"FATAL: '{name}' is NOT running on GPU (providers: {active})")

    for label, sess in [("Inswapper", swapper.session), ("Occluder", occ_session)]:
        active = sess.get_providers()
        print(f"{label} active providers: {active}")
        if 'CUDAExecutionProvider' not in active:
            raise RuntimeError(f"FATAL: {label} is NOT running on GPU (providers: {active})")

    occ_input = occ_session.get_inputs()[0]
    occ_shape = occ_input.shape
    print(f"Occluder input: {occ_input.name} {occ_shape}")
    occ_nchw = (len(occ_shape) == 4 and occ_shape[1] == 3)
    try:
        occ_size = (int(occ_shape[3]), int(occ_shape[2])) if occ_nchw else (int(occ_shape[2]), int(occ_shape[1]))
    except Exception:
        occ_size = (256, 256)

    print("=== Worker Successfully Initialized: Models Residing in VRAM ===")
    sys.stdout.flush()

except Exception as init_err:
    print("FATAL INITIALIZATION ERROR:")
    traceback.print_exc()
    sys.stdout.flush()
    raise init_err


def occlusion_mask(crop_bgr):
    img = cv2.resize(crop_bgr, occ_size).astype(np.float32) / 255.0
    if occ_nchw:
        img = img.transpose(2, 0, 1)
    out = occ_session.run(None, {occ_input.name: img[None]})[0][0]
    m = np.clip(np.squeeze(out), 0, 1).astype(np.float32)
    m = cv2.resize(m, (crop_bgr.shape[1], crop_bgr.shape[0]))
    m = (np.clip(cv2.GaussianBlur(m, (0, 0), 5), 0.5, 1) - 0.5) * 2
    return m


def box_mask(size=128, blur=0.5):
    amount = int(size * 0.5 * blur)
    edge = max(amount // 2, 1)
    m = np.ones((size, size), np.float32)
    m[:edge, :] = 0
    m[-edge:, :] = 0
    m[:, :edge] = 0
    m[:, -edge:] = 0
    return cv2.GaussianBlur(m, (0, 0), max(amount * 0.25, 0.1))


def swap_face(frame, face, source_face):
    bgr_fake, M = swapper.get(frame, face, source_face, paste_back=False)
    bgr_fake = np.ascontiguousarray(bgr_fake)
    h, w = frame.shape[:2]
    crop = cv2.warpAffine(frame, M, (128, 128), borderValue=0)
    mask = np.minimum(box_mask(), occlusion_mask(crop))
    IM = cv2.invertAffineTransform(M)
    fake_w = cv2.warpAffine(bgr_fake, IM, (w, h), borderValue=0).astype(np.float32)
    mask_w = cv2.warpAffine(mask, IM, (w, h), borderValue=0)[:, :, None]
    out = frame.astype(np.float32) * (1 - mask_w) + fake_w * mask_w
    return np.clip(out, 0, 255).astype(np.uint8)


def resolve_input_to_file(data_or_url, out_path):
    if data_or_url.startswith("http://") or data_or_url.startswith("https://"):
        resp = requests.get(data_or_url, timeout=60)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
    else:
        if "," in data_or_url:
            data_or_url = data_or_url.split(",", 1)[1]
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(data_or_url))


def decode_image(data_or_url):
    temp = f"/tmp/img_{uuid.uuid4().hex[:6]}.jpg"
    try:
        resolve_input_to_file(data_or_url, temp)
        return cv2.imread(temp)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


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
        centroid = np.mean([detections[idx]['embedding'] for idx in c_indices], axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        _, buf = cv2.imencode(".jpg", detections[best_idx]['crop'])
        unique_characters.append({
            "character_id": int(l),
            "thumbnail_base64": base64.b64encode(buf).decode("utf-8"),
            "embedding": centroid.tolist()
        })
    return unique_characters


def collect_faces(frame, detections, min_size=0):
    for f in face_app.get(frame):
        box = f.bbox.astype(int)
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
        if (x2 - x1) > min_size and (y2 - y1) > min_size:
            detections.append({
                'crop': frame[y1:y2, x1:x2],
                'embedding': f.normed_embedding,
                'score': (x2 - x1) * (y2 - y1)
            })


def detect_unique_people(media_path, is_animation=True):
    detections = []
    if not is_animation:
        collect_faces(cv2.imread(media_path), detections)
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
                collect_faces(frame, detections, min_size=40)
            frame_idx += 1
        cap.release()

    if not detections:
        return []
    return cluster_embeddings_numpy(detections)


def apply_swaps(frame, swap_targets):
    for face in face_app.get(frame):
        for target in swap_targets:
            sim = float(np.dot(face.normed_embedding, target["target_embedding"]))
            if sim >= 0.40:
                frame = swap_face(frame, face, target["source_face"])
                break
    return frame


def execute_swap(media_path, media_type, swap_targets):
    if media_type == "image":
        frame = apply_swaps(cv2.imread(media_path), swap_targets)
        _, out_buf = cv2.imencode(".jpg", frame)
        return {"status": "success", "media_type": "image", "base64": base64.b64encode(out_buf).decode("utf-8")}

    temp_id = uuid.uuid4().hex[:8]
    tmp_swapped_mp4 = f"/tmp/swap_{temp_id}.mp4"
    tmp_final = f"/tmp/final_{temp_id}.{ 'gif' if media_type == 'gif' else 'mp4' }"

    cap = cv2.VideoCapture(media_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 60:
        fps = 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(tmp_swapped_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(apply_swaps(frame, swap_targets))

    cap.release()
    writer.release()

    if media_type == "gif":
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_swapped_mp4,
            "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5",
            "-loop", "0", tmp_final
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_swapped_mp4, "-i", media_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0?", "-shortest", tmp_final
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    with open(tmp_final, "rb") as f:
        media_b64 = base64.b64encode(f.read()).decode("utf-8")

    for fpath in [tmp_swapped_mp4, tmp_final]:
        if os.path.exists(fpath):
            os.remove(fpath)

    return {"status": "success", "media_type": media_type, "base64": media_b64}


def handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action", "swap")
    media_data = job_input.get("media")

    if not media_data:
        return {"error": "Missing 'media' payload."}

    if job_input.get("is_gif") or (isinstance(media_data, str) and ".gif" in media_data.lower()[:100]):
        media_type, ext = "gif", "gif"
    elif job_input.get("is_video", True):
        media_type, ext = "video", "mp4"
    else:
        media_type, ext = "image", "jpg"

    tmp_media = f"/tmp/input_{uuid.uuid4().hex[:8]}.{ext}"

    try:
        resolve_input_to_file(media_data, tmp_media)

        if action == "detect":
            characters = detect_unique_people(tmp_media, is_animation=(media_type in ["video", "gif"]))
            return {"status": "success", "characters": characters}

        elif action == "swap":
            swap_targets = []
            for s in job_input.get("swaps", []):
                src_faces = face_app.get(decode_image(s["new_face"]))
                if src_faces:
                    swap_targets.append({
                        "target_embedding": np.array(s["target_embedding"], dtype=np.float32),
                        "source_face": src_faces[0]
                    })
            return execute_swap(tmp_media, media_type, swap_targets)

        return {"error": f"Unknown action '{action}'"}

    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(tmp_media):
            os.remove(tmp_media)


runpod.serverless.start({"handler": handler})
