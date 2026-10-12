import os
import io
import cv2
import base64
import tempfile
import urllib.request
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import runpod

# Absolute model paths inside container
APP_DIR = os.path.dirname(os.path.abspath(__file__))
INSWAPPER_PATH = os.environ.get("INSWAPPER_PATH", "/app/inswapper_128.onnx")

# Providers strictly configured for CUDA
PROVIDERS = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 14 * 1024 * 1024 * 1024, # 14GB allocation
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True
    }),
    'CPUExecutionProvider'
]

print("[Worker] Initializing InsightFace on GPU...")
app = FaceAnalysis(name='buffalo_l', providers=PROVIDERS)
app.prepare(ctx_id=0, det_size=(640, 640))

if not os.path.exists(INSWAPPER_PATH):
    # fallback check in local folder
    if os.path.exists("inswapper_128.onnx"):
        INSWAPPER_PATH = "inswapper_128.onnx"

print(f"[Worker] Loading Inswapper from: {INSWAPPER_PATH}")
swapper = insightface.model_zoo.get_model(INSWAPPER_PATH, providers=PROVIDERS)
print("[Worker] Ready for GPU inference.")


def decode_b64(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    data = base64.b64decode(b64_str)
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def encode_b64(img, ext=".jpg"):
    _, buf = cv2.imencode(ext, img)
    return base64.b64encode(buf).decode("utf-8")

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action")

    # 1. ACTION: DETECT (Detects faces and returns embeddings + thumbnails)
    if action == "detect":
        media_b64 = job_input.get("media")
        img = decode_b64(media_b64)
        faces = app.get(img)

        characters = []
        h, w, _ = img.shape
        for f in faces:
            box = f.bbox.astype(int)
            x1, y1 = max(0, box[0]), max(0, box[1])
            x2, y2 = min(w, box[2]), min(h, box[3])
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                characters.append({
                    "embedding": f.embedding.tolist(),
                    "thumbnail_base64": encode_b64(crop)
                })
        return {"characters": characters}

    # 2. ACTION: SWAP (Multi-face embedding match & swap)
    elif action == "swap":
        media_b64 = job_input.get("media")
        swaps = job_input.get("swaps", [])
        
        # Pre-process source faces
        active_swaps = []
        for s in swaps:
            src_img = decode_b64(s["new_face"])
            src_faces = app.get(src_img)
            if src_faces:
                # pick largest face
                largest = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                active_swaps.append({
                    "target_emb": s["target_embedding"],
                    "source_face": largest
                })

        img = decode_b64(media_b64)
        target_faces = app.get(img)

        # Match each detected target face against assignments
        for t_face in target_faces:
            for s in active_swaps:
                if cosine_sim(t_face.embedding, s["target_emb"]) >= 0.50:
                    img = swapper.get(img, t_face, s["source_face"], paste_back=True)
                    break

        return {
            "media_type": "image",
            "base64": encode_b64(img)
        }

    return {"error": f"Invalid action: {action}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
