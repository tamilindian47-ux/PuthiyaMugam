import os
import io
import cv2
import base64
import numpy as np
import torch
import runpod
import insightface
from insightface.app import FaceAnalysis

print("[Worker] Initializing Models & Providers...")

# Force CUDA Execution Providers
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# 1. Initialize Face Detection & Alignment (buffalo_l)
face_app = FaceAnalysis(name="buffalo_l", providers=providers)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Load Inswapper Model
SWAPPER_PATH = os.environ.get("SWAPPER_PATH", "/models/inswapper_128.onnx")
if not os.path.exists(SWAPPER_PATH):
    SWAPPER_PATH = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")

swapper = insightface.model_zoo.get_model(SWAPPER_PATH, download=False, providers=providers)

# 3. Optional Enhancers (GFPGAN / CodeFormer)
gfpgan_enhancer = None
try:
    from gfpgan import GFPGANer
    GFPGAN_PATH = "/models/GFPGANv1.4.pth"
    if os.path.exists(GFPGAN_PATH):
        gfpgan_enhancer = GFPGANer(
            model_path=GFPGAN_PATH,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("[Worker] GFPGAN successfully loaded.")
except Exception as e:
    print(f"[Worker] GFPGAN init skipped/deferred: {e}")

codeformer_net = None
try:
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    from torchvision.transforms.functional import normalize
    # CodeFormer can be plugged here if installed
except Exception:
    pass

print("[Worker] Successfully Initialized and Ready for Jobs.")


def decode_base64_img(base64_str: str) -> np.ndarray:
    """Decodes a data URI or raw base64 string to a BGR NumPy array."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def encode_img_base64(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    """Encodes a BGR image to base64 JPEG format."""
    _, buffer = cv2.imencode(ext, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return base64.b64encode(buffer).decode("utf-8")


def detect_faces(img_bgr: np.ndarray):
    """Detects faces and returns bounding boxes, landmark previews, and crops."""
    faces = face_app.get(img_bgr)
    results = []
    for idx, face in enumerate(faces):
        bbox = [int(v) for v in face.bbox.tolist()]
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(img_bgr.shape[1], bbox[2]), min(img_bgr.shape[2] if len(img_bgr.shape) > 2 else img_bgr.shape[0], bbox[3])
        crop = img_bgr[y1:y2, x1:x2]
        crop_b64 = encode_img_base64(crop) if crop.size > 0 else ""

        results.append({
            "index": idx,
            "bbox": bbox,
            "det_score": float(face.det_score) if hasattr(face, "det_score") else 1.0,
            "crop_base64": f"data:image/jpeg;base64,{crop_b64}"
        })
    return faces, results


def apply_enhancer(img_bgr: np.ndarray, pipeline: str) -> np.ndarray:
    """Enhances skin and facial fidelity based on pipeline selection."""
    if pipeline == "gfpgan" and gfpgan_enhancer is not None:
        try:
            _, _, restored_img = gfpgan_enhancer.enhance(
                img_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            return restored_img
        except Exception as e:
            print(f"[Warning] GFPGAN enhancement failed: {e}")
            return img_bgr
    return img_bgr


def handler(event):
    """
    RunPod Serverless Request Handler
    Expected event payload:
    {
        "input": {
            "action": "detect" | "swap",
            "target_image": "<base64_string>",
            "source_image": "<base64_string>",      # required if action == 'swap'
            "target_face_index": 0,                 # optional index
            "source_face_index": 0,                 # optional index
            "pipeline": "fast" | "gfpgan" | "codeformer"
        }
    }
    """
    job_input = event.get("input", {})
    action = job_input.get("action", "swap")

    if "target_image" not in job_input:
        return {"error": "Missing 'target_image' in request payload."}

    try:
        target_img = decode_base64_img(job_input["target_image"])
    except Exception as e:
        return {"error": f"Failed decoding target_image: {str(e)}"}

    # --- Mode 1: Face Detection / Scanning Step ---
    if action == "detect":
        _, face_data = detect_faces(target_img)
        return {
            "status": "success",
            "faces_detected": len(face_data),
            "faces": face_data
        }

    # --- Mode 2: Face Swapping Step ---
    elif action == "swap":
        if "source_image" not in job_input:
            return {"error": "Missing 'source_image' for swap operation."}

        try:
            source_img = decode_base64_img(job_input["source_image"])
        except Exception as e:
            return {"error": f"Failed decoding source_image: {str(e)}"}

        source_faces = face_app.get(source_img)
        if not source_faces:
            return {"error": "No face found in source image."}

        target_faces = face_app.get(target_img)
        if not target_faces:
            return {"error": "No face found in target image."}

        # Select face indices
        s_idx = min(job_input.get("source_face_index", 0), len(source_faces) - 1)
        t_idx = min(job_input.get("target_face_index", 0), len(target_faces) - 1)

        source_face = source_faces[s_idx]
        target_face = target_faces[t_idx]

        # Execute Inswapper
        swapped_img = swapper.get(target_img, target_face, source_face, paste_back=True)

        # Apply selected quality enhancement pipeline
        pipeline = job_input.get("pipeline", "fast").lower()
        if pipeline in ["gfpgan", "codeformer"]:
            swapped_img = apply_enhancer(swapped_img, pipeline)

        result_b64 = encode_img_base64(swapped_img)
        return {
            "status": "success",
            "pipeline_used": pipeline,
            "image": f"data:image/jpeg;base64,{result_b64}"
        }

    return {"error": f"Unknown action '{action}'"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
