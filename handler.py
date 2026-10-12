import os
import cv2
import base64
import tempfile
import numpy as np
import torch
import runpod
import insightface
from insightface.app import FaceAnalysis

print("[Worker] Initializing Models & Providers...")

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# 1. Initialize Face Detection & Alignment
face_app = FaceAnalysis(name="buffalo_l", providers=providers)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Load Inswapper Model
SWAPPER_PATH = os.environ.get("SWAPPER_PATH", "/models/inswapper_128.onnx")
if not os.path.exists(SWAPPER_PATH):
    SWAPPER_PATH = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")

swapper = insightface.model_zoo.get_model(SWAPPER_PATH, download=False, providers=providers)

# 3. Optional Enhancers
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
    print(f"[Worker] GFPGAN skipped: {e}")

print("[Worker] Initialized and Ready.")


def strip_prefix(b64_str: str) -> str:
    return b64_str.split(",", 1)[1] if "," in b64_str else b64_str


def decode_bytes(b64_str: str) -> bytes:
    return base64.b64decode(strip_prefix(b64_str))


def decode_img(b64_str: str) -> np.ndarray:
    data = decode_bytes(b64_str)
    np_arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def encode_img_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return base64.b64encode(buf).decode("utf-8")


def extract_first_frame(media_bytes: bytes) -> np.ndarray:
    """Extracts first valid frame from image or video bytes."""
    # Try as static image first
    np_arr = np.frombuffer(media_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is not None and img.size > 0:
        return img

    # Try as video/gif container
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(media_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    ok, frame = cap.read()
    cap.release()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return frame if (ok and frame is not None and frame.size > 0) else None


def detect_faces(img_bgr: np.ndarray):
    """Detects all faces in frame and returns clean crops for the UI."""
    faces = face_app.get(img_bgr)
    results = []
    h, w = img_bgr.shape[:2]

    for idx, face in enumerate(faces):
        bbox = [int(v) for v in face.bbox.tolist()]
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w, bbox[2]), min(h, bbox[3])

        crop = img_bgr[y1:y2, x1:x2]
        crop_b64 = encode_img_b64(crop) if crop.size > 0 else ""

        results.append({
            "index": idx,
            "bbox": bbox,
            "det_score": float(face.det_score) if hasattr(face, "det_score") else 1.0,
            "crop_base64": f"data:image/jpeg;base64,{crop_b64}"
        })
    return faces, results


def apply_enhancer(img_bgr: np.ndarray, pipeline: str) -> np.ndarray:
    if pipeline == "gfpgan" and gfpgan_enhancer is not None:
        try:
            _, _, res = gfpgan_enhancer.enhance(img_bgr, paste_back=True)
            return res
        except Exception:
            return img_bgr
    return img_bgr


def swap_frame_multi(frame: np.ndarray, target_faces_first_frame, mappings, pipeline: str):
    """Swaps assigned faces on a single frame."""
    current_faces = face_app.get(frame)
    if not current_faces:
        return frame

    out = frame.copy()
    for m in mappings:
        t_idx = m["target_index"]
        src_face = m["source_face"]

        # Track face by proximity / index
        if t_idx < len(current_faces):
            out = swapper.get(out, current_faces[t_idx], src_face, paste_back=True)

    if pipeline in ["gfpgan", "codeformer"]:
        out = apply_enhancer(out, pipeline)
    return out


def handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action", "swap")

    if "target_image" not in job_input:
        return {"error": "Missing 'target_image'"}

    raw_bytes = decode_bytes(job_input["target_image"])
    is_video = job_input.get("is_video", False)
    pipeline = job_input.get("pipeline", "fast").lower()

    # -------------------------------------------------------------
    # ACTION 1: DETECT FACES (Used for Image, GIF, and Video previews)
    # -------------------------------------------------------------
    if action == "detect":
        first_frame = extract_first_frame(raw_bytes)
        if first_frame is None:
            return {"error": "Could not read media frames for detection."}

        _, face_data = detect_faces(first_frame)
        return {
            "status": "success",
            "faces_detected": len(face_data),
            "faces": face_data
        }

    # -------------------------------------------------------------
    # ACTION 2: SWAP FACES (With multi-mapping & apply-all support)
    # -------------------------------------------------------------
    elif action == "swap":
        # Resolve all source faces from mappings
        faces_map = job_input.get("faces_map", [])
        if not faces_map and "source_image" in job_input:
            faces_map = [{"target_index": 0, "source_image": job_input["source_image"]}]

        if not faces_map:
            return {"error": "No source image or faces_map provided."}

        prepared_mappings = []
        for m in faces_map:
            s_img = decode_img(m["source_image"])
            if s_img is not None:
                s_faces = face_app.get(s_img)
                if s_faces:
                    prepared_mappings.append({
                        "target_index": int(m.get("target_index", 0)),
                        "source_face": s_faces[0]
                    })

        if not prepared_mappings:
            return {"error": "No faces could be identified in any provided source images."}

        # --- A. STATIC IMAGE SWAP ---
        if not is_video:
            target_img = decode_img(job_input["target_image"])
            if target_img is None:
                return {"error": "Failed to decode static target image."}

            output_img = swap_frame_multi(target_img, None, prepared_mappings, pipeline)
            return {
                "status": "success",
                "image": f"data:image/jpeg;base64,{encode_img_b64(output_img)}"
            }

        # --- B. FULL VIDEO / GIF FRAME-BY-FRAME SWAP ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as in_tmp:
            in_tmp.write(raw_bytes)
            in_video_path = in_tmp.name

        out_video_path = in_video_path.replace(".mp4", "_out.mp4")

        cap = cv2.VideoCapture(in_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            swapped_frame = swap_frame_multi(frame, None, prepared_mappings, pipeline)
            writer.write(swapped_frame)

        cap.release()
        writer.release()

        with open(out_video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Cleanup temp files
        for p in [in_video_path, out_video_path]:
            if os.path.exists(p):
                os.remove(p)

        return {
            "status": "success",
            "video": f"data:video/mp4;base64,{video_b64}"
        }

    return {"error": f"Unknown action '{action}'"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
