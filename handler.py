import os
import cv2
import base64
import tempfile
import numpy as np
import runpod
import insightface
from insightface.app import FaceAnalysis

print("[Worker] Starting Puthiya Mugam Handler...")

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# 1. Models
face_app = FaceAnalysis(name="buffalo_l", providers=providers)
face_app.prepare(ctx_id=0, det_size=(640, 640))

SWAPPER_PATH = os.environ.get("SWAPPER_PATH", "/models/inswapper_128.onnx")
if not os.path.exists(SWAPPER_PATH):
    SWAPPER_PATH = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")

swapper = insightface.model_zoo.get_model(SWAPPER_PATH, download=False, providers=providers)
print("[Worker] Ready for Jobs.")


def decode_media_to_image(b64_str: str) -> np.ndarray:
    """Safely decodes any Base64 data (image, video, GIF) into a valid BGR NumPy array."""
    if not b64_str:
        return None

    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    raw_bytes = base64.b64decode(b64_str.strip())

    # 1. Try decoding directly as standard image (JPG, PNG, WebP)
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is not None and getattr(img, "size", 0) > 0:
        return img

    # 2. If it's a video or GIF, write to temporary file and grab frame 0
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(raw_bytes)
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return frame
    except Exception as e:
        print(f"[Worker] Media fallback decode error: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return None


def encode_img_base64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return base64.b64encode(buf).decode("utf-8")


def detect_faces(img_bgr: np.ndarray):
    faces = face_app.get(img_bgr)
    results = []
    h, w = img_bgr.shape[:2]

    for idx, face in enumerate(faces):
        bbox = [int(v) for v in face.bbox.tolist()]
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w, bbox[2]), min(h, bbox[3])

        crop = img_bgr[y1:y2, x1:x2]
        crop_b64 = encode_img_base64(crop) if crop is not None and crop.size > 0 else ""

        results.append({
            "index": idx,
            "bbox": bbox,
            "det_score": float(face.det_score) if hasattr(face, "det_score") else 1.0,
            "crop_base64": f"data:image/jpeg;base64,{crop_b64}"
        })
    return faces, results


def handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action", "swap")

    if "target_image" not in job_input:
        return {"error": "Missing 'target_image' in request payload."}

    # Safe decode: guarantees a non-None valid BGR array or returns an explicit error
    target_img = decode_media_to_image(job_input["target_image"])
    if target_img is None:
        return {"error": "Could not decode target_image into a valid image or video frame."}

    # --- STEP 1: DETECT ---
    if action == "detect":
        try:
            _, face_data = detect_faces(target_img)
            return {
                "status": "success",
                "faces_detected": len(face_data),
                "faces": face_data
            }
        except Exception as e:
            return {"error": f"Face detection failed: {str(e)}"}

    # --- STEP 2: SWAP (Supports individual mapping & Apply-to-All) ---
    elif action == "swap":
        target_faces = face_app.get(target_img)
        if not target_faces:
            return {"error": "No face found in target media."}

        output_img = target_img.copy()

        # Check if UI sent multi-face mapping (from PuthiyaMugam-1.html)
        faces_map = job_input.get("faces_map")

        if faces_map and isinstance(faces_map, list):
            for mapping in faces_map:
                src_b64 = mapping.get("source_image")
                if not src_b64:
                    continue

                src_img = decode_media_to_image(src_b64)
                if src_img is None:
                    continue

                src_faces = face_app.get(src_img)
                if not src_faces:
                    continue

                t_idx = int(mapping.get("target_index", 0))
                if t_idx < len(target_faces):
                    target_face = target_faces[t_idx]
                    source_face = src_faces[0]
                    output_img = swapper.get(output_img, target_face, source_face, paste_back=True)
        else:
            # Fallback to single swap
            if "source_image" not in job_input:
                return {"error": "Missing 'source_image' for swap."}

            src_img = decode_media_to_image(job_input["source_image"])
            if src_img is None:
                return {"error": "Failed to decode source_image."}

            src_faces = face_app.get(src_img)
            if not src_faces:
                return {"error": "No face found in source image."}

            s_idx = min(int(job_input.get("source_face_index", 0)), len(src_faces) - 1)
            t_idx = min(int(job_input.get("target_face_index", 0)), len(target_faces) - 1)

            output_img = swapper.get(output_img, target_faces[t_idx], src_faces[s_idx], paste_back=True)

        return {
            "status": "success",
            "image": f"data:image/jpeg;base64,{encode_img_base64(output_img)}"
        }

    return {"error": f"Unknown action '{action}'"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
