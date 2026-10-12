import os
import cv2
import base64
import tempfile
import urllib.request
import numpy as np
import runpod
import insightface
from insightface.app import FaceAnalysis

# Initialize InsightFace model zoo with available ONNX providers
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Path to inswapper model inside container
MODEL_PATH = os.environ.get('INSWAPPER_PATH', '/root/.insightface/models/inswapper_128.onnx')
swapper = None
if os.path.exists(MODEL_PATH):
    swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


def decode_media_to_frame(media_input):
    """
    Decodes an image URL, base64 string, or video file into a valid BGR numpy frame.
    Prevents NoneType passing to face_app.get().
    """
    if media_input is None:
        return None

    # Handle direct numpy array
    if isinstance(media_input, np.ndarray):
        return media_input

    # Handle Web URL
    if isinstance(media_input, str) and (media_input.startswith("http://") or media_input.startswith("https://")):
        req = urllib.request.Request(media_input, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
    elif isinstance(media_input, str):
        # Strip data URL prefixes (e.g., data:image/png;base64, or data:video/mp4;base64,)
        if "," in media_input:
            media_input = media_input.split(",", 1)[1]
        try:
            data = base64.b64decode(media_input)
        except Exception:
            return None
    else:
        return None

    # Attempt 1: Standard image decode (JPEG, PNG, WebP)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None and getattr(img, 'shape', None) is not None:
        return img

    # Attempt 2: Video file decode (Extract frame 0 for face detection/scan)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        success, frame = cap.read()
        cap.release()

        if success and frame is not None and getattr(frame, 'shape', None) is not None:
            return frame
    except Exception as e:
        print(f"Video extraction error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return None


def detect_faces(img_bgr):
    """Safely extracts bounding boxes and landmarks without NoneType crashes."""
    if img_bgr is None or not hasattr(img_bgr, 'shape'):
        raise ValueError("Invalid target image: Received None or corrupted media buffer.")

    faces = face_app.get(img_bgr)
    face_data = []
    for idx, face in enumerate(faces):
        bbox = [int(v) for v in face.bbox.tolist()]
        kps = [[int(pt[0]), int(pt[1])] for pt in face.kps.tolist()]
        face_data.append({
            "index": idx,
            "bbox": bbox,
            "det_score": float(face.det_score) if hasattr(face, 'det_score') else 1.0,
            "kps": kps
        })
    return faces, face_data


def encode_img_to_b64(img_bgr):
    """Encodes BGR numpy image to base64 JPEG string."""
    ret, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ret:
        raise RuntimeError("Failed to encode output image")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')


def handler(job):
    """RunPod Serverless Job Handler."""
    job_input = job.get("input", {})
    action = job_input.get("action", "swap")  # 'detect' or 'swap'

    # Retrieve source & target data
    target_raw = job_input.get("target_image") or job_input.get("target_video") or job_input.get("target_media")
    source_raw = job_input.get("source_image") or job_input.get("source_face")

    target_img = decode_media_to_frame(target_raw)
    if target_img is None:
        return {"error": "Could not decode target_image as an image or extract a video frame."}

    # Action: Scan / Detect Faces only
    if action == "detect":
        _, face_data = detect_faces(target_img)
        return {
            "status": "success",
            "faces_detected": len(face_data),
            "faces": face_data,
            "width": int(target_img.shape[1]),
            "height": int(target_img.shape[0])
        }

    # Action: Swap Faces
    if not source_raw:
        return {"error": "Missing source_image in payload."}

    source_img = decode_media_to_frame(source_raw)
    if source_img is None:
        return {"error": "Could not decode source_image as an image."}

    source_faces = face_app.get(source_img)
    if not source_faces:
        return {"error": "No face detected in source_image."}
    source_face = source_faces[0]

    target_faces, _ = detect_faces(target_img)
    if not target_faces:
        return {"error": "No face detected in target media."}

    if swapper is None:
        return {"error": "Inswapper model is not initialized or not found on worker."}

    # Execute Face Swap
    target_face_index = job_input.get("target_face_index", 0)
    res_img = target_img.copy()

    if target_face_index == -1 or job_input.get("swap_all", False):
        for tf in target_faces:
            res_img = swapper.get(res_img, tf, source_face, paste_back=True)
    else:
        idx = min(target_face_index, len(target_faces) - 1)
        res_img = swapper.get(res_img, target_faces[idx], source_face, paste_back=True)

    result_b64 = encode_img_to_b64(res_img)
    return {
        "status": "success",
        "result_image": result_b64
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
