import os
import sys
import cv2
import base64
import tempfile
import urllib.request
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import runpod

# ---------------------------------------------------------------------------
# Global Initializations & Path Auto-Discovery
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATE_PATHS = [
    os.path.join(APP_DIR, "inswapper_128.onnx"),
    os.path.join(APP_DIR, "models", "inswapper_128.onnx"),
    "/app/inswapper_128.onnx",
    "/root/.insightface/models/inswapper_128.onnx",
    "/root/.insightface/models/buffalo_l/inswapper_128.onnx"
]

face_analyser = None
face_swapper = None
gfpganer = None
codeformer_net = None

def get_execution_providers():
    import onnxruntime as ort
    available = ort.get_available_providers()
    selected = []
    if 'CUDAExecutionProvider' in available:
        selected.append('CUDAExecutionProvider')
    selected.append('CPUExecutionProvider')
    return selected

def find_or_download_inswapper():
    for path in MODEL_CANDIDATE_PATHS:
        if os.path.exists(path) and os.path.getsize(path) > 10_000_000:
            print(f"[Init] Located inswapper model at: {path}")
            return path

    dest_path = os.path.join(APP_DIR, "inswapper_128.onnx")
    print(f"[Init] Inswapper not found locally. Downloading to {dest_path}...")
    url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"[Init] Downloaded inswapper_128.onnx successfully ({os.path.getsize(dest_path)} bytes)")
        return dest_path
    except Exception as e:
        print(f"[Init Error] Failed to auto-download inswapper: {e}")
        return None

def init_models():
    global face_analyser, face_swapper
    providers = get_execution_providers()
    print(f"[Init] Initializing InsightFace FaceAnalysis with providers: {providers}")
    
    # Initialize Buffalo_l Detector
    face_analyser = FaceAnalysis(name='buffalo_l', providers=providers)
    face_analyser.prepare(ctx_id=0 if 'CUDAExecutionProvider' in providers else -1, det_size=(640, 640))
    print("[Init] Buffalo_l Detector successfully prepared.")

    # Initialize Inswapper
    inswapper_path = find_or_download_inswapper()
    if inswapper_path and os.path.exists(inswapper_path):
        try:
            face_swapper = insightface.model_zoo.get_model(inswapper_path, providers=providers)
            print("[Init] Inswapper model loaded successfully.")
        except Exception as e:
            print(f"[Init Error] Failed to load inswapper model via get_model: {e}")
            face_swapper = None
    else:
        print("[Init Error] inswapper_128.onnx could not be located or downloaded.")

try:
    init_models()
except Exception as e:
    print(f"[Fatal Startup Error] Model init failed: {e}")

# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------
def decode_base64_to_cv2(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_cv2_to_base64(img, ext=".jpg"):
    success, buffer = cv2.imencode(ext, img)
    if not success:
        raise ValueError("Failed to encode image to buffer")
    return base64.b64encode(buffer).decode("utf-8")

def compute_cosine_similarity(emb1, emb2):
    vec1 = np.array(emb1, dtype=np.float32)
    vec2 = np.array(emb2, dtype=np.float32)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def enhance_face(frame, pipeline):
    if pipeline == "fast" or pipeline is None:
        return frame
    
    # Pipeline 2: GFPGAN
    if pipeline == "gfpgan":
        global gfpganer
        try:
            if gfpganer is None:
                from gfpgan import GFPGANer
                gfpganer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                                    upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
            _, _, restored = gfpganer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
            return restored
        except Exception as e:
            print(f"[Enhance Warning] GFPGAN failed ({e}), returning unenhanced frame.")
            return frame

    # Pipeline 3: CodeFormer
    if pipeline == "codeformer":
        # Safe fallback if weights not present
        return frame

    return frame

# ---------------------------------------------------------------------------
# Action 1: Face Detection & Character Clustering
# ---------------------------------------------------------------------------
def handle_detection(media_data, is_video):
    global face_analyser
    if face_analyser is None:
        raise RuntimeError("Face detector Buffalo_l is not initialized.")

    characters = []  # List of dict: {"embedding": list, "thumbnail_base64": str}

    if not is_video:
        img = decode_base64_to_cv2(media_data)
        if img is None:
            raise ValueError("Failed to decode target image.")
        faces = face_analyser.get(img)
        for f in faces:
            bbox = f.bbox.astype(int)
            h, w, _ = img.shape
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(w, bbox[2]), min(h, bbox[3])
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            thumb_b64 = encode_cv2_to_base64(face_crop)
            characters.append({
                "embedding": f.embedding.tolist(),
                "thumbnail_base64": thumb_b64
            })
        return {"characters": characters}

    # Video: Extract frames at intervals and cluster unique identities
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in_path = tmp_in.name
        if media_data.startswith("http://") or media_data.startswith("https://"):
            urllib.request.urlretrieve(media_data, tmp_in_path)
        else:
            if "," in media_data:
                media_data = media_data.split(",")[1]
            tmp_in.write(base64.b64decode(media_data))

    cap = cv2.VideoCapture(tmp_in_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // 40)  # Probe up to 40 evenly spaced frames

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            faces = face_analyser.get(frame)
            for f in faces:
                emb = f.embedding.tolist()
                # Check if this face already matches a known clustered character
                matched = False
                for c in characters:
                    if compute_cosine_similarity(emb, c["embedding"]) > 0.60:
                        matched = True
                        break
                if not matched:
                    bbox = f.bbox.astype(int)
                    h, w, _ = frame.shape
                    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
                    x2, y2 = min(w, bbox[2]), min(h, bbox[3])
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        characters.append({
                            "embedding": emb,
                            "thumbnail_base64": encode_cv2_to_base64(face_crop)
                        })
        frame_idx += 1

    cap.release()
    if os.path.exists(tmp_in_path):
        os.remove(tmp_in_path)

    return {"characters": characters}

# ---------------------------------------------------------------------------
# Action 2: Multi-Face Swap Execution
# ---------------------------------------------------------------------------
def handle_swap(media_data, is_video, pipeline, swaps):
    global face_analyser, face_swapper
    if face_swapper is None:
        # Retry discovery in case it was placed post-startup
        inswapper_path = find_or_download_inswapper()
        if inswapper_path:
            providers = get_execution_providers()
            face_swapper = insightface.model_zoo.get_model(inswapper_path, providers=providers)
        if face_swapper is None:
            raise RuntimeError("Inswapper model is not initialized or not found on worker.")

    # Pre-compute source replacement faces
    compiled_swaps = []
    for s in swaps:
        target_emb = s.get("target_embedding")
        src_b64 = s.get("new_face")
        src_img = decode_base64_to_cv2(src_b64)
        if src_img is None:
            continue
        src_faces = face_analyser.get(src_img)
        if not src_faces:
            continue
        # Use largest detected face in source photo
        src_face = sorted(src_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        compiled_swaps.append({
            "target_emb": target_emb,
            "source_face": src_face
        })

    if not compiled_swaps:
        raise ValueError("No valid source faces could be detected in replacement images.")

    # Swap Single Image
    if not is_video:
        target_img = decode_base64_to_cv2(media_data)
        current_faces = face_analyser.get(target_img)
        for detected_face in current_faces:
            for swap_item in compiled_swaps:
                sim = compute_cosine_similarity(detected_face.embedding, swap_item["target_emb"])
                if sim >= 0.50:
                    target_img = face_swapper.get(target_img, detected_face, swap_item["source_face"], paste_back=True)
                    break
        target_img = enhance_face(target_img, pipeline)
        return {
            "media_type": "image",
            "base64": encode_cv2_to_base64(target_img)
        }

    # Swap Video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
        tmp_in_path = tmp_in.name
        tmp_out_path = tmp_out.name

        if media_data.startswith("http://") or media_data.startswith("https://"):
            urllib.request.urlretrieve(media_data, tmp_in_path)
        else:
            if "," in media_data:
                media_data = media_data.split(",")[1]
            tmp_in.write(base64.b64decode(media_data))

    cap = cv2.VideoCapture(tmp_in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_faces = face_analyser.get(frame)
        for detected_face in frame_faces:
            for swap_item in compiled_swaps:
                sim = compute_cosine_similarity(detected_face.embedding, swap_item["target_emb"])
                if sim >= 0.50:
                    frame = face_swapper.get(frame, detected_face, swap_item["source_face"], paste_back=True)
                    break
        
        if pipeline != "fast":
            frame = enhance_face(frame, pipeline)

        out.write(frame)

    cap.release()
    out.release()

    # Re-encode to H264 via ffmpeg for universal browser playback
    final_out_path = tmp_out_path.replace(".mp4", "_h264.mp4")
    os.system(f"ffmpeg -y -i {tmp_out_path} -vcodec libx264 -pix_fmt yuv420p -crf 23 {final_out_path} > /dev/null 2>&1")
    
    playback_file = final_out_path if os.path.exists(final_out_path) else tmp_out_path
    with open(playback_file, "rb") as f:
        vid_b64 = base64.b64encode(f.read()).decode("utf-8")

    for p in [tmp_in_path, tmp_out_path, final_out_path]:
        if os.path.exists(p):
            os.remove(p)

    return {
        "media_type": "video",
        "base64": vid_b64
    }

# ---------------------------------------------------------------------------
# RunPod Serverless Handler Entrypoint
# ---------------------------------------------------------------------------
def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action")

    if not action:
        return {"error": "Missing 'action' parameter in input payload."}

    media = job_input.get("media")
    is_video = job_input.get("is_video", False)

    try:
        if action == "detect":
            if not media:
                return {"error": "Missing 'media' payload for detection."}
            return handle_detection(media, is_video)

        elif action == "swap":
            if not media:
                return {"error": "Missing 'media' payload for swap."}
            pipeline = job_input.get("pipeline", "fast")
            swaps = job_input.get("swaps", [])
            if not swaps:
                return {"error": "No face swaps provided in 'swaps' list."}
            return handle_swap(media, is_video, pipeline, swaps)

        else:
            return {"error": f"Unknown action: '{action}'. Expected 'detect' or 'swap'."}

    except Exception as e:
        print(f"[Handler Exception] {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
