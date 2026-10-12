import os
import io
import cv2
import base64
import tempfile
import subprocess
import urllib.request
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis
from PIL import Image, ImageSequence
import runpod

INSWAPPER_PATH = os.environ.get("INSWAPPER_PATH", "/models/inswapper_128.onnx")
GFPGAN_PATH = "/models/GFPGANv1.4.pth"

CLUSTER_THRESH = 0.45
MATCH_THRESH = 0.40
DETECT_SAMPLES = 30

PROVIDERS = [
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": 6 * 1024 * 1024 * 1024,
        "cudnn_conv_algo_search": "HEURISTIC",
    }),
    "CPUExecutionProvider",
]

print("[Worker] ORT providers:", ort.get_available_providers(), flush=True)
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA provider missing"

app = FaceAnalysis(
    name="buffalo_l",
    providers=PROVIDERS,
    allowed_modules=["detection", "recognition"],
)
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)

swapper = insightface.model_zoo.get_model(INSWAPPER_PATH, providers=PROVIDERS)
assert swapper.session.get_providers()[0] == "CUDAExecutionProvider", "Swapper fell back to CPU"
print("[Worker] Ready on GPU.", flush=True)

_gfpgan = None


def get_gfpgan():
    global _gfpgan
    if _gfpgan is None:
        try:
            from gfpgan import GFPGANer
            _gfpgan = GFPGANer(model_path=GFPGAN_PATH, upscale=1, arch="clean",
                               channel_multiplier=2, bg_upsampler=None)
        except Exception as e:
            print("[Worker] GFPGAN unavailable:", e, flush=True)
            _gfpgan = False
    return _gfpgan


def b64_to_bytes(s):
    s = (s or "").strip()
    if s.startswith("http://") or s.startswith("https://"):
        req = urllib.request.Request(s, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    if "," in s[:100]:
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def bytes_to_img(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def img_to_b64(img, ext=".jpg"):
    ok, buf = cv2.imencode(ext, img)
    return base64.b64encode(buf).decode("utf-8")


def sniff_kind(data, is_video_hint):
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[4:8] == b"ftyp" or data[:4] == b"\x1aE\xdf\xa3" or data[:4] == b"RIFF":
        return "video"
    return "video" if is_video_hint else "image"


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def face_area(f):
    return (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])


def write_temp(data, suffix):
    t = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    t.write(data)
    t.close()
    return t.name


def gif_frames(data):
    im = Image.open(io.BytesIO(data))
    frames, durations = [], []
    for fr in ImageSequence.Iterator(im):
        durations.append(fr.info.get("duration", 80))
        frames.append(cv2.cvtColor(np.array(fr.convert("RGB")), cv2.COLOR_RGB2BGR))
    return frames, durations


def sample_video_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    if total > 0:
        idxs = np.linspace(0, total - 1, min(n, total)).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
    else:
        while len(frames) < n:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
    cap.release()
    return frames


def crop_face(img, f):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = f.bbox.astype(int)
    px, py = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return img[y1:y2, x1:x2]


def cluster_faces(frames):
    clusters = []
    for frame in frames:
        for f in app.get(frame):
            emb = f.embedding
            best, best_sim = None, -1
            for c in clusters:
                s = cosine(emb, c["embedding"])
                if s > best_sim:
                    best, best_sim = c, s
            if best is not None and best_sim >= CLUSTER_THRESH:
                if face_area(f) > best["area"]:
                    crop = crop_face(frame, f)
                    if crop.size > 0:
                        best.update(area=face_area(f), crop=crop, embedding=emb)
            else:
                crop = crop_face(frame, f)
                if crop.size > 0:
                    clusters.append({"embedding": emb, "area": face_area(f), "crop": crop})
    return [{
        "embedding": c["embedding"].tolist(),
        "thumbnail_base64": img_to_b64(c["crop"]),
    } for c in clusters]


def build_assignments(swaps):
    out = []
    for s in swaps:
        src = bytes_to_img(b64_to_bytes(s["new_face"]))
        if src is None:
            continue
        faces = app.get(src)
        if not faces:
            continue
        out.append({
            "target_emb": np.asarray(s["target_embedding"], dtype=np.float32),
            "source_face": max(faces, key=face_area),
        })
    return out


def process_frame(frame, assignments, enhance):
    for tf in app.get(frame):
        best, best_sim = None, -1
        for a in assignments:
            s = cosine(tf.embedding, a["target_emb"])
            if s > best_sim:
                best, best_sim = a, s
        if best is not None and best_sim >= MATCH_THRESH:
            frame = swapper.get(frame, tf, best["source_face"], paste_back=True)
    if enhance:
        g = get_gfpgan()
        if g:
            try:
                _, _, frame = g.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
            except Exception as e:
                print("[Worker] enhance failed:", e, flush=True)
    return frame


def handle_detect(media_b64, is_video):
    data = b64_to_bytes(media_b64)
    kind = sniff_kind(data, is_video)
    if kind == "image":
        img = bytes_to_img(data)
        if img is None:
            return {"error": "Could not decode image"}
        frames = [img]
    elif kind == "gif":
        all_frames, _ = gif_frames(data)
        idxs = np.linspace(0, len(all_frames) - 1, min(DETECT_SAMPLES, len(all_frames))).astype(int)
        frames = [all_frames[i] for i in idxs]
    else:
        path = write_temp(data, ".mp4")
        try:
            frames = sample_video_frames(path, DETECT_SAMPLES)
        finally:
            os.remove(path)
        if not frames:
            return {"error": "Could not read video frames"}
    return {"characters": cluster_faces(frames), "media_kind": kind}


def handle_swap(media_b64, is_video, swaps, pipeline):
    assignments = build_assignments(swaps)
    if not assignments:
        return {"error": "No usable face found in the replacement photos"}
    enhance = str(pipeline).lower() in ("pipeline2", "pipeline3", "gfpgan", "codeformer")
    data = b64_to_bytes(media_b64)
    kind = sniff_kind(data, is_video)

    if kind == "image":
        img = bytes_to_img(data)
        if img is None:
            return {"error": "Could not decode image"}
        img = process_frame(img, assignments, enhance)
        return {"media_type": "image", "base64": img_to_b64(img)}

    if kind == "gif":
        frames, durations = gif_frames(data)
        out = []
        for fr in frames:
            fr = process_frame(fr, assignments, enhance)
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        buf = io.BytesIO()
        out[0].save(buf, format="GIF", save_all=True, append_images=out[1:],
                    duration=durations, loop=0, disposal=2)
        return {"media_type": "gif", "base64": base64.b64encode(buf.getvalue()).decode("utf-8")}

    in_path = write_temp(data, ".mp4")
    raw_path = in_path + "_raw.mp4"
    final_path = in_path + "_final.mp4"
    try:
        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            writer.write(process_frame(fr, assignments, enhance))
        cap.release()
        writer.release()

        cmd = ["ffmpeg", "-y", "-i", raw_path, "-i", in_path,
               "-map", "0:v:0", "-map", "1:a:0?",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               "-c:a", "aac", "-shortest", "-movflags", "+faststart", final_path]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        target = final_path if (r.returncode == 0 and os.path.exists(final_path)) else raw_path
        with open(target, "rb") as f:
            out_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"media_type": "video", "base64": out_b64}
    finally:
        for p in (in_path, raw_path, final_path):
            if os.path.exists(p):
                os.remove(p)


def handler(job):
    job_input = job.get("input", {}) or {}
    action = job_input.get("action")
    media_b64 = (job_input.get("media") or job_input.get("target_image")
                 or job_input.get("target"))
    is_video = bool(job_input.get("is_video", False))

    if not media_b64:
        return {"error": "No media in payload", "keys_received": list(job_input.keys())}

    try:
        if action == "detect":
            return handle_detect(media_b64, is_video)
        if action == "swap":
            return handle_swap(media_b64, is_video, job_input.get("swaps", []),
                               job_input.get("pipeline", "fast"))
        return {"error": f"Invalid action: {action}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
