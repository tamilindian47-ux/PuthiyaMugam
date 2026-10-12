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

face_app = FaceAnalysis(name="buffalo_l", providers=providers)
face_app.prepare(ctx_id=0, det_size=(640, 640))

SWAPPER_PATH = os.environ.get("SWAPPER_PATH", "/models/inswapper_128.onnx")
if not os.path.exists(SWAPPER_PATH):
    SWAPPER_PATH = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")

swapper = insightface.model_zoo.get_model(
    SWAPPER_PATH,
    download=False,
    providers=providers
)

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

print("[Worker] Successfully Initialized and Ready for Jobs.")


def strip_data_uri_prefix(base64_str: str) -> str:
    if "," in base64_str:
        return base64_str.split(",", 1)[1]
    return base64_str


def decode_base64_bytes(base64_str: str) -> bytes:
    clean_b64 = strip_data_uri_prefix(base64_str)
    return base64.b64decode(clean_b64)


def decode_base64_img(base64_str: str) -> np.ndarray:
    img_data = decode_base64_bytes(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def decode_media_frame(base64_str: str) -> np.ndarray:
    media_bytes = decode_base64_bytes(base64_str)

    np_arr = np.frombuffer(media_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is not None and getattr(img, "size", 0) > 0:
        return img

    temp_path = None
    for suffix in [".mp4", ".mov", ".gif", ".webm", ".bin"]:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(media_bytes)
                temp_path = tmp.name

            cap = cv2.VideoCapture(temp_path)
            ok, frame = cap.read()
            cap.release()

            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                return frame
        except Exception:
            pass
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            temp_path = None

    return None


def encode_img_base64(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    ok, buffer = cv2.imencode(ext, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("Failed to encode output image.")
    return base64.b64encode(buffer).decode("utf-8")


def detect_faces(img_bgr: np.ndarray):
    if img_bgr is None or not hasattr(img_bgr, "shape") or img_bgr.size == 0:
        raise ValueError("detect_faces received an empty or invalid image.")

    faces = face_app.get(img_bgr)
    results = []

    h, w = img_bgr.shape[:2]

    for idx, face in enumerate(faces):
        bbox = [int(v) for v in face.bbox.tolist()]
        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(w, bbox[2])
        y2 = min(h, bbox[3])

        crop = img_bgr[y1:y2, x1:x2]
        crop_b64 = encode_img_base64(crop) if crop is not None and crop.size > 0 else ""

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


def pick_face(face_list, idx: int):
    if not face_list:
        return None
    idx = max(0, min(int(idx), len(face_list) - 1))
    return face_list[idx]


def handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action", "swap")

    if "target_image" not in job_input:
        return {"error": "Missing 'target_image' in request payload."}

    try:
        if action == "detect":
            target_img = decode_media_frame(job_input["target_image"])
        else:
            target_img = decode_base64_img(job_input["target_image"])
            if target_img is None:
                target_img = decode_media_frame(job_input["target_image"])
    except Exception as e:
        return {"error": f"Failed decoding target_image: {str(e)}"}

    if target_img is None:
        return {"error": "target_image could not be decoded as an image or video frame."}

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

    elif action == "swap":
        if "source_image" not in job_input:
            return {"error": "Missing 'source_image' for swap operation."}

        try:
            source_img = decode_base64_img(job_input["source_image"])
        except Exception as e:
            return {"error": f"Failed decoding source_image: {str(e)}"}

        if source_img is None:
            return {"error": "source_image could not be decoded as a valid image."}

        source_faces = face_app.get(source_img)
        if not source_faces:
            return {"error": "No face found in source image."}

        target_faces = face_app.get(target_img)
        if not target_faces:
            return {"error": "No face found in target image."}

        pipeline = job_input.get("pipeline", "fast").lower()

        faces_map = job_input.get("faces_map", None)
        output_img = target_img.copy()

        if faces_map and isinstance(faces_map, list):
            for mapping in faces_map:
                try:
                    target_index = int(mapping.get("target_index", 0))
                    src_b64 = mapping.get("source_image")

                    if not src_b64:
                        continue

                    mapped_source_img = decode_base64_img(src_b64)
                    if mapped_source_img is None:
                        continue

                    mapped_source_faces = face_app.get(mapped_source_img)
                    if not mapped_source_faces:
                        continue

                    src_face = pick_face(mapped_source_faces, mapping.get("source_face_index", 0))
                    tgt_face = pick_face(target_faces, target_index)

                    if src_face is None or tgt_face is None:
                        continue

                    output_img = swapper.get(output_img, tgt_face, src_face, paste_back=True)
                except Exception as e:
                    print(f"[Warning] faces_map item skipped: {e}")
                    continue
        else:
            s_idx = min(int(job_input.get("source_face_index", 0)), len(source_faces) - 1)
            t_idx = min(int(job_input.get("target_face_index", 0)), len(target_faces) - 1)

            source_face = source_faces[s_idx]
            target_face = target_faces[t_idx]

            output_img = swapper.get(output_img, target_face, source_face, paste_back=True)

        if pipeline in ["gfpgan", "codeformer"]:
            output_img = apply_enhancer(output_img, pipeline)

        result_b64 = encode_img_base64(output_img)
        return {
            "status": "success",
            "pipeline_used": pipeline,
            "image": f"data:image/jpeg;base64,{result_b64}"
        }

    return {"error": f"Unknown action '{action}'"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
