"""
Helper utility to automatically download MediaPipe Pose Landmarker models if missing.
"""
import os
import urllib.request

MODELS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}

def ensure_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for fname, url in MODELS.items():
        dst = os.path.join(base_dir, fname)
        if not os.path.exists(dst):
            print(f"Downloading {fname} from {url}...")
            urllib.request.urlretrieve(url, dst)
            print(f"Saved {fname} ({os.path.getsize(dst)} bytes).")
        else:
            print(f"Found {fname} ({os.path.getsize(dst)} bytes).")

if __name__ == "__main__":
    ensure_models()
