"""Signals the best-checkpoint finder computes for every sample (all models local, permissive licenses):

  face_mean / face_nearest   cosine of the largest face (OpenCV Zoo YuNet MIT + SFace Apache-2.0) to the mean training
                             face / to the nearest single training face; face_found
  dino_nearest_train         DINOv2 (Apache-2.0) cosine to the nearest training image (copying)
  prompt                     SigLIP2 (Apache-2.0) image-text cosine with the trigger replaced by its class word
  saturation, colourfulness, gray_std, clipped, hf_energy   colour and sharpness statistics ("fried" look)
Videos are scored on frames sampled across the clip (mean of the frame scores; temporal consistency = mean
DINOv2 cosine of consecutive sampled frames). Scores are cached per sample file in the run's report folder."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import urllib.request

import numpy as np
from PIL import Image

YUNET = ("face_detection_yunet_2023mar.onnx", "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE = ("face_recognition_sface_2021dec.onnx", "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx")
DINO = "facebook/dinov2-base"
SIGLIP = "google/siglip2-so400m-patch14-384"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".avi")
VIDEO_FRAMES = 8


def models_dir() -> str:
    """<install root>/Training_Models_Checkpoint_Finder (next to the other Training_Models_* folders)."""
    app = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.environ.get("MUSUBI_FINDER_MODELS", os.path.join(os.path.dirname(app), "Training_Models_Checkpoint_Finder"))


def _fetch(name_url) -> str:
    name, url = name_url
    path = os.path.join(models_dir(), name)
    if not os.path.isfile(path):
        os.makedirs(models_dir(), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def _hf(repo: str) -> str:
    """The model folder: downloaded once, then read without asking the Hub (the finder also runs offline)."""
    target = os.path.join(models_dir(), repo.replace("/", "__"))
    if os.path.isfile(os.path.join(target, "config.json")) and any(name.endswith(".safetensors") for name in os.listdir(target)):
        return target
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo} to {target} (first use of the best checkpoint finder)", flush=True)
    return snapshot_download(repo, local_dir=target, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.spm"])


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def video_frames(path: str, count: int = VIDEO_FRAMES) -> list[np.ndarray]:
    """``count`` frames spread over the clip (PyAV); only the picked frames are kept in memory."""
    import av

    with av.open(path) as container:
        total = container.streams.video[0].frames or sum(1 for packet in container.demux(video=0) if packet.size)
    if total <= 0:
        return []
    picks = {round(i * (total - 1) / max(1, count - 1)) for i in range(count)}
    frames = []
    with av.open(path) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in picks:
                frames.append(frame.to_ndarray(format="rgb24"))
            if index >= max(picks):
                break
    return frames


def colour_stats(rgb: np.ndarray) -> dict:
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "saturation": float(hsv[..., 1].mean() / 255.0),
        "colourfulness": float(math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)),
        "gray_std": float(gray.std()),
        "clipped": float(((gray <= 2) | (gray >= 253)).mean()),
        "hf_energy": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
    }


class Scorer:
    """Loads its models on first use; ``device`` "cuda" or "cpu"."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._faces = None
        self._dino = None
        self._siglip = None

    # faces ----------------------------------------------------------------------------------------------------
    def face(self, rgb: np.ndarray):
        import cv2

        if self._faces is None:
            self._faces = (
                cv2.FaceDetectorYN.create(_fetch(YUNET), "", (320, 320), 0.8, 0.3, 5000),
                cv2.FaceRecognizerSF.create(_fetch(SFACE), ""),
            )
        detector, recognizer = self._faces
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        feature = recognizer.feature(recognizer.alignCrop(bgr, face)).reshape(-1).astype(np.float32)
        return feature / (np.linalg.norm(feature) + 1e-12)

    # DINOv2 ---------------------------------------------------------------------------------------------------
    def dino(self, images: list[np.ndarray]):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self._dino is None:
            path = _hf(DINO)
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self._dino = (AutoImageProcessor.from_pretrained(path), AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval(), dtype)
        processor, model, dtype = self._dino
        with torch.no_grad():
            inputs = processor(images=[Image.fromarray(i) for i in images], return_tensors="pt").to(self.device, dtype)
            pooled = model(**inputs).pooler_output.float()
        return torch.nn.functional.normalize(pooled, dim=-1).cpu()

    # SigLIP2 --------------------------------------------------------------------------------------------------
    def prompt_scores(self, images: list[np.ndarray], texts: list[str]) -> list[float]:
        import torch
        from transformers import AutoModel, AutoProcessor

        if self._siglip is None:
            path = _hf(SIGLIP)
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self._siglip = (AutoProcessor.from_pretrained(path), AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval(), dtype)
        processor, model, dtype = self._siglip
        with torch.no_grad():
            inputs = processor(text=texts, images=[Image.fromarray(i) for i in images], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            inputs = {k: (v.to(self.device, dtype) if v.is_floating_point() else v.to(self.device)) for k, v in inputs.items()}
            out = model(**inputs)
            image = torch.nn.functional.normalize(out.image_embeds.float(), dim=-1)
            text = torch.nn.functional.normalize(out.text_embeds.float(), dim=-1)
        return (image * text).sum(-1).cpu().tolist()


def clean_prompt(prompt: str, subject: str, class_word: str) -> str:
    text = re.sub(r"<lora:[^>]*>", "", prompt)
    return re.sub(re.escape(subject), class_word, text, flags=re.I) if subject else text


def dataset_files(folder: str) -> list[str]:
    """Training images and videos of a dataset folder (recursively; cache folders skipped)."""
    paths = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.lower().startswith(("cache", "."))]
        paths += [os.path.join(root, f) for f in files if f.lower().endswith(IMAGE_EXT + VIDEO_EXT)]
    return sorted(paths)


def _frames_of(path: str, video_frames_count: int) -> list[np.ndarray]:
    try:
        if path.lower().endswith(VIDEO_EXT):
            return video_frames(path, video_frames_count)
        return [load_rgb(path)]
    except Exception:  # noqa: BLE001 - an unreadable training file is skipped, as the trainers' decode check reports it
        return []


def reference_set(scorer: Scorer, paths: list[str], batch: int = 16) -> dict:
    """Face and DINOv2 embeddings of the training images (three frames of each training video), streamed in
    batches so a large dataset never sits in memory at once."""
    faces, dino, pending, count = [], [], [], 0
    for path in paths:
        for frame in _frames_of(path, 3):
            count += 1
            face = scorer.face(frame)
            if face is not None:
                faces.append(face)
            pending.append(frame)
            if len(pending) == batch:
                dino.append(scorer.dino(pending))
                pending = []
    if pending:
        dino.append(scorer.dino(pending))
    mean_face = None
    if faces:
        mean_face = np.mean(faces, axis=0)
        mean_face = mean_face / (np.linalg.norm(mean_face) + 1e-12)
    import torch

    return {"count": count, "faces": faces, "mean_face": mean_face, "dino": torch.cat(dino) if dino else None}


def score_samples(scorer: Scorer, samples: list[dict], prompts: list[str], references: dict, subject: str, class_word: str,
                  cache: dict) -> list[dict]:
    """Per-sample signals (cached by path + size + mtime in ``cache``); ``prompts[i]`` is the prompt of sample line i."""
    results = []
    for item in samples:
        stat = os.stat(item["path"])
        key = f"{item['path']}|{stat.st_size}|{int(stat.st_mtime)}"
        if key not in cache:
            cache[key] = _score_one(scorer, item, prompts, references, subject, class_word)
        results.append({**item, **cache[key]})
    return results


def _score_one(scorer: Scorer, item: dict, prompts: list[str], references: dict, subject: str, class_word: str) -> dict:
    frames = _frames_of(item["path"], VIDEO_FRAMES)
    if not frames:
        return {"error": "unreadable"}
    prompt = prompts[item["prompt_index"]] if item["prompt_index"] < len(prompts) else ""
    dino = scorer.dino(frames)
    stats = [colour_stats(frame) for frame in frames]
    scores = {key: statistics.mean(s[key] for s in stats) for key in stats[0]}
    if prompt:
        texts = [clean_prompt(prompt, subject, class_word)] * len(frames)
        scores["prompt"] = float(statistics.mean(scorer.prompt_scores(frames, texts)))
    found = [f for f in (scorer.face(frame) for frame in frames) if f is not None]
    scores["face_found"] = len(found) / len(frames)
    if found and references["mean_face"] is not None:
        scores["face_mean"] = float(statistics.mean(float(f @ references["mean_face"]) for f in found))
        scores["face_nearest"] = float(max(max(float(f @ r) for r in references["faces"]) for f in found))
    if references["dino"] is not None:
        scores["dino_nearest_train"] = float(statistics.mean(float((references["dino"] @ d).max()) for d in dino))
    if len(frames) > 1:
        scores["temporal"] = float(statistics.mean(float(dino[i] @ dino[i + 1]) for i in range(len(frames) - 1)))
    scores["dino_embedding"] = dino.mean(0).tolist()
    return scores


def summarize(samples: list[dict]) -> dict:
    """Checkpoint summary: medians of the sample signals, variety (1 - mean pairwise DINOv2 cosine between different
    prompts with the same seed) and the share of samples with a face."""
    import torch

    def median(key):
        values = [s[key] for s in samples if s.get(key) is not None]
        return statistics.median(values) if values else None

    summary = {key: median(key) for key in ("face_mean", "face_nearest", "dino_nearest_train", "prompt", "saturation",
                                            "colourfulness", "gray_std", "clipped", "hf_energy", "temporal")}
    found = [s.get("face_found") for s in samples if s.get("face_found") is not None]
    summary["face_found"] = statistics.mean(found) if found else None
    by_seed: dict[str, list] = {}
    for s in samples:
        if s.get("dino_embedding"):
            by_seed.setdefault(s.get("seed", ""), []).append(torch.tensor(s["dino_embedding"]))
    pair = []
    for embeddings in by_seed.values():
        if len(embeddings) > 1:
            stack = torch.nn.functional.normalize(torch.stack(embeddings), dim=-1)
            cos = stack @ stack.t()
            n = len(embeddings)
            pair.append(float((cos.sum() - n) / (n * (n - 1))))
    summary["variety"] = 1.0 - statistics.mean(pair) if pair else None
    summary["samples"] = len(samples)
    return summary


def load_cache(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def save_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(cache, handle)
    os.replace(path + ".tmp", path)
