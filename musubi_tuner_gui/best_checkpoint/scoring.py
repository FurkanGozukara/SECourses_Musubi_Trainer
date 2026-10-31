"""Signals the best-checkpoint finder computes for every sample (all models local):

  face_mean / face_nearest   cosine of the largest face (OpenCV Zoo YuNet MIT + SFace Apache-2.0) to the mean training
                             face / to the nearest single training face; face_found
  dino_nearest_train         DINOv2 (Apache-2.0) cosine to the nearest training image (copying)
  scene_nearest_train        DINOv2 cosine of the background (the patch tokens outside the person) to the nearest training
                             background: the training photos' or clip's surroundings bleeding into other prompts' scenes
  prompt                     SigLIP2 (Apache-2.0) image-text cosine with the trigger replaced by its class word
  saturation, colourfulness, gray_std, clipped, hf_energy   colour and sharpness statistics ("fried" look)
Videos are scored on frames sampled across the clip (mean of the frame scores; temporal consistency = mean
DINOv2 cosine of consecutive sampled frames). Scores are cached per sample file, training set, subject and prompt in the
run's report folder."""

from __future__ import annotations

import hashlib
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
SIGLIP_TOKENS = 64  # SigLIP2's text length
PROMPT_CHUNKS = 4  # sentence groups of a long prompt that are scored (mean)
SCORES_VERSION = 6  # raise when a signal changes: the cached scores of other versions are computed again
# 6: the cache key holds the training set, subject and prompt (version 5 reused a sample's likeness scores when the run
# was analyzed again against another dataset)
# the person in a frame, for the background descriptor: the largest face's box widened to the shoulders and run down
# to the bottom of the frame (head and body), as shares of the face box; fewer background patches than this: none
PERSON_BOX = (0.4, 0.4, 1.4)  # left, above, right of the face box (x - 0.4 w, y - 0.4 h, x + 1.4 w)
MIN_BACKGROUND_PATCHES = 8


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


def person_patches(shape: tuple, box, grid: int, processor) -> np.ndarray:
    """grid x grid booleans, True on the person: the face box widened by PERSON_BOX and run down to the bottom of the
    frame, mapped through the DINOv2 processor's resize (shortest side) and centre crop. All False without a face."""
    mask = np.zeros((grid, grid), bool)
    if box is None:
        return mask
    h, w = shape[:2]
    size = getattr(processor, "size", None) or {}
    crop = getattr(processor, "crop_size", None) or {}
    short = size.get("shortest_edge") or min(h, w)
    crop_h, crop_w = crop.get("height") or short, crop.get("width") or short
    scale = short / min(h, w)
    off_x, off_y = (w * scale - crop_w) / 2, (h * scale - crop_h) / 2
    x, y, fw, fh = box
    left, above, right = PERSON_BOX
    x0, y0, x1, y1 = x - left * fw, y - above * fh, x + right * fw, h
    cell_w, cell_h = crop_w / grid, crop_h / grid
    c0, c1 = math.floor((x0 * scale - off_x) / cell_w), math.ceil((x1 * scale - off_x) / cell_w)
    r0, r1 = math.floor((y0 * scale - off_y) / cell_h), math.ceil((y1 * scale - off_y) / cell_h)
    mask[max(0, r0):max(0, r1), max(0, c0):max(0, c1)] = True
    return mask


class Scorer:
    """Loads its models on first use; ``device`` "cuda" or "cpu"."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._faces = None
        self._dino = None
        self._siglip = None

    # faces ----------------------------------------------------------------------------------------------------
    def face(self, rgb: np.ndarray):
        return self.face_and_area(rgb)[0]

    def face_and_area(self, rgb: np.ndarray):
        """The normalized SFace embedding of the largest face and the share of the image its box covers (None, None
        without a face)."""
        return self.face_area_box(rgb)[:2]

    def face_area_box(self, rgb: np.ndarray):
        """``face_and_area`` and the face box (x, y, width, height in pixels); (None, None, None) without a face."""
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
            return None, None, None
        face = max(faces, key=lambda f: f[2] * f[3])
        feature = recognizer.feature(recognizer.alignCrop(bgr, face)).reshape(-1).astype(np.float32)
        return (feature / (np.linalg.norm(feature) + 1e-12), float(face[2] * face[3]) / (bgr.shape[0] * bgr.shape[1]),
                tuple(float(v) for v in face[:4]))

    # DINOv2 ---------------------------------------------------------------------------------------------------
    def dino(self, images: list[np.ndarray]):
        return self.dino_and_background(images)[0]

    def dino_and_background(self, images: list[np.ndarray], boxes: list | None = None):
        """Normalized DINOv2 image embeddings and, with ``boxes`` (each frame's face box or None), background
        descriptors from the same forward: the patch tokens outside the person, each normalized (DINOv2 without
        registers has a few high-norm tokens in plain background areas), averaged; None where the person covers nearly
        the whole crop."""
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self._dino is None:
            path = _hf(DINO)
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self._dino = (AutoImageProcessor.from_pretrained(path), AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval(), dtype)
        processor, model, dtype = self._dino
        with torch.no_grad():
            inputs = processor(images=[Image.fromarray(i) for i in images], return_tensors="pt").to(self.device, dtype)
            output = model(**inputs)
        pooled = torch.nn.functional.normalize(output.pooler_output.float(), dim=-1).cpu()
        if boxes is None:
            return pooled, None
        tokens = output.last_hidden_state[:, 1:, :].float().cpu()
        grid = round(math.sqrt(tokens.shape[1]))
        backgrounds = []
        for image, box, patches in zip(images, boxes, tokens):
            keep = ~torch.from_numpy(person_patches(image.shape, box, grid, processor).reshape(-1))
            if int(keep.sum()) < MIN_BACKGROUND_PATCHES:
                backgrounds.append(None)
                continue
            mean = torch.nn.functional.normalize(patches[keep], dim=-1).mean(0)
            backgrounds.append(torch.nn.functional.normalize(mean, dim=0))
        return pooled, backgrounds

    # SigLIP2 --------------------------------------------------------------------------------------------------
    def _siglip_model(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        if self._siglip is None:
            path = _hf(SIGLIP)
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self._siglip = (AutoProcessor.from_pretrained(path), AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval(), dtype, {})
        return self._siglip

    def text_features(self, text: str):
        """Normalized SigLIP2 embeddings of the prompt's chunks (``prompt_chunks``), computed once per prompt."""
        import torch

        processor, model, _dtype, cache = self._siglip_model()
        if text not in cache:
            chunks = prompt_chunks(text, processor.tokenizer)
            with torch.no_grad():
                ids = processor(text=chunks, padding="max_length", max_length=SIGLIP_TOKENS, truncation=True, return_tensors="pt")["input_ids"]
                features = _pooled(model.get_text_features(input_ids=ids.to(self.device)))
            cache[text] = torch.nn.functional.normalize(features.float(), dim=-1).cpu()
        return cache[text]

    def prompt_scores(self, images: list[np.ndarray], text: str) -> list[float]:
        """Image-text cosine of each image with the prompt: the mean over the prompt's chunks, so a long prompt counts
        beyond the text encoder's 64 tokens."""
        import torch

        processor, model, dtype, _cache = self._siglip_model()
        chunks = self.text_features(text)
        with torch.no_grad():
            pixels = processor(images=[Image.fromarray(i) for i in images], return_tensors="pt")["pixel_values"].to(self.device, dtype)
            image = torch.nn.functional.normalize(_pooled(model.get_image_features(pixel_values=pixels)).float(), dim=-1).cpu()
        return (image @ chunks.t()).mean(dim=1).tolist()


def _pooled(output):
    """transformers 5 returns a model output from get_*_features (the projected embedding is its pooler_output)."""
    return output if not hasattr(output, "pooler_output") else output.pooler_output


def prompt_chunks(text: str, tokenizer, limit: int = SIGLIP_TOKENS, most: int = PROMPT_CHUNKS) -> list[str]:
    """The prompt as consecutive groups of whole sentences that each fit the text encoder (a longer sentence is cut
    there); at most ``most`` groups."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text.strip()) if s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        joined = f"{current} {sentence}".strip()
        if current and len(tokenizer(joined)["input_ids"]) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = joined
    if current:
        chunks.append(current)
    return chunks[:most] or [text]


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
    batches so a large dataset never sits in memory at once, their median face box area (the framing an
    overtrained model drifts towards) and their background descriptors (the surroundings it copies)."""
    faces, areas, dino, backgrounds, pending, boxes, count = [], [], [], [], [], [], 0
    fingerprint = hashlib.sha1()  # the files' path, size and time: part of every cached sample score's key
    for path in paths:
        try:
            stat = os.stat(path)
            fingerprint.update(f"{os.path.abspath(path)}|{stat.st_size}|{int(stat.st_mtime)}\n".encode("utf-8"))
        except OSError:
            fingerprint.update(f"{os.path.abspath(path)}|missing\n".encode("utf-8"))

    def flush():
        pooled, found = scorer.dino_and_background(pending, boxes)
        dino.append(pooled)
        backgrounds.extend(b for b in found if b is not None)
        pending.clear()
        boxes.clear()

    for path in paths:
        for frame in _frames_of(path, 3):
            count += 1
            face, area, box = scorer.face_area_box(frame)
            if face is not None:
                faces.append(face)
                areas.append(area)
            pending.append(frame)
            boxes.append(box)
            if len(pending) == batch:
                flush()
    if pending:
        flush()
    mean_face = None
    if faces:
        mean_face = np.mean(faces, axis=0)
        mean_face = mean_face / (np.linalg.norm(mean_face) + 1e-12)
    import torch

    return {"count": count, "faces": faces, "mean_face": mean_face, "dino": torch.cat(dino) if dino else None,
            "background": torch.stack(backgrounds) if backgrounds else None, "face_area": statistics.median(areas) if areas else None,
            "fingerprint": fingerprint.hexdigest()}


def score_samples(scorer: Scorer, samples: list[dict], prompts: list[str], references: dict, subject: str, class_word: str,
                  cache: dict) -> list[dict]:
    """Per-sample signals (cached in ``cache`` by path + size + mtime and by what they were scored against: the training
    set, subject and prompt); ``prompts[i]`` is the prompt of sample line i."""
    results = []
    context = f"{references.get('fingerprint', '')}|{subject}|{class_word}"
    for item in samples:
        stat = os.stat(item["path"])
        prompt = prompts[item["prompt_index"]] if item["prompt_index"] < len(prompts) else ""
        against = hashlib.sha1(f"{context}|{prompt}".encode("utf-8")).hexdigest()[:16]
        key = f"{item['path']}|{stat.st_size}|{int(stat.st_mtime)}|{against}"
        if key not in cache:
            cache[key] = _score_one(scorer, item, prompts, references, subject, class_word)
        results.append({**item, **cache[key]})
    return results


def _score_one(scorer: Scorer, item: dict, prompts: list[str], references: dict, subject: str, class_word: str) -> dict:
    frames = _frames_of(item["path"], VIDEO_FRAMES)
    if not frames:
        return {"error": "unreadable"}
    prompt = prompts[item["prompt_index"]] if item["prompt_index"] < len(prompts) else ""
    detected = [scorer.face_area_box(frame) for frame in frames]
    dino, backgrounds = scorer.dino_and_background(frames, [box for _face, _area, box in detected])
    stats = [colour_stats(frame) for frame in frames]
    scores = {key: statistics.mean(s[key] for s in stats) for key in stats[0]}
    if prompt:
        scores["prompt"] = float(statistics.mean(scorer.prompt_scores(frames, clean_prompt(prompt, subject, class_word))))
    faces = [(face, area) for face, area, _box in detected if face is not None]
    scores["face_found"] = len(faces) / len(frames)
    found = [face for face, _area in faces]
    if faces:  # framing: an overtrained subject is drawn as close as in its training photos, whatever the prompt asks
        scores["face_area"] = float(statistics.mean(area for _face, area in faces))
    if found and references["mean_face"] is not None:
        scores["face_mean"] = float(statistics.mean(float(f @ references["mean_face"]) for f in found))
        scores["face_nearest"] = float(max(max(float(f @ r) for r in references["faces"]) for f in found))
    if found:  # for the identity margin against the base model's face of the same prompt and seed
        mean = np.mean(found, axis=0)
        scores["face_embedding"] = (mean / (np.linalg.norm(mean) + 1e-12)).astype(np.float32).round(5).tolist()
    if references["dino"] is not None:
        scores["dino_nearest_train"] = float(statistics.mean(float((references["dino"] @ d).max()) for d in dino))
    nearest_scene = [float((references["background"] @ b).max()) for b in backgrounds if b is not None] if references.get("background") is not None else []
    if nearest_scene:
        scores["scene_nearest_train"] = float(statistics.mean(nearest_scene))
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

    summary = {key: median(key) for key in ("face_mean", "face_nearest", "face_area", "dino_nearest_train", "scene_nearest_train", "prompt",
                                            "saturation", "colourfulness", "gray_std", "clipped", "hf_energy", "temporal")}
    nearest = [s["dino_nearest_train"] for s in samples if s.get("dino_nearest_train") is not None]
    summary["dino_nearest_max"] = max(nearest) if nearest else None  # the sample nearest to a training image (copying)
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
    """Sample file -> scores; scores of another SCORES_VERSION (or a file of the first, unversioned format) are dropped."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data.get("scores", {}) if isinstance(data, dict) and data.get("version") == SCORES_VERSION else {}


def save_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump({"version": SCORES_VERSION, "scores": cache}, handle)
    os.replace(path + ".tmp", path)


def identity_margin(samples: list[dict], base_samples: list[dict], mean_face) -> tuple[float, float] | None:
    """How far each face moved from the base model's identity towards the subject: its similarity to the mean training
    face minus its similarity to the base model's face for the same prompt and seed (median), and the share of pairs
    where the subject's side wins. SFace still rates a base-model face that only took the subject's glasses and hair
    0.55-0.65 against the training faces (DoRA at half its learning rate, epochs 60-80 of a Krea 2 run); the margin
    jumps where the face visibly becomes the subject's. None without pairs."""
    if mean_face is None:
        return None
    base = {(s["prompt_index"], s.get("seed", "")): s.get("face_embedding") for s in base_samples}
    margins = []
    for sample in samples:
        before, face = base.get((sample["prompt_index"], sample.get("seed", ""))), sample.get("face_embedding")
        if before is None or face is None:
            continue
        face = np.asarray(face, dtype=np.float32)
        margins.append(float(face @ mean_face) - float(face @ np.asarray(before, dtype=np.float32)))
    if not margins:
        return None
    return statistics.median(margins), sum(1 for margin in margins if margin > 0) / len(margins)


def paired_change(samples: list[dict], base_samples: list[dict], key: str, ratio: bool = False) -> tuple[float, float] | None:
    """How a signal moved from the base model's sample of the same prompt and seed: the median change (log ratio for
    scale-like signals) and the share of pairs that moved that way. Pairing removes the differences between prompts,
    which are far larger than a checkpoint's own change (sharpness differs 30x between a studio portrait and a starship
    bridge). None without pairs."""
    base = {(s["prompt_index"], s.get("seed", "")): s.get(key) for s in base_samples}
    changes = []
    for sample in samples:
        before, value = base.get((sample["prompt_index"], sample.get("seed", ""))), sample.get(key)
        if before is None or value is None:
            continue
        if ratio:
            if before > 0 and value > 0:
                changes.append(math.log(value / before))
        else:
            changes.append(value - before)
    if not changes:
        return None
    middle = statistics.median(changes)
    agree = sum(1 for change in changes if change and (change > 0) == (middle > 0)) / len(changes)
    return middle, agree
