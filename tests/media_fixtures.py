"""Tiny real media files for dataset tests (written with PyAV on the CPU; no GPU, no downloads)."""

from __future__ import annotations

import math
import os

import av
import numpy as np
from PIL import Image


def write_image(path: str, size=(64, 64), color=(80, 120, 200)) -> str:
    Image.new("RGB", size, color).save(path)
    return path


def _tone(rate: int, seconds: float, channels: int) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    wave = (0.3 * np.sin(2 * math.pi * 440.0 * t) * 32767).astype(np.int16)
    return np.stack([wave] * channels, axis=0)


def write_wav(path: str, seconds: float = 0.5, rate: int = 16000, channels: int = 1) -> str:
    samples = _tone(rate, seconds, channels)
    layout = "mono" if channels == 1 else "stereo"
    with av.open(path, "w", format="wav") as container:
        stream = container.add_stream("pcm_s16le", rate=rate, layout=layout)
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1) if channels == 1 else samples.T.reshape(1, -1), format="s16", layout=layout)
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def write_video(path: str, frames: int = 9, size=(64, 64), fps: int = 24, audio_seconds: float | None = None, rate: int = 48000) -> str:
    """MP4 with H.264 video and, when ``audio_seconds`` is set, a stereo AAC track."""
    with av.open(path, "w") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height = size
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("aac", rate=rate, layout="stereo") if audio_seconds else None
        for index in range(frames):
            pixels = np.full((size[1], size[0], 3), (index * 20) % 255, dtype=np.uint8)
            for packet in video.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        if audio is not None:
            samples = _tone(rate, audio_seconds, 2).astype(np.float32) / 32767.0
            chunk = 1024
            for offset in range(0, samples.shape[1], chunk):
                part = np.ascontiguousarray(samples[:, offset : offset + chunk])
                frame = av.AudioFrame.from_ndarray(part, format="fltp", layout="stereo")
                frame.sample_rate = rate
                frame.pts = offset
                for packet in audio.encode(frame):
                    container.mux(packet)
            for packet in audio.encode(None):
                container.mux(packet)
    return path


def audio_seconds(path: str) -> float:
    with av.open(path) as container:
        stream = next(item for item in container.streams if item.type == "audio")
        samples = sum(frame.samples for frame in container.decode(stream))
        return samples / float(stream.rate)


def caption(path: str, text: str, extension: str = ".txt") -> str:
    target = os.path.splitext(path)[0] + extension
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text)
    return target
