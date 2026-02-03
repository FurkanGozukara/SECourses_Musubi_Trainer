#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


DTYPE_NBYTES: Dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: List[int]
    nbytes: int
    src_path: Path
    src_data_start: int
    src_offsets: Tuple[int, int]  # offsets relative to src_data_start


def _human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(f)} {u}"
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{n} B"


def _prod(shape: Iterable[int]) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out


def tensor_nbytes(dtype: str, shape: List[int]) -> int:
    if dtype not in DTYPE_NBYTES:
        raise ValueError(f"Unsupported dtype {dtype!r}. Known: {sorted(DTYPE_NBYTES)}")
    if any(int(d) < 0 for d in shape):
        raise ValueError(f"Invalid shape with negative dims: {shape}")
    return _prod(shape) * DTYPE_NBYTES[dtype]


def read_safetensors_header(path: Path) -> Tuple[int, Dict[str, Any], int, int]:
    with path.open("rb") as f:
        header_len_raw = f.read(8)
        if len(header_len_raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (too small)")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path}: truncated header (expected {header_len} bytes)")
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"{path}: invalid JSON header: {e}") from e
    file_size = path.stat().st_size
    data_start = 8 + header_len
    if data_start > file_size:
        raise ValueError(f"{path}: invalid header_len={header_len} (past EOF)")
    if not isinstance(header, dict):
        raise ValueError(f"{path}: invalid header type {type(header)} (expected object)")
    return header_len, header, data_start, file_size


def iter_tensor_entries(
    src_path: Path,
    src_data_start: int,
    header: Dict[str, Any],
) -> Iterator[TensorEntry]:
    for name, v in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(v, dict):
            raise ValueError(f"{src_path}: {name}: invalid entry type {type(v)}")
        dtype = v.get("dtype")
        shape = v.get("shape")
        offsets = v.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
            raise ValueError(f"{src_path}: {name}: missing/invalid dtype/shape/data_offsets")
        if len(offsets) != 2:
            raise ValueError(f"{src_path}: {name}: data_offsets must have 2 items")
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end < start:
            raise ValueError(f"{src_path}: {name}: invalid data_offsets {offsets}")
        shape_ints = [int(d) for d in shape]
        nbytes = tensor_nbytes(dtype, shape_ints)
        if (end - start) != nbytes:
            raise ValueError(
                f"{src_path}: {name}: offset span {end-start} != expected nbytes {nbytes} "
                f"(dtype={dtype}, shape={shape_ints})"
            )
        yield TensorEntry(
            name=name,
            dtype=dtype,
            shape=shape_ints,
            nbytes=nbytes,
            src_path=src_path,
            src_data_start=src_data_start,
            src_offsets=(start, end),
        )


def merge_metadata(metadatas: List[Optional[Dict[str, str]]], conflict: str) -> Optional[Dict[str, str]]:
    merged: Dict[str, str] = {}
    for md in metadatas:
        if not md:
            continue
        if not isinstance(md, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in md.items()):
            raise ValueError("__metadata__ must be an object of string->string")
        for k, v in md.items():
            if k in merged and merged[k] != v:
                if conflict == "error":
                    raise ValueError(f"Metadata conflict for key {k!r}: {merged[k]!r} vs {v!r}")
                if conflict == "first":
                    continue
                if conflict == "last":
                    merged[k] = v
                    continue
                raise ValueError(f"Unknown metadata conflict mode: {conflict}")
            merged[k] = v
    return merged or None


def find_index_file(root: Path) -> Optional[Path]:
    candidates = sorted(root.glob("*.safetensors.index.json"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def files_from_index(index_path: Path) -> List[Path]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "weight_map" not in data:
        raise ValueError(f"{index_path}: not a valid *.safetensors.index.json (missing weight_map)")
    weight_map = data["weight_map"]
    if not isinstance(weight_map, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in weight_map.items()):
        raise ValueError(f"{index_path}: invalid weight_map")
    files = sorted({(index_path.parent / v).resolve() for v in weight_map.values()})
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{index_path}: missing shard files: {missing}")
    return files


def ensure_disk_space(out_path: Path, required_bytes: int) -> None:
    usage = shutil.disk_usage(out_path.parent)
    if usage.free < required_bytes:
        raise RuntimeError(
            f"Not enough free disk space in {out_path.parent}.\n"
            f"Required: {required_bytes} ({_human_bytes(required_bytes)})\n"
            f"Free: {usage.free} ({_human_bytes(usage.free)})"
        )


def build_combined_header(
    entries: List[TensorEntry],
    metadata: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    offset = 0
    for e in entries:
        header[e.name] = {
            "dtype": e.dtype,
            "shape": e.shape,
            "data_offsets": [offset, offset + e.nbytes],
        }
        offset += e.nbytes

    return header


def verify_safetensors_file(path: Path) -> None:
    header_len, header, data_start, file_size = read_safetensors_header(path)
    max_end = 0
    for name, v in header.items():
        if name == "__metadata__":
            continue
        end = int(v["data_offsets"][1])
        if end > max_end:
            max_end = end
    expected_size = data_start + max_end
    if expected_size != file_size:
        raise ValueError(
            f"{path}: size mismatch: file_size={file_size}, header_len={header_len}, "
            f"max_end={max_end} => expected_size={expected_size}"
        )


def combine_safetensors(
    shard_files: List[Path],
    out_path: Path,
    overwrite: bool,
    metadata_conflict: str,
    chunk_size: int,
    quiet: bool,
    verify: bool,
    write_index: Optional[Path],
    dry_run: bool,
    skip_disk_check: bool,
) -> None:
    out_path = out_path.resolve()
    tmp_path = Path(str(out_path) + ".tmp")

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out_path} (use --overwrite)")
    if tmp_path.exists():
        if overwrite:
            tmp_path.unlink()
        else:
            raise FileExistsError(f"Temp output already exists: {tmp_path} (use --overwrite)")

    shard_files = [p.resolve() for p in shard_files]
    shard_files = [p for p in shard_files if p != out_path and p != tmp_path]
    if not shard_files:
        raise ValueError("No shard files found.")

    file_order = {p: i for i, p in enumerate(shard_files)}

    all_entries: List[TensorEntry] = []
    all_metadata: List[Optional[Dict[str, str]]] = []
    seen: Dict[str, Path] = {}

    for p in shard_files:
        _, header, data_start, file_size = read_safetensors_header(p)
        md = header.get("__metadata__")
        all_metadata.append(md if isinstance(md, dict) else None)

        # Structural sanity: max end should match data section length for valid files
        max_end = 0
        for k, v in header.items():
            if k == "__metadata__":
                continue
            end = int(v["data_offsets"][1])
            if end > max_end:
                max_end = end
        if data_start + max_end != file_size:
            raise ValueError(f"{p}: invalid file structure (max tensor end does not match file size)")

        for e in iter_tensor_entries(p, data_start, header):
            if e.name in seen:
                raise ValueError(f"Duplicate tensor key {e.name!r} in {e.src_path} (already in {seen[e.name]})")
            seen[e.name] = e.src_path
            all_entries.append(e)

    # Sort to minimize seeks: by shard order, then by on-disk offset
    all_entries.sort(key=lambda e: (file_order[e.src_path], e.src_offsets[0]))

    merged_metadata = merge_metadata(all_metadata, conflict=metadata_conflict)
    header_obj = build_combined_header(all_entries, merged_metadata)
    header_bytes = json.dumps(header_obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    total_data = sum(e.nbytes for e in all_entries)
    required = 8 + len(header_bytes) + total_data
    if not skip_disk_check:
        ensure_disk_space(out_path, required)

    if not quiet:
        print(f"Shards: {len(shard_files)}")
        print(f"Tensors: {len(all_entries)}")
        print(f"Output: {out_path}")
        print(f"Data: {_human_bytes(total_data)}")
        print(f"Header: {_human_bytes(len(header_bytes))}")
        print(f"Total: {_human_bytes(required)}")

    if dry_run:
        return

    started = time.monotonic()
    last_report_t = started
    copied = 0

    src_handles: Dict[Path, Any] = {}
    try:
        for p in shard_files:
            src_handles[p] = p.open("rb")

        with tmp_path.open("wb", buffering=1024 * 1024) as out:
            out.write(struct.pack("<Q", len(header_bytes)))
            out.write(header_bytes)

            current_file: Optional[Path] = None
            for e in all_entries:
                if current_file != e.src_path:
                    current_file = e.src_path
                    if not quiet:
                        print(f"Copying from {current_file.name} ...")

                f = src_handles[e.src_path]
                f.seek(e.src_data_start + e.src_offsets[0])

                remaining = e.nbytes
                while remaining:
                    buf = f.read(min(chunk_size, remaining))
                    if not buf:
                        raise IOError(f"{e.src_path}: unexpected EOF while reading tensor {e.name}")
                    out.write(buf)
                    remaining -= len(buf)
                    copied += len(buf)

                    now = time.monotonic()
                    if (now - last_report_t) >= 5.0 and not quiet:
                        elapsed = now - started
                        rate = copied / elapsed if elapsed > 0 else 0.0
                        pct = (copied / total_data * 100.0) if total_data else 100.0
                        print(f"  {pct:6.2f}%  {_human_bytes(copied)} / {_human_bytes(total_data)}  ({_human_bytes(int(rate))}/s)")
                        last_report_t = now

        os.replace(tmp_path, out_path)

        if write_index:
            index_path = write_index.resolve()
            weight_map = {e.name: out_path.name for e in all_entries}
            total_parameters = sum(_prod(e.shape) for e in all_entries)
            index_obj = {
                "metadata": {"total_parameters": total_parameters, "total_size": total_data},
                "weight_map": weight_map,
            }
            index_path.write_text(json.dumps(index_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if verify:
            verify_safetensors_file(out_path)

    finally:
        for f in src_handles.values():
            try:
                f.close()
            except Exception:
                pass
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def find_shards_without_index(root: Path, out_path: Path) -> List[Path]:
    out_path = out_path.resolve()
    tmp_path = Path(str(out_path) + ".tmp")
    shards = []
    for p in sorted(root.rglob("*.safetensors")):
        rp = p.resolve()
        if rp == out_path or rp == tmp_path:
            continue
        shards.append(rp)
    return shards


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine sharded .safetensors into a single .safetensors without loading tensors into RAM."
    )
    parser.add_argument("--out", type=Path, default=Path("model.safetensors"), help="Output .safetensors path")
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Optional *.safetensors.index.json to select shard files (auto-detected if present)",
    )
    parser.add_argument("--no-index", action="store_true", help="Ignore any auto-detected index file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    parser.add_argument(
        "--metadata-conflict",
        choices=["error", "first", "last"],
        default="error",
        help="How to handle conflicting __metadata__ values across shards",
    )
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024, help="Copy chunk size in bytes")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    parser.add_argument("--verify", action="store_true", help="Verify output structure after writing")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate shards, but don't write output")
    parser.add_argument("--no-disk-check", action="store_true", help="Skip free-space check (not recommended)")
    parser.add_argument(
        "--write-index",
        type=Path,
        default=None,
        help="Write a new *.safetensors.index.json pointing all weights to the combined file",
    )

    args = parser.parse_args(argv)

    root = Path.cwd()
    index_path: Optional[Path] = None
    if args.index is not None:
        index_path = args.index
    elif not args.no_index:
        index_path = find_index_file(root)

    if index_path is not None:
        shard_files = files_from_index(index_path)
    else:
        shard_files = find_shards_without_index(root, args.out)

    if not shard_files:
        print("No .safetensors files found.", file=sys.stderr)
        return 2

    combine_safetensors(
        shard_files=shard_files,
        out_path=args.out,
        overwrite=args.overwrite,
        metadata_conflict=args.metadata_conflict,
        chunk_size=max(1024 * 1024, int(args.chunk_size)),
        quiet=args.quiet,
        verify=args.verify,
        write_index=args.write_index,
        dry_run=args.dry_run,
        skip_disk_check=args.no_disk_check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
