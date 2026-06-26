#!/usr/bin/env python3
"""Prepare StreamingBench/Qwen2-VL assets for offline Platformr evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


OFFLINE_ANNOTATIONS = (
    "questions_real.json",
    "questions_omni.json",
    "questions_sqa.json",
    "questions_proactive.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage StreamingBench annotations/videos and Qwen2-VL weights for offline Platformr evaluation."
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", "/mnt/scratch/group/li968/zliu2346/huggingface"),
        help="Asset root used on Platformr. Defaults to HF_HOME or the group scratch HF_HOME.",
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="StreamingBench repo root.",
    )
    parser.add_argument("--dataset-id", default="mjuicem/StreamingBench", help="Hugging Face dataset id.")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct", help="Hugging Face model id.")
    parser.add_argument(
        "--streamingbench-root",
        type=Path,
        default=None,
        help="Output root for StreamingBench assets. Defaults to HF_HOME/streamingbench.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Output directory for model weights. Defaults to HF_HOME/models/Qwen2-VL-7B-Instruct.",
    )
    parser.add_argument(
        "--download-work-dir",
        type=Path,
        default=None,
        help=(
            "Local filesystem directory used for Hugging Face downloads before copying to HF_HOME. "
            "Use this when HF_HOME is on SMB/GVFS/NFS and file locks are unsupported. "
            "Defaults to TMPDIR/streamingbench_platformr_downloads."
        ),
    )
    parser.add_argument("--skip-dataset-download", action="store_true", help="Do not download the dataset snapshot.")
    parser.add_argument("--skip-model-download", action="store_true", help="Do not download the Qwen2-VL snapshot.")
    parser.add_argument("--skip-video-prepare", action="store_true", help="Do not extract/copy videos.")
    parser.add_argument("--strict", action="store_true", help="Fail if any annotation video is missing after preparation.")
    return parser.parse_args()


def copy_directory_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def safe_repo_dir_name(repo_id: str, repo_type: str) -> str:
    return f"{repo_type}--{repo_id.replace('/', '--')}"


def snapshot_download(repo_id: str, repo_type: str, local_dir: Path, download_work_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Install it before preparing offline assets.") from exc

    download_dir = download_work_dir / safe_repo_dir_name(repo_id, repo_type)
    download_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        hf_snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(download_dir),
        )
    )
    if snapshot_path.resolve() != local_dir.resolve():
        print(f"Copying completed download {snapshot_path} -> {local_dir}")
        copy_directory_contents(snapshot_path, local_dir)
    return local_dir


def extract_archives(raw_root: Path, extracted_root: Path) -> int:
    extracted_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for archive in sorted(raw_root.rglob("*.zip")):
        target = extracted_root / archive.stem
        marker = target / ".extracted"
        if marker.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive} -> {target}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        marker.touch()
        count += 1
    return count


def iter_objects(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_objects(child)


def expected_video_names(annotation_root: Path) -> set[str]:
    names: set[str] = set()
    for filename in OFFLINE_ANNOTATIONS:
        with (annotation_root / filename).open("r", encoding="utf-8") as f:
            data = json.load(f)
        for obj in iter_objects(data):
            video_path = obj.get("video_path")
            if isinstance(video_path, str):
                names.add(Path(video_path).name)
    return names


def streamingbench_video_name(video_file: Path, expected_names: set[str]) -> str:
    if video_file.name in expected_names:
        return video_file.name
    if len(video_file.parents) >= 2:
        sample_folder = video_file.parent.name
        task_type = video_file.parent.parent.name.replace(" ", "_")
        generated = f"{sample_folder}_{task_type}.mp4"
        if generated in expected_names:
            return generated
        return generated
    return video_file.name


def copy_videos(search_roots: list[Path], video_root: Path, expected_names: set[str]) -> int:
    video_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    seen_sources: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for src in sorted(root.rglob("*.mp4")):
            if "__MACOSX" in src.parts:
                continue
            if src in seen_sources:
                continue
            seen_sources.add(src)
            dest = video_root / streamingbench_video_name(src, expected_names)
            if dest.exists() and dest.stat().st_size == src.stat().st_size:
                continue
            shutil.copy2(src, dest)
            copied += 1
    return copied


def write_platformr_annotations(source_root: Path, annotation_root: Path, video_root: Path) -> set[str]:
    annotation_root.mkdir(parents=True, exist_ok=True)
    missing: set[str] = set()
    for filename in OFFLINE_ANNOTATIONS:
        src = source_root / filename
        dst = annotation_root / filename
        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for obj in iter_objects(data):
            video_path = obj.get("video_path")
            if not isinstance(video_path, str):
                continue
            resolved = video_root / Path(video_path).name
            obj["video_path"] = str(resolved)
            if not resolved.exists():
                missing.add(str(resolved))
        with dst.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Wrote {dst}")
    return missing


def main() -> None:
    args = parse_args()
    hf_home = Path(args.hf_home).expanduser().resolve()
    streamingbench_root = args.streamingbench_root or hf_home / "streamingbench"
    model_dir = args.model_dir or hf_home / "models" / "Qwen2-VL-7B-Instruct"
    download_work_dir = args.download_work_dir or Path(tempfile.gettempdir()) / "streamingbench_platformr_downloads"
    download_work_dir = download_work_dir.expanduser().resolve()
    source_annotation_root = args.repo_root / "src" / "data"

    raw_root = streamingbench_root / "raw"
    extracted_root = streamingbench_root / "extracted"
    video_root = streamingbench_root / "videos"
    annotation_root = streamingbench_root / "annotations"

    expected_names = expected_video_names(source_annotation_root)

    if not args.skip_dataset_download:
        print(f"Downloading dataset {args.dataset_id} to {raw_root}")
        print(f"Using local download work dir: {download_work_dir}")
        snapshot_download(args.dataset_id, "dataset", raw_root, download_work_dir)

    if not args.skip_video_prepare:
        archive_count = extract_archives(raw_root, extracted_root)
        copied_count = copy_videos([raw_root, extracted_root], video_root, expected_names)
        print(f"Extracted archives: {archive_count}")
        print(f"Copied/updated videos: {copied_count}")

    missing = write_platformr_annotations(source_annotation_root, annotation_root, video_root)

    if not args.skip_model_download:
        print(f"Downloading model {args.model_id} to {model_dir}")
        print(f"Using local download work dir: {download_work_dir}")
        snapshot_download(args.model_id, "model", model_dir, download_work_dir)

    print("\nPrepared offline inputs:")
    print(f"  annotations: {annotation_root}")
    print(f"  videos:      {video_root}")
    print(f"  model:       {model_dir}")
    print(f"  work dir:    {download_work_dir}")
    if missing:
        print(f"  missing videos referenced by annotations: {len(missing)}")
        for path in sorted(missing)[:20]:
            print(f"    {path}")
        if len(missing) > 20:
            print("    ...")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
