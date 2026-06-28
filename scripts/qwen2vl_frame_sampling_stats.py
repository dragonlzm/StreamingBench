#!/usr/bin/env python3
"""Estimate Qwen2-VL video frame counts used by the StreamingBench evaluator.

The original StreamingBench Qwen2-VL wrapper first clips each source video to
`[time_start, timestamp]` and then passes the clip to qwen-vl-utils with a
dynamic `fps` value. This script mirrors that logic and reports summary stats
over the annotation JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        default="src/data/questions_real.json",
        help="StreamingBench annotation JSON. Defaults to the real split used by the Qwen2-VL run.",
    )
    parser.add_argument("--context-time", type=int, default=-1, help="Seconds of context before each query; <=0 uses 0.")
    parser.add_argument(
        "--mode",
        choices=("qwen-smart", "raw-fps"),
        default="qwen-smart",
        help="qwen-smart mirrors qwen-vl-utils smart_nframes; raw-fps only reports duration * selected fps.",
    )
    parser.add_argument("--frame-factor", type=int, default=2, help="qwen-vl-utils FRAME_FACTOR.")
    parser.add_argument("--min-frames", type=int, default=4, help="qwen-vl-utils FPS_MIN_FRAMES.")
    parser.add_argument("--max-frames", type=int, default=768, help="qwen-vl-utils FPS_MAX_FRAMES.")
    parser.add_argument(
        "--fallback-source-fps",
        type=float,
        default=30.0,
        help="Source fps used for total-frame bounding if ffprobe cannot read a video.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write detailed JSON stats.")
    parser.add_argument("--print-examples", type=int, default=0, help="Print the first N per-question rows.")
    return parser.parse_args()


def parse_timestamp(value: Any) -> int:
    parts = [int(float(part)) for part in str(value).strip().split(":")]
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def qwen2vl_selected_fps(duration_seconds: int) -> float:
    if duration_seconds > 300 and duration_seconds < 600:
        return 0.5
    if duration_seconds >= 600:
        return 0.2
    return 1.0


def floor_by_factor(value: float, factor: int) -> int:
    return int(math.floor(value / factor) * factor)


def ceil_by_factor(value: float, factor: int) -> int:
    return int(math.ceil(value / factor) * factor)


def parse_rate(rate: str) -> float | None:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_value = float(den)
        return float(num) / den_value if den_value else None
    return float(rate)


@lru_cache(maxsize=None)
def ffprobe_video(path: str) -> dict[str, float | int | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        streams = json.loads(result.stdout).get("streams", [])
        stream = streams[0] if streams else {}
    except Exception:
        return {"fps": None, "duration": None, "nb_frames": None}

    fps = parse_rate(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""))
    duration = float(stream["duration"]) if stream.get("duration") not in (None, "N/A") else None
    nb_frames = int(stream["nb_frames"]) if stream.get("nb_frames") not in (None, "N/A") else None
    return {"fps": fps, "duration": duration, "nb_frames": nb_frames}


def iter_entries(data: Any):
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            yield from iter_entries(item)


def estimate_total_clip_frames(video_path: Path, duration_seconds: int, fallback_source_fps: float) -> tuple[int, float]:
    meta = ffprobe_video(str(video_path))
    source_fps = float(meta["fps"] or fallback_source_fps)
    return max(1, int(round(duration_seconds * source_fps))), source_fps


def qwen_smart_nframes(
    duration_seconds: int,
    selected_fps: float,
    source_video_path: Path,
    args: argparse.Namespace,
) -> tuple[int, float, int]:
    total_frames, source_fps = estimate_total_clip_frames(
        source_video_path,
        duration_seconds,
        args.fallback_source_fps,
    )
    if args.mode == "raw-fps":
        return int(round(duration_seconds * selected_fps)), source_fps, total_frames

    min_frames = ceil_by_factor(args.min_frames, args.frame_factor)
    max_frames = floor_by_factor(min(args.max_frames, total_frames), args.frame_factor)
    raw = duration_seconds * selected_fps
    nframes = min(min(max(raw, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, args.frame_factor)
    return max(args.frame_factor, nframes), source_fps, total_frames


def summarize(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file).expanduser()
    data = json.loads(data_file.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for entry_idx, entry in enumerate(iter_entries(data)):
        questions = entry.get("questions")
        video_path_value = entry.get("video_path")
        if not isinstance(questions, list) or not video_path_value:
            continue
        video_path = Path(video_path_value).expanduser()
        if not video_path.is_absolute():
            video_path = (data_file.parent / video_path).resolve()
        for question_idx, question in enumerate(questions):
            timestamp = parse_timestamp(question["time_stamp"])
            start = max(0, timestamp - args.context_time) if args.context_time > 0 else 0
            duration = max(0, timestamp - start)
            selected_fps = qwen2vl_selected_fps(duration)
            sampled_frames, source_fps, total_clip_frames = qwen_smart_nframes(
                duration,
                selected_fps,
                video_path,
                args,
            )
            rows.append(
                {
                    "entry_index": entry_idx,
                    "question_index": question_idx,
                    "video_path": str(video_path),
                    "time_stamp": question["time_stamp"],
                    "duration_seconds": duration,
                    "selected_fps": selected_fps,
                    "estimated_source_fps": source_fps,
                    "estimated_total_clip_frames": total_clip_frames,
                    "sampled_frames": sampled_frames,
                }
            )

    if not rows:
        raise RuntimeError(f"No questions found in {data_file}")

    frame_values = [row["sampled_frames"] for row in rows]
    by_fps: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_fps[str(row["selected_fps"])].append(row["sampled_frames"])

    output = {
        "data_file": str(data_file),
        "context_time": args.context_time,
        "mode": args.mode,
        "qwen2vl_fps_rule": {
            "duration_seconds < 300": 1.0,
            "300 < duration_seconds < 600": 0.5,
            "duration_seconds >= 600": 0.2,
            "duration_seconds == 300": 1.0,
        },
        "frame_factor": args.frame_factor,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "overall": summarize(frame_values),
        "by_selected_fps": {fps: summarize(values) for fps, values in sorted(by_fps.items(), key=lambda item: float(item[0]))},
        "selected_fps_counts": dict(sorted(Counter(row["selected_fps"] for row in rows).items())),
    }

    print(json.dumps(output, indent=2))
    if args.print_examples:
        print("\nexamples:")
        for row in rows[: args.print_examples]:
            print(json.dumps(row, ensure_ascii=False))
    if args.output_json:
        out = Path(args.output_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**output, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
