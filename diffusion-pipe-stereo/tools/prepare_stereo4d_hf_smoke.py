#!/usr/bin/env python3
"""Prepare a tiny paired Stereo4D-derived dataset for wan_stereo smoke tests."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-root", required=True, type=Path)
    parser.add_argument("--right-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--num-clips", type=int, default=10)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--frames", type=int, default=84)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument(
        "--caption",
        default="a stereoscopic VR180 scene with natural motion",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"Required executable not found on PATH: {name}")
    return path


def run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Command failed: " + " ".join(cmd)) from exc


def probe_video(ffprobe: str, path: Path) -> dict:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,nb_read_frames,avg_frame_rate,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found: {path}")
    stream = streams[0]
    frames = stream.get("nb_read_frames")
    try:
        frame_count = int(frames) if frames not in (None, "N/A") else None
    except ValueError:
        frame_count = None
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": frame_count,
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "duration": stream.get("duration"),
    }


def probe_duration(ffprobe: str, path: Path) -> float | None:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nk=1:nw=1",
        str(path),
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        return float(raw)
    except Exception:
        return None


def index_videos(root: Path, suffix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*.mp4")):
        stem = path.stem
        if not stem.endswith(suffix):
            continue
        key = stem[: -len(suffix)]
        out.setdefault(key, path)
    return out


def transcode(
    ffmpeg: str,
    src: Path,
    dst: Path,
    fps: int,
    frames: int,
    width: int,
    height: int,
    overwrite: bool,
) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        vf,
        "-frames:v",
        str(frames),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    run(cmd)


def main() -> int:
    args = parse_args()
    ffmpeg = require_executable("ffmpeg")
    ffprobe = require_executable("ffprobe")

    left_index = index_videos(args.left_root, "-left_rectified")
    right_index = index_videos(args.right_root, "-right_rectified")
    keys = sorted(set(left_index) & set(right_index))
    if not keys:
        raise SystemExit("No matching left/right rectified mp4 pairs found.")

    out_left = args.output_root / "left"
    out_right = args.output_root / "right"
    out_left.mkdir(parents=True, exist_ok=True)
    out_right.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_root / "manifest.jsonl"
    records: list[dict] = []
    skipped: list[dict] = []

    for key in keys:
        clip_id = len(records) + 1
        if clip_id > args.num_clips:
            break
        name = f"clip_{clip_id:06d}"
        left_dst = out_left / f"{name}.mp4"
        right_dst = out_right / f"{name}.mp4"
        caption_dst = out_right / f"{name}.txt"
        try:
            min_duration = args.frames / args.fps
            left_duration = probe_duration(ffprobe, left_index[key])
            right_duration = probe_duration(ffprobe, right_index[key])
            if left_duration is None or right_duration is None:
                raise RuntimeError(
                    f"could not probe source duration: left={left_duration} right={right_duration}"
                )
            if left_duration < min_duration or right_duration < min_duration:
                raise RuntimeError(
                    "source too short: "
                    f"left={left_duration:.3f}s right={right_duration:.3f}s "
                    f"min={min_duration:.3f}s"
                )
            transcode(
                ffmpeg,
                left_index[key],
                left_dst,
                args.fps,
                args.frames,
                args.width,
                args.height,
                args.overwrite,
            )
            transcode(
                ffmpeg,
                right_index[key],
                right_dst,
                args.fps,
                args.frames,
                args.width,
                args.height,
                args.overwrite,
            )
            left_probe = probe_video(ffprobe, left_dst)
            right_probe = probe_video(ffprobe, right_dst)
            for side, probe in [("left", left_probe), ("right", right_probe)]:
                if probe["width"] != args.width or probe["height"] != args.height:
                    raise RuntimeError(f"{side} output has wrong size: {probe}")
                if probe["frames"] is not None and probe["frames"] < args.frames:
                    raise RuntimeError(f"{side} output has too few frames: {probe}")
            caption_dst.write_text(args.caption + "\n", encoding="utf-8")
            record = {
                "clip": name,
                "source_key": key,
                "left_source": str(left_index[key]),
                "right_source": str(right_index[key]),
                "left_output": str(left_dst),
                "right_output": str(right_dst),
                "caption": args.caption,
                "target": {
                    "fps": args.fps,
                    "frames": args.frames,
                    "width": args.width,
                    "height": args.height,
                },
                "source_duration": {
                    "left": left_duration,
                    "right": right_duration,
                },
                "left_probe": left_probe,
                "right_probe": right_probe,
            }
            records.append(record)
            print(f"[ok] {name}: {key}")
        except Exception as exc:
            skipped.append({"source_key": key, "error": str(exc)})
            for path in [left_dst, right_dst, caption_dst]:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            print(f"[skip] {key}: {exc}", file=sys.stderr)

    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if skipped:
        with (args.output_root / "skipped.jsonl").open("w", encoding="utf-8") as f:
            for record in skipped:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Prepared {len(records)} clips at {args.output_root}")
    print(f"Manifest: {manifest_path}")
    if len(records) < args.num_clips:
        raise SystemExit(f"Only prepared {len(records)} / {args.num_clips} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
