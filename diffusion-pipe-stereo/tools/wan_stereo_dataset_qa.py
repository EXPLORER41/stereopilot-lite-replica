import argparse
import json
from pathlib import Path

import imageio.v3 as iio

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.avi'}


def media_files(path):
    return {
        item.name: item
        for item in sorted(Path(path).glob('*'))
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
    }


def video_info(path, expected_fps):
    meta = iio.immeta(path)
    first_frame = next(iio.imiter(path))
    height, width = first_frame.shape[:2]
    duration = float(meta.get('duration') or 0)
    fps = float(meta.get('fps') or expected_fps)
    frames = int(round(duration * fps)) if duration > 0 else None
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'duration': duration,
        'frames': frames,
    }


def close_enough(a, b, tolerance):
    if a is None or b is None:
        return True
    return abs(a - b) <= tolerance


def main():
    parser = argparse.ArgumentParser(description='QA paired left/right video folders for wan_stereo training.')
    parser.add_argument('--right', required=True, help='Right-eye target video directory.')
    parser.add_argument('--left', required=True, help='Left-eye conditioning video directory.')
    parser.add_argument('--fps', type=float, default=16.0, help='Expected fps used for frame-count estimates.')
    parser.add_argument('--min-frames', type=int, default=81)
    parser.add_argument('--min-width', type=int, default=512)
    parser.add_argument('--min-height', type=int, default=288)
    parser.add_argument('--frame-tolerance', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0, help='Only inspect the first N matched pairs; 0 means all.')
    parser.add_argument('--json', type=Path, default=None, help='Optional path to write a JSON report.')
    args = parser.parse_args()

    right_files = media_files(args.right)
    left_files = media_files(args.left)
    names = sorted(set(right_files) & set(left_files))
    if args.limit:
        names = names[:args.limit]

    report = {
        'right_dir': str(Path(args.right).resolve()),
        'left_dir': str(Path(args.left).resolve()),
        'right_count': len(right_files),
        'left_count': len(left_files),
        'matched_count': len(names),
        'missing_left': sorted(set(right_files) - set(left_files)),
        'missing_right': sorted(set(left_files) - set(right_files)),
        'errors': [],
        'warnings': [],
    }

    for name in names:
        right = right_files[name]
        left = left_files[name]
        caption = right.with_suffix('.txt')
        if not caption.exists():
            report['warnings'].append({'file': name, 'issue': 'missing_caption'})

        try:
            right_info = video_info(right, args.fps)
            left_info = video_info(left, args.fps)
        except Exception as exc:
            report['errors'].append({'file': name, 'issue': 'unreadable_video', 'detail': str(exc)})
            continue

        if right_info['width'] != left_info['width'] or right_info['height'] != left_info['height']:
            report['errors'].append({
                'file': name,
                'issue': 'resolution_mismatch',
                'right': [right_info['width'], right_info['height']],
                'left': [left_info['width'], left_info['height']],
            })

        if right_info['width'] < args.min_width or right_info['height'] < args.min_height:
            report['warnings'].append({
                'file': name,
                'issue': 'target_resolution_below_minimum',
                'right': [right_info['width'], right_info['height']],
            })

        if right_info['frames'] is not None and right_info['frames'] < args.min_frames:
            report['errors'].append({
                'file': name,
                'issue': 'right_too_short',
                'frames': right_info['frames'],
            })
        if left_info['frames'] is not None and left_info['frames'] < args.min_frames:
            report['errors'].append({
                'file': name,
                'issue': 'left_too_short',
                'frames': left_info['frames'],
            })

        if not close_enough(right_info['frames'], left_info['frames'], args.frame_tolerance):
            report['errors'].append({
                'file': name,
                'issue': 'frame_count_mismatch',
                'right_frames': right_info['frames'],
                'left_frames': left_info['frames'],
            })

    print(f"Matched pairs: {report['matched_count']}")
    print(f"Missing left videos: {len(report['missing_left'])}")
    print(f"Missing right videos: {len(report['missing_right'])}")
    print(f"Warnings: {len(report['warnings'])}")
    print(f"Errors: {len(report['errors'])}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Wrote JSON report: {args.json}')

    raise SystemExit(1 if report['errors'] else 0)


if __name__ == '__main__':
    main()
