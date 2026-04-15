#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def resize(input_path: Path, output_path: Path, width: int = 1280, height: int = 720) -> None:
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "copy",
        "-y",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize 1920x1080 video to 1280x720 (or custom size).")
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("output", type=Path, nargs="?", help="Output video file (default: <input>_720p.<ext>)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input.with_name(f"{args.input.stem}_720p{args.input.suffix}")
    resize(args.input, output, args.width, args.height)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
