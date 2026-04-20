"""
Standalone video dot-annotation tool.

Usage:
    python annotate_video.py <input_video> [-o OUTPUT]

Controls:
    Left click       - place a dot (or erase, if eraser mode is on)
    Left drag        - in eraser mode, sweep to erase
    Right click      - remove the nearest dot on the current frame
    e                - toggle eraser mode (cursor shows a circle)
    Space            - play / pause
    a / d            - step one frame back / forward
    c                - clear all dots on the current frame
    C (shift+c)      - clear ALL dots on every frame
    p                - toggle "persist": dot stays on every following frame
    s                - save the annotated video to OUTPUT
    q or Esc         - quit

Trackbars:
    Frame  - scrub through the video
    Size   - dot radius in pixels
    Hold   - how many frames a new dot stays visible (1 = current frame only)
    R/G/B  - dot color
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2


WINDOW = "annotate_video"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Path to the input video file")
    p.add_argument("-o", "--output", help="Output video path (default: <input>_annotated.mp4)")
    return p.parse_args()


def load_all_frames(cap, total):
    frames = []
    for _ in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def draw_dots(frame, dots):
    out = frame.copy()
    for d in dots:
        x, y, r, color = d[0], d[1], d[2], d[3]
        cv2.circle(out, (int(x), int(y)), int(r), color, -1, lineType=cv2.LINE_AA)
    return out


def visible_dots(idx, per_frame_dots, persist):
    """Return all dots that should be drawn on the given frame."""
    out = []
    for f, dots in per_frame_dots.items():
        if f > idx:
            continue
        for d in dots:
            hold = d[4]
            if persist or f + hold > idx:
                out.append(d)
    return out


def render_frame(frames, idx, per_frame_dots, persist):
    return draw_dots(frames[idx], visible_dots(idx, per_frame_dots, persist))


def nearest_visible_dot(per_frame_dots, idx, persist, x, y):
    """Return (frame, list_index) of the nearest visible dot, or None."""
    best, best_d2 = None, float("inf")
    for f, dots in per_frame_dots.items():
        if f > idx:
            continue
        for i, d in enumerate(dots):
            hold = d[4]
            if not (persist or f + hold > idx):
                continue
            d2 = (d[0] - x) ** 2 + (d[1] - y) ** 2
            if d2 < best_d2:
                best, best_d2 = (f, i), d2
    return best


def save_video(frames, per_frame_dots, persist, out_path, fps):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Failed to open writer for {out_path}", file=sys.stderr)
        return False
    for i in range(len(frames)):
        writer.write(render_frame(frames, i, per_frame_dots, persist))
    writer.release()
    print(f"Saved annotated video -> {out_path}")
    return True


def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or os.path.splitext(args.input)[0] + "_annotated.mp4"

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Could not open video: {args.input}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Loading {total} frames at {fps:.2f} fps ...")
    frames = load_all_frames(cap, total)
    cap.release()
    if not frames:
        print("No frames decoded.", file=sys.stderr)
        sys.exit(1)
    total = len(frames)
    print(f"Loaded {total} frames.")

    per_frame_dots = defaultdict(list)
    state = {
        "idx": 0,
        "playing": False,
        "persist": False,
        "syncing": False,
        "eraser": False,
        "cursor": None,
    }

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def on_frame_bar(v):
        if state["syncing"]:
            return
        state["idx"] = max(0, min(total - 1, v))

    def noop(_):
        pass

    cv2.createTrackbar("Frame", WINDOW, 0, max(total - 1, 1), on_frame_bar)
    cv2.createTrackbar("Size", WINDOW, 6, 100, noop)
    cv2.createTrackbar("Hold", WINDOW, 1, max(total, 1), noop)
    cv2.createTrackbar("R", WINDOW, 0, 255, noop)
    cv2.createTrackbar("G", WINDOW, 0, 255, noop)
    cv2.createTrackbar("B", WINDOW, 255, 255, noop)

    def current_color():
        b = cv2.getTrackbarPos("B", WINDOW)
        g = cv2.getTrackbarPos("G", WINDOW)
        r = cv2.getTrackbarPos("R", WINDOW)
        return (b, g, r)

    def current_size():
        return max(1, cv2.getTrackbarPos("Size", WINDOW))

    def current_hold():
        return max(1, cv2.getTrackbarPos("Hold", WINDOW))

    def erase_at(x, y):
        r = current_size()
        r2 = r * r
        idx = state["idx"]
        persist = state["persist"]
        for f in list(per_frame_dots.keys()):
            if f > idx:
                continue
            dots = per_frame_dots[f]
            kept = []
            for d in dots:
                hold = d[4]
                visible = persist or f + hold > idx
                if visible and (d[0] - x) ** 2 + (d[1] - y) ** 2 <= r2:
                    continue
                kept.append(d)
            per_frame_dots[f] = kept

    def on_mouse(event, x, y, flags, _):
        state["cursor"] = (x, y)
        if state["eraser"]:
            if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
                erase_at(x, y)
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                per_frame_dots[state["idx"]].append(
                    (x, y, current_size(), current_color(), current_hold())
                )
        if event == cv2.EVENT_RBUTTONDOWN:
            hit = nearest_visible_dot(per_frame_dots, state["idx"], state["persist"], x, y)
            if hit is not None:
                f, i = hit
                per_frame_dots[f].pop(i)

    cv2.setMouseCallback(WINDOW, on_mouse)

    delay_ms = max(1, int(1000.0 / fps))
    print("Ready. Press 'h' in terminal for controls or read the script header.")

    while True:
        idx = state["idx"]
        if cv2.getTrackbarPos("Frame", WINDOW) != idx:
            state["syncing"] = True
            cv2.setTrackbarPos("Frame", WINDOW, idx)
            state["syncing"] = False

        view = render_frame(frames, idx, per_frame_dots, state["persist"])
        if state["eraser"] and state["cursor"] is not None:
            cx, cy = state["cursor"]
            cv2.circle(view, (cx, cy), current_size(), (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(view, (cx, cy), current_size(), (255, 255, 255), 1, cv2.LINE_AA)
        hud = (
            f"frame {idx+1}/{total}  "
            f"{'PLAY' if state['playing'] else 'PAUSE'}  "
            f"persist={'on' if state['persist'] else 'off'}  "
            f"{'ERASER' if state['eraser'] else 'DRAW'}  "
            f"size={current_size()} hold={current_hold()} color={current_color()}"
        )
        cv2.putText(view, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, view)

        wait = delay_ms if state["playing"] else 20
        key = cv2.waitKey(wait) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            state["playing"] = not state["playing"]
        elif key == ord("a"):
            state["playing"] = False
            state["idx"] = max(0, idx - 1)
        elif key == ord("d"):
            state["playing"] = False
            state["idx"] = min(total - 1, idx + 1)
        elif key == ord("c"):
            per_frame_dots.pop(idx, None)
        elif key == ord("C"):
            per_frame_dots.clear()
        elif key == ord("p"):
            state["persist"] = not state["persist"]
        elif key == ord("e"):
            state["eraser"] = not state["eraser"]
        elif key == ord("s"):
            save_video(frames, per_frame_dots, state["persist"], out_path, fps)

        if state["playing"]:
            if idx + 1 >= total:
                state["playing"] = False
            else:
                state["idx"] = idx + 1

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
