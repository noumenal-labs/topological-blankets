"""Split a demo GIF into 3 phase-based segments for presentation.

Usage:
    python split_demo_gif.py <input.gif> [--fps 3] [--out-dir ./output]

Splits the demo into 3 GIFs:
  1. Normal autonomous behavior (frames before perturbation)
  2. Uncertainty spike + human intervention (perturbation through release)
  3. Return to autonomous completion (release through end)

The split points are determined by frame index ratios matching the
Wednesday demo phase boundaries. Override with --split1 and --split2
to set exact frame indices.

Aesthetic improvements applied:
  - Configurable FPS (default 3, slower than raw recording)
  - Phase label overlay on each frame
  - Color-coded border per phase (green/red/green)
"""

import argparse
import sys
from pathlib import Path

try:
    import imageio.v3 as iio
except ImportError:
    try:
        import imageio
        iio = None
    except ImportError:
        print("ERROR: imageio is required. Install with: pip install imageio[pillow]")
        sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


PHASE_COLORS = {
    1: (46, 204, 113),    # green - autonomous
    2: (231, 76, 60),     # red - uncertainty/intervention
    3: (46, 204, 113),    # green - resumed autonomous
}

PHASE_LABELS = {
    1: "Autonomous Behavior",
    2: "Uncertainty Spike + Human Intervention",
    3: "Autonomous Completion",
}


def add_overlay(frame, phase: int, step: int, total: int) -> "Image.Image":
    """Add phase label and color-coded border to a frame."""
    if not HAS_PIL:
        return frame

    if isinstance(frame, Image.Image):
        img = frame.copy()
    else:
        img = Image.fromarray(frame)

    color = PHASE_COLORS[phase]
    label = PHASE_LABELS[phase]
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Color border (4px)
    for i in range(4):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)

    # Phase label background
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    text = f"{label}  (frame {step + 1}/{total})"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([8, h - th - 16, tw + 16, h - 8], fill=(0, 0, 0, 180))
    draw.text((12, h - th - 12), text, fill=color, font=font)

    return img


def load_gif(path: str):
    """Load all frames from a GIF."""
    if iio is not None:
        frames = iio.imread(path, plugin="pillow", mode="RGBA")
        return [Image.fromarray(f) for f in frames]
    else:
        reader = imageio.get_reader(path)
        return [Image.fromarray(f) for f in reader]


def save_gif(frames, path: str, fps: int):
    """Save frames as a GIF."""
    duration_ms = 1000.0 / fps
    if iio is not None:
        import numpy as np
        arrays = [np.array(f) for f in frames]
        iio.imwrite(path, arrays, plugin="pillow",
                     duration=duration_ms, loop=0)
    else:
        imageio.mimsave(path, [f if not isinstance(f, Image.Image)
                               else f.convert("RGBA")
                               for f in frames],
                        duration=duration_ms / 1000.0, loop=0)


def main():
    parser = argparse.ArgumentParser(
        description="Split a demo GIF into 3 phase segments")
    parser.add_argument("input", help="Path to input GIF")
    parser.add_argument("--fps", type=int, default=3,
                        help="Output FPS (default: 3, slower for readability)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--split1", type=float, default=0.25,
                        help="Fraction for split between phase 1 and 2 "
                             "(default: 0.25, i.e. perturbation at 25%%)")
    parser.add_argument("--split2", type=float, default=0.65,
                        help="Fraction for split between phase 2 and 3 "
                             "(default: 0.65, i.e. release at 65%%)")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip phase label overlays")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    print(f"Loading {input_path}...")
    frames = load_gif(str(input_path))
    n = len(frames)
    print(f"  {n} frames loaded")

    # Compute split points
    s1 = int(n * args.split1)
    s2 = int(n * args.split2)
    print(f"  Split points: phase1=[0:{s1}], phase2=[{s1}:{s2}], phase3=[{s2}:{n}]")

    segments = {
        1: frames[:s1],
        2: frames[s1:s2],
        3: frames[s2:],
    }

    for phase, seg_frames in segments.items():
        if not seg_frames:
            print(f"  Phase {phase}: no frames, skipping")
            continue

        if not args.no_overlay and HAS_PIL:
            seg_frames = [add_overlay(f, phase, i, len(seg_frames))
                          for i, f in enumerate(seg_frames)]

        out_path = out_dir / f"{stem}_phase{phase}_{PHASE_LABELS[phase].lower().replace(' ', '_')}.gif"
        save_gif(seg_frames, str(out_path), args.fps)
        duration_sec = len(seg_frames) / args.fps
        print(f"  Phase {phase}: {len(seg_frames)} frames, {duration_sec:.1f}s @ {args.fps}fps -> {out_path.name}")

    print("\nDone. Suggested improvements for each segment:")
    print("  Phase 1: Should show confident, smooth gripper motion toward object")
    print("  Phase 2: Ensemble disagreement bands should visibly widen; "
          "consider adding a vertical red line at perturbation frame")
    print("  Phase 3: Bands narrow again as agent resumes; "
          "consider a subtle green pulse on the border at release frame")


if __name__ == "__main__":
    main()
