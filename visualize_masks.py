from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _resolve_sequence_paths(video_root: Path, mask_root: Path, seq: str) -> tuple[Path, Path]:
    img_seq_dir = video_root / seq
    if not img_seq_dir.exists():
        raise FileNotFoundError(f"Image sequence '{seq}' not found in {video_root}")
    mask_seq_dir = mask_root / seq
    if not mask_seq_dir.exists():
        raise FileNotFoundError(f"Mask sequence '{seq}' not found in {mask_root}")
    return img_seq_dir, mask_seq_dir


def _discover_mask_files(mask_seq_dir: Path) -> list[Path]:
    mask_files = sorted(mask_seq_dir.glob("dyn_mask_*.npz"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_seq_dir}")
    return mask_files


def _match_image_path(img_seq_dir: Path, frame_token: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = img_seq_dir / f"{frame_token}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Frame '{frame_token}' not found in {img_seq_dir}")


def _load_mask(mask_path: Path) -> np.ndarray:
    with np.load(mask_path) as data:
        mask = data["dyn_mask"]
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(
            f"Mask in {mask_path} has unexpected shape {mask.shape} after squeeze"
        )
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return mask


def _blend_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    mask_bool = mask > 0
    if not mask_bool.any():
        return image
    overlay = image.copy()
    color_arr = np.array(color, dtype=np.float32)
    blended_pixels = (
        (1.0 - alpha) * overlay[mask_bool].astype(np.float32) + alpha * color_arr
    ).clip(0, 255)
    overlay[mask_bool] = blended_pixels.astype(np.uint8)
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(255, 255, 255), thickness=1)
    return overlay


def _ensure_vis_dirs(mask_seq_dir: Path) -> tuple[Path, Path]:
    vis_dir = mask_seq_dir / "vis"
    imgs_dir = vis_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    return vis_dir, imgs_dir


def _init_writer(video_path: Path, frame_size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {video_path}")
    return writer


def render_sequence(
    video_root: Path,
    mask_root: Path,
    seq: str,
    overwrite: bool,
    fps: float,
    alpha: float,
) -> None:
    img_seq_dir, mask_seq_dir = _resolve_sequence_paths(video_root, mask_root, seq)
    vis_dir, imgs_dir = _ensure_vis_dirs(mask_seq_dir)
    video_path = vis_dir / "vis.mp4"

    existing_imgs = list(imgs_dir.glob("*.jpg"))
    if video_path.exists() and existing_imgs and not overwrite:
        print(f"[AutoMask][Vis] Found existing visualisation in {vis_dir}, skipping.")
        return

    if overwrite:
        for old_img in existing_imgs:
            old_img.unlink()
        if video_path.exists():
            video_path.unlink()

    mask_files = _discover_mask_files(mask_seq_dir)

    writer: cv2.VideoWriter | None = None
    frame_size: tuple[int, int] | None = None

    for mask_path in mask_files:
        frame_token = mask_path.stem.replace("dyn_mask_", "")
        frame_path = _match_image_path(img_seq_dir, frame_token)
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image {frame_path}")
        mask = _load_mask(mask_path)
        if mask.size == 0:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        elif mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        frame = _blend_overlay(image, mask, color=(0, 0, 255), alpha=alpha)

        out_path = imgs_dir / f"{frame_token}.jpg"
        if not cv2.imwrite(str(out_path), frame):
            raise RuntimeError(f"Failed to write visualisation frame {out_path}")

        if writer is None:
            frame_size = (frame.shape[1], frame.shape[0])
            writer = _init_writer(video_path, frame_size, fps)
        assert writer is not None
        writer.write(frame)

    if writer is not None:
        writer.release()
    print(f"[AutoMask][Vis] Saved frames to {imgs_dir} and video to {video_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render overlay visualisations for dynamic masks")
    parser.add_argument("--video_dir", type=Path, required=True, help="Root directory containing image sequences")
    parser.add_argument("--mask_dir", type=Path, required=True, help="Root directory containing mask sequences")
    parser.add_argument("--seq", type=str, required=True, help="Sequence name")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second for the output video")
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay opacity (0-1)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate visualisations even if they exist")
    args = parser.parse_args(argv)

    render_sequence(
        video_root=args.video_dir.resolve(),
        mask_root=args.mask_dir.resolve(),
        seq=args.seq,
        overwrite=args.overwrite,
        fps=args.fps,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
