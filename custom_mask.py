
from __future__ import annotations

import argparse
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2_video_predictor


def _select_best_box(result: dict[int, np.ndarray], scores, labels) -> tuple[np.ndarray, float, str]:
    """Select the highest-scoring detection and normalise outputs to Python types."""

    if isinstance(scores, torch.Tensor):
        score_tensor = scores.detach().cpu()
        best_idx = int(score_tensor.argmax().item())
        best_score = float(score_tensor[best_idx].item())
    else:
        score_array = np.asarray(scores)
        best_idx = int(score_array.argmax())
        best_score = float(score_array[best_idx])

    boxes = result["boxes"]
    if isinstance(boxes, torch.Tensor):
        box_values = boxes.detach().cpu().numpy()
    else:
        box_values = np.asarray(boxes)

    label_values = labels
    if isinstance(labels, torch.Tensor):
        label_values = labels.detach().cpu().tolist()
    elif isinstance(labels, np.ndarray):
        label_values = labels.tolist()

    return box_values[best_idx].tolist(), best_score, label_values[best_idx]


def _prepare_paths(video_root: Path, seq: str, save_root: Path) -> tuple[Path, Path]:
    seq_dir = video_root / seq
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence '{seq}' not found in {video_root}")
    save_dir = save_root / seq
    save_dir.mkdir(parents=True, exist_ok=True)
    return seq_dir, save_dir


def _discover_frames(seq_dir: Path) -> list[Path]:
    frames = sorted([p for p in seq_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'}], key=lambda p: int(p.stem))
    if not frames:
        raise FileNotFoundError(f"No JPEG frames found in {seq_dir}")
    return frames


def _write_masks(video_segments: dict[int, dict[int, np.ndarray]], frame_paths: list[Path], save_dir: Path, obj_id: int, overwrite: bool) -> list[Path]:
    written = []
    for frame_idx in sorted(video_segments):
        obj_map = video_segments[frame_idx]
        if obj_id not in obj_map:
            continue
        if frame_idx >= len(frame_paths):
            continue
        mask = obj_map[obj_id]
        frame_path = frame_paths[frame_idx]
        out_path = save_dir / f"dyn_mask_{frame_path.stem}.npz"
        if out_path.exists() and not overwrite:
            continue
        np.savez_compressed(out_path, dyn_mask=mask[np.newaxis, ...].astype(np.uint8))
        written.append(out_path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate SAM2 masks for a sequence using a text prompt.")
    parser.add_argument('--video_dir', type=Path, required=True, help='Directory containing per-sequence image folders')
    parser.add_argument('--save_dir', type=Path, default=None, help='Output directory root for masks (defaults to video_dir/../sam_v2_dyn_mask)')
    parser.add_argument('--seq', type=str, required=True, help='Sequence name (subdirectory of video_dir)')
    parser.add_argument('--text_prompt', type=str, default='person', help='Text prompt for GroundingDINO')
    parser.add_argument('--box_threshold', type=float, default=0.4, help='GroundingDINO box confidence threshold')
    parser.add_argument('--text_threshold', type=float, default=0.3, help='GroundingDINO text confidence threshold')
    parser.add_argument('--model_id', type=str, default='IDEA-Research/grounding-dino-tiny', help='GroundingDINO model identifier')
    parser.add_argument('--sam_checkpoint', type=Path, default=Path('AutoMask/checkpoints/sam2_hiera_large.pt'), help='Path to SAM2 checkpoint')
    parser.add_argument('--sam_config', type=str, default='sam2_hiera_l.yaml', help='SAM2 config file name')
    parser.add_argument('--device', type=str, default='cuda', help='Torch device to run on')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing mask files')
    args = parser.parse_args(argv)

    video_root = args.video_dir.resolve()
    save_root = args.save_dir if args.save_dir is not None else (video_root.parent / 'sam_v2_dyn_mask')
    save_root = save_root.resolve()

    seq_dir, seq_save_dir = _prepare_paths(video_root, args.seq, save_root)
    frame_paths = _discover_frames(seq_dir)

    if not args.overwrite:
        existing = list(seq_save_dir.glob('dyn_mask_*.npz'))
        if existing:
            print(f"[AutoMask] Found {len(existing)} mask files in {seq_save_dir}, skipping generation.")
            return

    print(frame_paths[0], 'frame_paths[0')
    first_frame = Image.open(frame_paths[0]).convert('RGB')

    processor = AutoProcessor.from_pretrained(args.model_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(args.device)

    inputs = processor(images=first_frame, text=[args.text_prompt], return_tensors='pt').to(args.device)
    with torch.no_grad():
        outputs = detector(**inputs)
    detections = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[first_frame.size[::-1]],
    )[0]

    if len(detections['boxes']) == 0:
        raise RuntimeError(f"No detections found for prompt '{args.text_prompt}' in {frame_paths[0]}")

    label_source = detections.get('text_labels', detections['labels'])
    box, score, label = _select_best_box(detections, detections['scores'], label_source)
    print(f"[AutoMask] Selected box {box} for label '{label}' (score={score:.3f})")

    predictor = build_sam2_video_predictor(
        config_file=args.sam_config,
        ckpt_path=str(args.sam_checkpoint),
        device=args.device,
    )

    inference_state = predictor.init_state(video_path=str(seq_dir))
    ann_obj_id = 1

    with torch.inference_mode(), torch.autocast(device_type='cuda' if args.device.startswith('cuda') else 'cpu', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=ann_obj_id,
            box=box,
        )
        video_segments: dict[int, dict[int, np.ndarray]] = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[idx] > 0.0).cpu().numpy()
                for idx, out_obj_id in enumerate(out_obj_ids)
            }

    written = _write_masks(video_segments, frame_paths, seq_save_dir, ann_obj_id, args.overwrite)
    if not written:
        print(f"[AutoMask] No masks written for {args.seq}; consider using --overwrite.")
        return

    metadata = {
        'sequence': args.seq,
        'prompt': args.text_prompt,
        'box': box,
        'score': score,
        'label': label,
        'mask_count': len(written),
    }
    with (seq_save_dir / 'metadata.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"[AutoMask] Saved {len(written)} masks to {seq_save_dir}")


if __name__ == '__main__':
    main()
