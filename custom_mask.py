import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imageio
from sam2.build_sam import build_sam2_video_predictor
import argparse
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def save_video_from_frames(frame_dir, output_path, fps=30):
    """Save frames as a video at specified fps."""
    frame_names = [
        p for p in os.listdir(frame_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frame_dir, frame_names[0]))
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame_name in frame_names:
        frame_path = os.path.join(frame_dir, frame_name)
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()

def main(args):
    video_dir = args.video_dir
    seq = args.seq  
    text_prompt = args.text_prompt
    video_dir = video_dir.strip().rstrip('/')
    seq = seq.strip()

    # Construct the path using os.path.join()
    print(video_dir, seq)
    pathhhh = os.path.join(video_dir, seq, '00000.jpg')
    print(pathhhh, 'wtfnsafkajsfaksjf')

    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_id = "IDEA-Research/grounding-dino-tiny"
    model_cfg = "sam2_hiera_l.yaml"
    device = "cuda"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    #  base_dir = '/data3/zihanwa3/_Robotics/_data/toy_exp_im/449_resize'
    text_labels = [[text_prompt]]
    image = Image.open(pathhhh)
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    result = results[0]

    best_box = None
    best_score = -1
    best_label = None

    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        if score > best_score:
            best_score = score
            best_box = box
            best_label = label

    if best_box is not None:
        box = [round(x, 2) for x in best_box.tolist()]
        print(f"Best detection: {best_label} with confidence {round(best_score.item(), 3)} at location {best_box}")

        

    video_dir = f'{video_dir}/{seq}'

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    h, w = first_frame.shape[:2]
    print(f"Video dimensions: height={h}, width={w}")
    
    frame_idx = 0
    inference_state = predictor.init_state(video_path=video_dir)


    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    start_idx = int(frame_names[0].split('.')[0])
    end_idx = int(frame_names[-1].split('.')[0])
    save_dir = f'{args.save_dir}/{seq}'
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, text_prompt)
    os.makedirs(save_dir, exist_ok=True)
    out_video = os.path.join(save_dir, 'masks.mp4')
    #imageio.mimsave(out_video, frames)


    for i in range(end_idx-start_idx+1):
        dyn_mask=video_segments[i][4]
        np.savez_compressed(os.path.join(save_dir, f'dyn_mask_{i+start_idx}.npz'), dyn_mask=dyn_mask)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set the base directory for the experiment.")
    
    parser.add_argument('--video_dir', type=str, default='/data3/zihanwa3/_Robotics/_data/toy_exp_im', help="Path to the base directory")
    parser.add_argument('--save_dir', type=str, default='/data3/zihanwa3/_Robotics/_data/toy_exp_msk', help="Path to the base directory")
    parser.add_argument('--seq', type=str, required=True, help="Path to the base directory")
    parser.add_argument('--text_prompt', type=str, default='person', help="Path to the base directory")

    args = parser.parse_args()
    main(args)