
import cv2
import os
import sys
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video, optionally resizing them."
    )
    parser.add_argument(
        "--video_path", type=str, required=True,
        help="Path to the input .mp4 video file."
    )
    parser.add_argument(
        "--resize", action="store_true",
        help="Flag to indicate if frames should be resized."
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Width to resize frames to (if --resize is used)."
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Height to resize frames to (if --resize is used)."
    )
    args = parser.parse_args()

    # subfolder ->  
    frame_dir=args.video_path
    print(frame_dir )
    tar_name = os.path.basename(frame_dir)
    base_dir = os.path.dirname(frame_dir)
    base_dir = base_dir.replace('_img', '_videos')

    os.makedirs(base_dir, exist_ok=True)

    output_video_path = os.path.join(base_dir, f'{tar_name}.mp4')
    print(output_video_path)
    #output_video_path = f'/data3/zihanwa3/_Robotics/_data/emdb_07_org_videos/org_{tar_name}.mp4'

    fps = 30

    # Gather all .jpg frames from the directory, sorted by filename
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))

    #if len(frame_paths) > 210:
    #  frame_paths = frame_paths[::2]

    # Ensure we actually have frames
    if not frame_paths:
        print(f"No frames found in {frame_dir}")
        exit(1)

    # Read the first frame to determine width and height
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', etc.
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loop over all frame paths, read each frame, and write to the video
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)

    # Release the VideoWriter
    out.release()

    print(f"Video successfully saved to: {output_video_path}")


if __name__ == "__main__":
    main()

