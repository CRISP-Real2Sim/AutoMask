#!/usr/bin/env python3

import cv2
import os
import sys
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

    # Extract the base name (e.g. '449' from '449.mp4')
    tgt = os.path.splitext(os.path.basename(args.video_path))[0]

    # Prepare output directories
    output_dir = args.video_path.replace('videos', 'img').replace('.mp4', '')
    os.makedirs(output_dir, exist_ok=True)
    #output_dir_re = f"/data3/zihanwa3/_Robotics/_data/door_push/{tgt}_resize"
    # os.makedirs(output_dir_re, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(args.video_path)

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Decide how often to save frames; currently it saves *every* frame
        # Adjust "ind" if you only want to save every nth frame
        ind = 1
        if frame_count % ind == 0:
            frame_filename = f"{int(frame_count/ind):05d}.jpg"
            
            # Path for original-size frames
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            # If we are resizing, also save resized version
            if args.resize:
                resized_frame = cv2.resize(frame, (args.width, args.height))
                frame_path_re = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path_re, resized_frame)

            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Video: {args.video_path}")
    print(f"Total frames read: {frame_count}")
    print(f"Total frames saved: {saved_count}")

if __name__ == "__main__":
    main()