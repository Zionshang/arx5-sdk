#!/usr/bin/env python3
"""Simple RealSense D435i viewer with snapshot saving.

Press "s" to save the current RGB frame into /home/jyx/python_ws/yolo_data/jpg_images
as Img<i>.jpg (i starts at 0).
Press "q" to quit the viewer.
"""

import os
import cv2
import numpy as np
import pyrealsense2 as rs

OUTPUT_DIR = "/home/jyx/python_ws/yolo_data/jpg_images"
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
FPS = 30


def ensure_output_dir(path: str) -> None:
    """Create the output directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)

    try:
        pipeline.start(config)
        index = 0
        print('[Info] RealSense pipeline started. Press "s" to save, "q" to quit.')

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            cv2.imshow('RealSense D435i', color_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('s'):
                filename = os.path.join(OUTPUT_DIR, f"Img{index}.jpg")
                cv2.imwrite(filename, color_image)
                print(f"[Info] Saved frame to {filename}")
                index += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print('[Info] RealSense pipeline stopped.')


if __name__ == '__main__':
    main()
