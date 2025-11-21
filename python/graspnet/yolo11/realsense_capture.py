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
import re

OUTPUT_DIR = "/home/zishang/python-ws/yolo_data/jpg_images"
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

    def _next_index_from_dir(path: str) -> int:
        """Scan output directory for existing Img<N>.jpg files and return next index."""
        try:
            files = os.listdir(path)
        except Exception:
            return 0
        pattern = re.compile(r'^Img(\d+)\.jpg$', flags=re.IGNORECASE)
        nums = []
        for fn in files:
            m = pattern.match(fn)
            if m:
                try:
                    nums.append(int(m.group(1)))
                except Exception:
                    continue
        if not nums:
            return 0
        return max(nums) + 1

    try:
        pipeline.start(config)
        index = _next_index_from_dir(OUTPUT_DIR)
        print(f'[Info] RealSense pipeline started. Press "s" to save, "q" to quit. Next index: {index}')

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
