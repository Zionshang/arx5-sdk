#!/usr/bin/env python3
"""Renumber images in a directory.

This script scans the target directory for image files, shuffles their order
randomly, and renames them in-place to sequential names `Img0.ext`, `Img1.ext`, ...

Behavior:
- Supports common image extensions: .jpg, .jpeg, .png, .bmp, .tiff, .webp
- PRESERVES each file's original extension (e.g. Img0.png stays .png).
- Does NOT create backups or hidden directories; the script will perform
    a two-step rename using temporary random names inside the same directory
    and will not leave extra files when finished.

Usage:
    python renumber_images.py /path/to/jpg_images

If no path is provided, default is: /home/zishang/python-ws/yolo_data/jpg_images
"""

import os
import sys
import time
import random
import shutil
from typing import List, Optional


DEFAULT_DIR = "/home/zishang/python-ws/yolo_data/jpg_images"
try:
    # keep this list in sync with typical image extensions
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
except Exception:
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

from uuid import uuid4


def list_image_files(folder: str) -> List[str]:
    items = []
    for fn in os.listdir(folder):
        path = os.path.join(folder, fn)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMAGE_EXTS:
            items.append(fn)
    return items


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def renumber_folder(folder: str) -> None:
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        print(f"[Error] Not a directory: {folder}")
        return

    files = list_image_files(folder)
    if not files:
        print(f"[Info] No image files found in {folder}")
        return

    random.shuffle(files)
    print(f"[Info] Found {len(files)} images. Performing in-place shuffle+rename...")

    # Phase 1: rename each original file to a unique temporary name in the
    # same directory to avoid collisions with target names. Temporary names do
    # not start with a dot and will be removed/renamed in Phase 2.
    tmp_names = []
    for fn in files:
        src = os.path.join(folder, fn)
        ext = os.path.splitext(fn)[1]
        tmp_fn = f"tmp_{uuid4().hex}{ext}"
        tmp_path = os.path.join(folder, tmp_fn)
        try:
            os.rename(src, tmp_path)
            tmp_names.append((tmp_path, ext))
        except Exception as e:
            print(f"[Error] Failed to rename {src} -> {tmp_path}: {e}")
            # attempt to continue; user should inspect folder if errors occur

    # Phase 2: rename temporary files to final sequential names Img0.ext ..
    for i, (tmp_path, ext) in enumerate(tmp_names):
        final_name = f"Img{i}{ext}"
        final_path = os.path.join(folder, final_name)
        try:
            os.rename(tmp_path, final_path)
        except Exception as e:
            print(f"[Error] Failed to rename {tmp_path} -> {final_path}: {e}")

    print(f"[Done] Renumbered {len(tmp_names)} images in {folder}.")


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    folder = argv[0] if len(argv) > 0 else DEFAULT_DIR
    renumber_folder(folder)


if __name__ == '__main__':
    main()
