# -*- coding: utf-8 -*-
"""Example: reading an .fmg slide with a multi-worker PyTorch DataLoader.

Run:  python test.py /path/to/slide.fmg
"""
import os
import sys

import cv2                      # noqa: F401  (kept for custom transforms)
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

import fmg


class SlideData(Dataset):
    def __init__(self, data_path, read_size=512, level=0, patch_cache=256):
        super().__init__()
        self.data_path = os.fspath(data_path)
        self.read_size = int(read_size)
        self.level = int(level)
        # NOTE: open_slide() only reads the JSON metadata.  LMDB environments
        # are opened on first use, in whichever process calls __getitem__.
        # That is what makes fork-based DataLoader workers safe.
        self.slide_image = fmg.open_slide(self.data_path, patch_cache=patch_cache)
        self.width, self.height = self.slide_image.dimensions  # (w, h)
        self.properties = self.slide_image.properties
        self.len_x = self.width // self.read_size
        self.len_y = self.height // self.read_size

    def __len__(self):
        return self.len_y * self.len_x

    def readroi(self, x, y, w, h, level=None):
        """Correct signature: location=(x, y) in level-0 frame, size=(w, h)."""
        level = self.level if level is None else level
        return self.slide_image.read_region((x, y), level, (w, h))

    def __getitem__(self, item):
        position_x = int(item % self.len_x) * self.read_size
        position_y = int(item // self.len_x) * self.read_size
        image = self.slide_image.read_region(
            (position_x, position_y), self.level,
            (self.read_size, self.read_size))
        # PIL -> HWC uint8 -> contiguous CHW tensor (zero copy after transpose).
        array = np.asarray(image, dtype=np.uint8)
        array = np.ascontiguousarray(array.transpose(2, 0, 1))
        position = np.array([position_x, position_y], dtype=np.int64)
        return torch.from_numpy(array), torch.from_numpy(position)


def worker_init_fn(worker_id):
    # OpenCV's thread pool does not survive fork(); disable it in the worker.
    # With the 'fork' context fmg does this automatically on pid change, and
    # with 'spawn' this explicit call is required.
    fmg.configure(num_threads=0)
    info = get_worker_info()
    if info is not None:
        np.random.seed((info.seed if info.seed is not None else torch.initial_seed()) % (2 ** 32))
        # Optional warm-up: open the env before the first batch so that all
        # workers do not hit the filesystem at the same moment.
        info.dataset.slide_image.read_region((0, 0), 0, (1, 1))


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else './demo.fmg'
    dataset = SlideData(path)
    print('properties:', dataset.properties)
    print('tiles:', len(dataset), 'dimensions (w,h):', dataset.slide_image.dimensions)

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,   # keep each worker's LMDB env across epochs
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,
        multiprocessing_context='fork',   # 'spawn' also works (slide is picklable)
    )
    for images, positions in dataloader:
        # images: [B, 3, H, W] uint8
        pass
    print('done')
