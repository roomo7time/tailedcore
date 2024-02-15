import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torchvision import transforms
import time
import os
from PIL import Image
from cachetools import LRUCache
import json

from src.datasets.labelme import IterablePatchDataset

# class ExhaustivePatchesIterableDataset(IterableDataset):
#     def __init__(self, image_paths, patch_size):
#         self.image_paths = image_paths
#         self.patch_size = patch_size

#     def process_image(self, image_path):
#         with Image.open(image_path) as img:
#             for x in range(0, img.width - self.patch_size + 1, self.patch_size):
#                 for y in range(0, img.height - self.patch_size + 1, self.patch_size):
#                     patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
#                     yield transforms.ToTensor()(patch)

#     def __iter__(self):
#         for image_path in self.image_paths:
#             yield from self.process_image(image_path)


class ExhaustivePatchesDataset(Dataset):

    def __init__(self, image_paths, patch_size, cache_size=6):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.image_cache = LRUCache(maxsize=cache_size)
        self.indices = self._calculate_indices()

    def _calculate_indices(self):
        indices = []
        for image_path in self.image_paths:
            with Image.open(image_path) as img:
                for x in range(0, img.width - self.patch_size + 1, self.patch_size):
                    for y in range(
                        0, img.height - self.patch_size + 1, self.patch_size
                    ):
                        indices.append((image_path, x, y))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_path, x, y = self.indices[idx]
        if image_path not in self.image_cache:
            image = Image.open(image_path)
            self.image_cache[image_path] = image
        else:
            image = self.image_cache[image_path]

        patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
        return transforms.ToTensor()(patch)


# Assuming ExhaustivePatchesIterableDataset is defined as before
# Modify it if necessary to suit this test
def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.image_paths) // worker_info.num_workers

    dataset.image_paths = dataset.image_paths[
        worker_id * split_size : (worker_id + 1) * split_size
    ]


def test_dataloader(
    prefetch_factor,
    num_workers=4,
    batch_size=4,
    iterable=True,
    pin_memory=False,
    use_worker_init_fn=False,
    shuffle=False,
):

    if iterable:
        dataset = IterablePatchDataset("./dummy_images", patch_size=256)
    else:
        raise NotImplementedError()

    if use_worker_init_fn:
        _worker_init_fn = worker_init_fn
    else:
        _worker_init_fn = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init_fn,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )

    start_time = time.time()
    for _ in dataloader:
        pass  # Iterate through all batches
    end_time = time.time()

    print(
        f"prefetch_factor={prefetch_factor}, num_workers={num_workers}, iterable={iterable}, pin_memory={pin_memory}, use_worker_init_fn={use_worker_init_fn}: Time taken = {end_time - start_time:.2f} seconds"
    )


# Test with different prefetch_factors

# test_dataloader(num_workers=4, iterable=True, use_worker_init_fn=True, prefetch_factor=2)
# test_dataloader(num_workers=4, iterable=True, use_worker_init_fn=True, prefetch_factor=8)
# test_dataloader(num_workers=4, iterable=True, use_worker_init_fn=True, prefetch_factor=32)
# test_dataloader(num_workers=4, iterable=True, use_worker_init_fn=True, prefetch_factor=1024)

# test_dataloader(num_workers=4, iterable=True, use_worker_init_fn=False, prefetch_factor=2)
# test_dataloader(num_workers=4, iterable=True, use_worker_init_fn=False, prefetch_factor=8)
# test_dataloader(num_workers=4, iterable=True, use_worker_init_fn=False, prefetch_factor=32)
# test_dataloader(num_workers=4, iterable=True, use_worker_init_fn=False, prefetch_factor=1024)

# test_dataloader(num_workers=4, iterable=False, use_worker_init_fn=True, prefetch_factor=2)
# test_dataloader(num_workers=4, iterable=False, use_worker_init_fn=True, prefetch_factor=8)
# test_dataloader(num_workers=4, iterable=False, use_worker_init_fn=True, prefetch_factor=32)
# test_dataloader(num_workers=4, iterable=False, use_worker_init_fn=True, prefetch_factor=1024)

# test_dataloader(num_workers=4, iterable=False, use_worker_init_fn=False, prefetch_factor=2)
# test_dataloader(num_workers=4, iterable=False, use_worker_init_fn=False, prefetch_factor=8)
# test_dataloader(num_workers=4, iterable=False, use_worker_init_fn=False, prefetch_factor=32)
# test_dataloader(num_workers=4, iterable=False, use_worker_init_fn=False, prefetch_factor=1024)

test_dataloader(num_workers=0, iterable=True, prefetch_factor=2)

# test_dataloader(num_workers=0, iterable=False,  prefetch_factor=2)

# test_dataloader(num_workers=0, iterable=False,  prefetch_factor=2, shuffle=True)
# test_dataloader(num_workers=4, iterable=False,  prefetch_factor=2, shuffle=True)
