import torch
import os
import json
import cv2
import numpy as np
import collections
from natsort import natsorted

from PIL import Image
from shapely.geometry import Polygon
from torch._six import string_classes
from typing import Dict, List
from torch.utils.data import IterableDataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class IterablePatchDataset(IterableDataset):

    def __init__(
        self,
        image_dir_path,
        patch_size,
        input_size,
        overlap_ratio=0.0,
        exclude_blob_area=False,
        roi=None,
    ):

        if overlap_ratio is None:
            overlap_ratio = 0.0

        self.image_dir_path = image_dir_path
        self.patch_size = patch_size
        self.input_size = input_size
        self.overlap_ratio = overlap_ratio
        self.exclude_blob_area = exclude_blob_area
        self.image_paths = self.find_image_paths()

        self.patch_dict, self.is_anomaly, self.image_sizes, self.blob_areas_dict = (
            self._get_data_info(roi)
        )

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        if input_size != patch_size:
            transform_list = [transforms.Resize(input_size)] + transform_list

        self.transform = transforms.Compose(transform_list)

    def find_image_paths(self):
        return self._find_image_paths(self.image_dir_path)

    def _find_image_paths(self, image_dir_path):
        image_paths = []
        for file in os.listdir(image_dir_path):
            if file.endswith((".bmp", "jpg", "jpeg")):
                image_paths.append(os.path.join(image_dir_path, file))
        return natsorted(image_paths)

    def _get_data_info(self, roi):
        is_anomaly = {}
        patch_dict = {}
        image_sizes = {}
        blob_areas_dict = {}

        for image_path in self.image_paths:
            blob_areas = self._get_blob_areas(image_path)
            blob_areas_dict[image_path] = blob_areas

            with Image.open(image_path) as img:
                patches = self._generate_patches(img, roi, blob_areas)
                patch_dict[image_path] = patches

            is_anomaly[image_path] = 0 if len(blob_areas) == 0 else 1
            image_sizes[image_path] = (img.height, img.width)

        return patch_dict, is_anomaly, image_sizes, blob_areas_dict

    def _generate_patches(self, img, roi, blob_areas):
        patches = []
        stride_x = int(self.patch_size * (1 - self.overlap_ratio))
        stride_y = int(self.patch_size * (1 - self.overlap_ratio))

        # Define ROI boundaries
        roi_x0, roi_y0, roi_x1, roi_y1 = roi if roi else (0, 0, img.width, img.height)

        assert roi_x0 >= 0
        assert roi_y0 >= 0
        assert roi_x1 <= img.width
        assert roi_y1 <= img.height

        for x in range(roi_x0, roi_x1, stride_x):
            for y in range(roi_y0, roi_y1, stride_y):

                x0 = x
                y0 = y
                x1 = x + self.patch_size
                y1 = y + self.patch_size

                if x1 > roi_x1:
                    x0 = roi_x1 - self.patch_size
                    x1 = roi_x1

                if y1 > roi_y1:
                    y0 = roi_y1 - self.patch_size
                    y1 = roi_y1

                patch = [x0, y0, x1, y1]

                if self.exclude_blob_area and self.is_overlapping_blob(
                    patch, blob_areas
                ):
                    continue

                patches.append(patch)

        return patches

    def is_overlapping_blob(self, patch, blob_areas):
        # Convert the patch rectangle to a Shapely polygon
        patch_polygon = Polygon(
            [
                (patch[0], patch[1]),
                (patch[2], patch[1]),
                (patch[2], patch[3]),
                (patch[0], patch[3]),
            ]
        )

        for blob_area in blob_areas:
            # Convert each blob_area to a Shapely polygon
            blob_polygon = Polygon(blob_area)

            # Check for an overlap
            if patch_polygon.intersects(blob_polygon):
                return True
        return False

    def _get_blob_areas(self, image_path):
        json_path = os.path.splitext(image_path)[0] + ".json"
        blob_areas = []
        image_height, image_width = None, None

        # Load the image using OpenCV to get its dimensions
        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except IOError:
            print(f"Error reading image file {image_path}")
            return None

        # Proceed only if the JSON file exists
        if (
            os.path.exists(json_path)
            and image_height is not None
            and image_width is not None
        ):
            try:
                with open(json_path, "r") as file:
                    data = json.load(file)
                    for shape in data.get("shapes", []):
                        points = []
                        if shape.get("shape_type") == "polygon":
                            points = shape.get("points", [])
                        elif shape.get("shape_type") == "rectangle":
                            rectangle_points = shape.get("points", [])
                            if len(rectangle_points) == 2:
                                tl, br = rectangle_points[0], rectangle_points[1]
                                # Define the four corners of the rectangle
                                points = [
                                    (tl[0], tl[1]),  # Top left
                                    (br[0], tl[1]),  # Top right
                                    (br[0], br[1]),  # Bottom right
                                    (tl[0], br[1]),  # Bottom left
                                ]
                        if len(points) >= 3:
                            blob_areas.append(points)
            except Exception as e:
                print(f"Error reading JSON file {json_path}: {e}")

        return blob_areas

    def _get_mask_gt(self, blob_areas, image_height, image_width):

        # Create an empty mask with the same dimensions as the image
        mask_gt = np.zeros((image_height, image_width), dtype=np.uint8)

        # Fill the mask with blob areas
        for points in blob_areas:
            # Convert points to a numpy array, reshape if necessary
            points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

            # Fill the polygon on the mask
            cv2.fillPoly(mask_gt, [points_array], 1)

        return mask_gt[np.newaxis, :, :]

    def get_masks(self):
        masks = {}
        for image_path in self.image_paths:
            blob_areas = self.blob_areas_dict[image_path]
            image_size = self.image_sizes[image_path]

            masks[image_path] = self._get_mask_gt(
                blob_areas, image_size[0], image_size[1]
            )
        return masks

    def process_image(self, image_path, patches):
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            for patch in patches:
                x0, y0, x1, y1 = patch
                cropped_img = img.crop((x0, y0, x1, y1))
                yield {
                    "image": self.transform(cropped_img),
                    "image_path": image_path,
                    "is_anomaly": self.is_anomaly[image_path],
                    "patch": np.array(patch),
                }

    def __len__(self):
        num_samples = 0
        for _, patches in self.patch_dict.items():
            num_samples += len(patches)

        return num_samples

    def __iter__(self):
        for image_path, patches in self.patch_dict.items():
            yield from self.process_image(image_path, patches)


# def _collate(batch):
#     elem = batch[0]
#     elem_type = type(elem)
#     NoneType = type(None)
#     if isinstance(elem, torch.Tensor):
#         out = None
#         if torch.utils.data.get_worker_info() is not None:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum(x.numel() for x in batch)
#             storage = elem.storage()._new_shared(numel)
#             out = elem.new(storage)
#         return torch.stack(batch, 0, out=out)
#     elif elem_type.__module__ == "numpy":
#         return batch
#     elif isinstance(elem, string_classes):
#         return batch
#     elif isinstance(elem, int):
#         return torch.tensor(batch)
#     elif isinstance(elem, list):
#         return batch
#     elif isinstance(elem, NoneType):
#         return batch
#     elif isinstance(elem, collections.abc.Mapping):
#         return {key: _collate([d[key] for d in batch]) for key in elem}
#     elif isinstance(elem, collections.abc.Sequence):
#         # check to make sure that the elements in batch have consistent size
#         it = iter(batch)
#         elem_size = len(next(it))
#         if not all(len(elem) == elem_size for elem in it):
#             raise RuntimeError("each element in list of batch should be of equal size")
#         transposed = zip(*batch)
#         return [_collate(samples) for samples in transposed]

#     raise TypeError("Error")

# def denormalize(img_tensor):
#     mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
#     std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
#     img_denorm = img_tensor * std + mean
#     return img_denorm.clamp(0, 1)

# # Function to denormalize a batch of images
# def denormalize(batch_imgs):
#     imgs_denorm = batch_imgs * torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
#     return imgs_denorm.clamp(0, 1)

# if __name__ == "__main__":
#     dataset = IterablePatchDataset('/home/jay/mnt/hdd01/data/hankook_tire/labeled01/data_toy/train', patch_size=320, input_size=320)        # 224, 320

#     dataloader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)
#     # dataloader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False, collate_fn=_collate)

#     save_dir = './artifacts_debug'
#     os.makedirs(save_dir, exist_ok=True)

#     for i, data in enumerate(dataloader):
#         print(data.keys())
#         image_paths = data['image_path']
#         images = data['image']

#         images = denormalize(images)

#         for j, img in enumerate(images):
#             save_path = os.path.join(save_dir, f'patch_{i}_{j}.png')
#             save_image(img, save_path)
#             print(f"Saved: {save_path}")

# def _collate(batch):
#     elem = batch[0]
#     elem_type = type(elem)
#     NoneType = type(None)
#     if isinstance(elem, torch.Tensor):
#         out = None
#         if torch.utils.data.get_worker_info() is not None:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum(x.numel() for x in batch)
#             storage = elem.storage()._new_shared(numel)
#             out = elem.new(storage)
#         return torch.stack(batch, 0, out=out)
#     elif elem_type.__module__ == "numpy":
#         return batch
#     elif isinstance(elem, string_classes):
#         return batch
#     elif isinstance(elem, int):
#         return torch.tensor(batch)
#     elif isinstance(elem, list):
#         return batch
#     elif isinstance(elem, NoneType):
#         return batch
#     elif isinstance(elem, collections.abc.Mapping):
#         return {key: _collate([d[key] for d in batch]) for key in elem}
#     elif isinstance(elem, collections.abc.Sequence):
#         # check to make sure that the elements in batch have consistent size
#         it = iter(batch)
#         elem_size = len(next(it))
#         if not all(len(elem) == elem_size for elem in it):
#             raise RuntimeError("each element in list of batch should be of equal size")
#         transposed = zip(*batch)
#         return [_collate(samples) for samples in transposed]

#     raise TypeError("Error")

# def denormalize(img_tensor):
#     mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
#     std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
#     img_denorm = img_tensor * std + mean
#     return img_denorm.clamp(0, 1)

# # Function to denormalize a batch of images
# def denormalize(batch_imgs):
#     imgs_denorm = batch_imgs * torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
#     return imgs_denorm.clamp(0, 1)

# if __name__ == "__main__":
#     dataset = IterablePatchDataset('/home/jay/mnt/hdd01/data/hankook_tire/labeled01/data_toy/train', patch_size=320, input_size=320)        # 224, 320

#     dataloader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)
#     # dataloader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False, collate_fn=_collate)

#     save_dir = './artifacts_debug'
#     os.makedirs(save_dir, exist_ok=True)

#     for i, data in enumerate(dataloader):
#         print(data.keys())
#         image_paths = data['image_path']
#         images = data['image']

#         images = denormalize(images)

#         for j, img in enumerate(images):
#             save_path = os.path.join(save_dir, f'patch_{i}_{j}.png')
#             save_image(img, save_path)
#             print(f"Saved: {save_path}")
