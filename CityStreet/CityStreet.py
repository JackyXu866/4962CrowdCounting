import torch
import torch.utils.data
import pandas as pd
import json
import numpy as np

from torchvision.io import read_image
from typing import *
import os


class CityStreet(torch.utils.data.Dataset):
    def __init__(
        self,
        path: os.PathLike = "./data",
        train: bool = True,
        view: int = 1,
        skip_empty: bool = True,
        target_resize_factor: int = 1,
        transform=None,
        target_transform=None
    ) -> None:
        self.view_to_cam = [0, 1, 3, 4]

        for dir in ["labels", "image_frames", "ROI_maps"]:
            if not os.path.exists(os.path.join(path, dir)):
                raise FileNotFoundError("The directory {} does not exist".format(dir))

        if view not in [1, 2, 3]:
            raise ValueError("The view should be 1, 2, or 3")

        self.path = path
        self.view = view
        self.train = train
        self.skip_empty = skip_empty
        self.target_resize_factor = target_resize_factor
        self.transform = transform
        self.target_transform = target_transform

        self.img_path = os.path.join(self.path, "image_frames", f"camera{self.view_to_cam[self.view]}")
        label_path = os.path.join(self.path, "labels", f"via_region_data_view{self.view}.json")
        roi_path = os.path.join(self.path, "ROI_maps", "ROIs", "camera_view", f"mask{self.view}_ic.npz")

        self.labels = self.__process_labels(label_path, self.__read_roi(roi_path))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.img_path, self.labels.iloc[index].name)
        img = read_image(img_path) / 255.0
        hmap = self.__create_heatmap(img, self.labels.iloc[index]["regions"])

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            hmap = self.target_transform(hmap)

        return img, hmap

    def get_original(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.img_path, self.labels.iloc[index].name)
        img = read_image(img_path)

        return img, self.labels.iloc[index]["regions"]

    def __process_labels(self, path: os.PathLike, roi: np.ndarray) -> pd.DataFrame:
        with open(path, "r") as f:
            label_json: dict = json.load(f)

        labels = pd.DataFrame(label_json.values(), index=label_json.keys())

        if self.train:
            labels = labels[labels["filename"] <= "frame_1234.jpg"]
        else:
            labels = labels[labels["filename"] > "frame_1234.jpg"]

        labels.drop(
            columns=["fileref", "filename", "base64_img_data", "size", "file_attributes"],
            inplace=True
        )

        if self.skip_empty:
            labels = labels[labels["regions"].apply(lambda x: len(x) > 0)]

        labels["regions"] = labels["regions"].apply(
            lambda x: [(v["shape_attributes"]["cx"], v["shape_attributes"]["cy"]) for v in x.values()]
        )

        labels["regions"] = labels["regions"].apply(
            lambda x: [v for v in x if self.__check_roi(v[0], v[1], roi)]
        )

        return labels

    def __create_heatmap(self, image: torch.Tensor, labels: List[Tuple[int, int]]) -> torch.Tensor:
        heatmap = torch.zeros(
            int(image.shape[1] / self.target_resize_factor),
            int(image.shape[2] / self.target_resize_factor)
        )

        for label in labels:
            label1 = int(label[1] / self.target_resize_factor)
            label0 = int(label[0] / self.target_resize_factor)
            if label1 >= heatmap.shape[0] or label0 >= heatmap.shape[1]:
                continue
            heatmap[label1][label0] += 1

        return heatmap.unsqueeze(0)

    def __read_roi(self, roi: os.PathLike) -> np.ndarray:
        with np.load(roi) as data:
            return data["arr_0"]

    def __check_roi(self, x: int, y: int, roi: np.ndarray) -> bool:
        if x not in range(roi.shape[1]) or y not in range(roi.shape[0]):
            return False
        return roi[y][x] ==1


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = CityStreet("", True, 1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

