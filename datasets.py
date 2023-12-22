import os
import napari
import numpy as np
import h5py
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Optional, Literal


def extract_patches(
    x: torch.tensor, 
    window_size: int
) -> torch.Tensor:
    """
    Input image is of shape: (N, H, W, D). 
    We want to crop it along W dimension.
    Output image is of shape: (N * W / window_size, H, window_size, D)

    Parameters:
        x: torch.Tensor
            Input tensor of size (N, H, W, D).
        window_size: int
            The size of the cropped windows.

    Return:
        torch.Tensor
            Input tensor of size (N * W / window_size, H, window_size, D).
    """
    _, H, W, D = x.shape
    assert not (W % window_size), "Cannot crop image exactly using this window size."

    x = torch.tensor(x)
    x = x.unfold(2, window_size, window_size)
    x = x[:, None, ...]
    x = x.transpose(1, 3).squeeze(3)
    x = x.reshape(-1, H, window_size, D)
    return x



class OCTDataset(Dataset):
    def __init__(
        self, 
        path_to_data: str, 
        split: Literal["train", "val", "test"] = "train",
        patch_size: Optional[int] = None,
        transform: Optional[bool] = False
    ):
        # Load data from h5py files 
        images = []
        labels = []
        data_files = os.listdir(path_to_data)
        data_files = [data_file for data_file in data_files if split in data_file]
        for data_file in tqdm(data_files):
            with h5py.File(os.path.join(path_to_data, data_file), "r") as F:
                curr_img = F["images"][:]
                curr_label = F["labels"][:]

                # Crop in patches
                self.patch_size = patch_size
                if self.patch_size:
                    num_patches = curr_img.shape[2] // self.patch_size
                    curr_patches = extract_patches(curr_img, window_size=self.patch_size)
                    curr_label = torch.tensor(curr_label).repeat_interleave(num_patches)

                images.append(curr_patches)
                labels.append(curr_label)

        # Transform into tensors
        self.images = torch.cat(images, dim=0).unsqueeze(1)
        self.labels = torch.cat(labels, dim=0)

        # Initialize transform
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]

        # if self.transform:
        #     sample = self.transform(sample)

        return img, label


def collate_cat_first_dim(batch):
    # Concatenate along the first dimension
    return torch.cat(batch[0], dim=0)


if __name__ == "__main__":
    val_dataset = OCTDataset(r"Preprocessed_Data", split="val", patch_size=100)
    print(val_dataset.images.shape)
    print(val_dataset.labels.shape)

    # v = napari.Viewer()
    # v.add_image(val_dataset[0][0].numpy())
    # napari.run()

    # custom_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=4,
    #     # collate_fn=collate_cat_first_dim
    # )
    # print("Loaded!!")