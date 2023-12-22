import os
import numpy as np
import h5py
from scipy.io import loadmat
from tqdm import tqdm
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from typing import Tuple, Dict, Optional


def load_OCT_file(
    path_to_file: str
) -> Tuple[np.ndarray, str]:
    """
    Load an OCT file.

    Parameters:
        path_to_file (str): Path to the OCT file.

    Returns:
        Tuple[np.ndarray, str]: A tuple containing the loaded image (np.ndarray) and its label (str).
    """
    fname = os.path.basename(path_to_file)
    label = "CTR" if "Control" in fname else "AMD"
    data = loadmat(path_to_file)
    img = data["images"]

    return img, label

def load_data(
    path_to_dir: str,
) -> Dict[str, Tuple[np.ndarray, str]]:
    """
    Load data from a directory of OCT files.

    Parameters:
        path_to_dir (str): Path to the directory containing OCT files.

    Returns:
        Dict[str, Tuple[np.ndarray, str]]: A dictionary containing file IDs as keys and tuples of images and labels as values.
    """
    data_dict = {}
    file_lst = os.listdir(path_to_dir)
    for fname in tqdm(file_lst):
        curr_img, curr_label = load_OCT_file(os.path.join(path_to_dir, fname))
        file_id = fname.split(".")[0].split("_")[-1] + "_" + curr_label
        data_dict[file_id] = curr_img, curr_label

    return data_dict

def image_preprocess(
    image: np.ndarray,
) -> np.ndarray:
    """
    Transform and save data.

    Parameters:
        data_dict (Dict[str, Tuple[np.ndarray, str]]): Dictionary containing file IDs and tuples of images and labels.
        save_dir (str): Directory to save preprocessed data.
        preprocessing (Optional[bool]): Flag indicating whether to apply preprocessing. Default is True.

    Returns:
        None
    """
    # 1. TV Denoising
    preproc_image = denoise_tv_chambolle(image, weight=0.1, eps=0.001, max_num_iter=100)

    # 2. Adaptive Histogram Equalization (enhance contrast)
    preproc_image = equalize_adapthist(preproc_image)

    return preproc_image

def transform_and_save(
    data_dict: Dict[str, Tuple[np.ndarray, str]],
    save_dir: str,
    preprocessing: Optional[bool] = True
) -> None:
    
    os.makedirs(save_dir, exist_ok=True)
    
    count = 0
    num_file = 0
    for k in tqdm(data_dict.keys()):
        curr_img = data_dict[k][0]
        # If required, preprocess image
        if preprocessing:
            curr_img = image_preprocess(curr_img)
        # Convert to uint8
        curr_img = np.clip((curr_img * 255.0 + 0.5).astype(np.uint8), 0, 255)
        
        # Transform label to number
        curr_label = 1 if data_dict[k][1] == "AMD" else 0
        
        if count == 0:
            # Initialize new dataset objects
            img_dataset = curr_img[np.newaxis, ...]
            label_dataset = []
            label_dataset.append(curr_label)
        elif not (count % 16): 
            # Save current datasets
            fname = f"OCT_dataset_{num_file}.h5"
            num_file += 1
            print(f"Saving dataset {fname} ... Shapes: {img_dataset.shape}, {label_dataset}") 
            with h5py.File(os.path.join(save_dir, fname), "w") as F:
                F.create_dataset(
                    name="images",                    
                    data=img_dataset,
                    dtype=np.uint8
                )
                F.create_dataset(
                    name="labels",
                    data=np.asarray(label_dataset),
                    dtype=np.uint8
                )
            # Initialize new dataset objects
            img_dataset = curr_img[np.newaxis, ...]
            label_dataset = []
            label_dataset.append(curr_label)
        else:
            # Append data to existing datasets
            img_dataset = np.concatenate([img_dataset, curr_img[np.newaxis, ...]], axis=0)
            label_dataset.append(curr_label)
        count += 1

    fname = f"OCT_dataset_{num_file}.h5"
    print(f"Saving dataset {fname} ... Shapes: {img_dataset.shape}, {label_dataset}") 
    with h5py.File(os.path.join(save_dir, fname), "w") as F:
        F.create_dataset(
            name="images",
            data=img_dataset,
            dtype=np.uint8
        )
        F.create_dataset(
            name="labels",
            data=np.asarray(label_dataset),
            dtype=np.uint8
        )

    
if __name__ == "__main__":

    ROOT = r"C:\Users\fede1\OneDrive - Politecnico di Milano\Desktop\Interview_Amsterdam\assignment"
    data_dict = load_data(os.path.join(ROOT, "Data"))
    print("Data loaded!!!")
    print(len(data_dict), data_dict.keys())
    print("--------------------------------------------------------------------------------------")

    transform_and_save(data_dict, os.path.join(ROOT, "Preprocessed_Data"), preprocessing=True)
