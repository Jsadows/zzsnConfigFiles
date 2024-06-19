"""
Car Seg dataset

Author: Marcel Zawadzki (config not the dataset)

"""

import os
import numpy as np
from pypcd4 import PointCloud

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class CarSegSynthDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        seq_list = [4]
        
        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]

        # Split data into train, val, test
        num_files = len(data_list)
        num_train = int(0.6 * num_files)  # 60% for training
        num_val = int(0.2 * num_files)    # 20% for validation
        num_test = num_files - num_train - num_val  # Remaining 20% for testing

        if self.split == 'train':
            data_list = data_list[:num_train]
        elif self.split == 'val':
            data_list = data_list[num_train:num_train + num_val]
        elif self.split == 'test':
            data_list = data_list[num_train + num_val:]
        else:
            raise NotImplementedError
        
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        pc: PointCloud = PointCloud.from_path(data_path)
        array: np.ndarray = pc.numpy()
        scan = array.reshape(-1, 3)
        coord = scan[:, :3]
        strength = np.ones(len(scan)).reshape([-1,1])

        label_file = data_path.replace("velodyne", "labels").replace(".pcd", ".npy")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.load(a).astype(np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: 0,
	    1: 1,
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            0: 0,
	    1: 1,
        }
        return learning_map_inv
