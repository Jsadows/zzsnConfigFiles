from typing import Callable, List, Optional, Union

import numpy as np
from mmdet3d.datasets import Seg3DDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class CarsSegDataset(Seg3DDataset):
    """NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='', img='', pts_instance_mask='', pts_semantic_mask='').
        pipeline (List[dict or Callable]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input, it
            usually has following keys:

            - use_camera: bool
            - use_lidar: bool

            Defaults to dict(use_lidar=True, use_camera=False).
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.classes) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (str or np.ndarray, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        'classes': ('background', 'car'),
        'palette': [[255, 120, 50], [255, 192, 203], [255, 255, 0]],
        'seg_valid_class_ids':
        tuple(range(2)),
        'seg_all_class_ids':
        tuple(range(2)),
    }

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        pts_semantic_mask_path = osp.join(self.data_root,"")

        anns_results = dict(pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def get_seg_label_mapping(self, metainfo):
        seg_label_mapping = np.zeros(metainfo['max_label'] + 1, dtype=np.int64)
        for idx in metainfo['seg_label_mapping']:
            seg_label_mapping[idx] = metainfo['seg_label_mapping'][idx]
        return seg_label_mapping
