from pypcd4 import PointCloud
import os
import numpy as np 
"00 training 01 training syntet 02 val 03 test"
dest_dir = "test_car_seg/dataset/sequences/04/velodyne/"
src_dir = "velodyne/cars_segmentation/004/velodyne/"
files = os.listdir(src_dir)
    
for file_name in files:
        pcd_data = PointCloud.from_path(src_dir+file_name)
        points = np.zeros([len(pcd_data.pc_data['x']), 4], dtype=np.float32)
        points[:, 0] = pcd_data.pc_data['x'].copy()
        points[:, 1] = pcd_data.pc_data['y'].copy()
        points[:, 2] = pcd_data.pc_data['z'].copy()
        points[:, 3] = np.ones(len(pcd_data.pc_data['z'])) #pcd_data.pc_data['intensity'].copy().astype(np.float32)
        with open(dest_dir+"00"+file_name.replace(".pcd", ".bin"), 'wb') as f:
            f.write(points.tobytes())
