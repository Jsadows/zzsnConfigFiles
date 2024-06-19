
import os
import numpy as np 

dest_dir = "test_people_dataset/dataset/sequences/07/velodyne/"
src_dir = "velodyne/human_segmentation/007/maps/"
files = os.listdir(src_dir)

for file_name in files:
	raw_data = np.load(src_dir + file_name).astype(np.float32).reshape(-1, 3)
	ones_column = np.ones((raw_data.shape[0], 1))
	raw_data = np.hstack((raw_data, ones_column))
	with open(dest_dir+"00"+file_name.replace(".npy", ".bin"), 'wb') as f:
		f.write(raw_data.tobytes())

