import os
import numpy as np 
dest_dir = "test_people_dataset/dataset/sequences/07/labels/"
src_dir = "velodyne/human_segmentation/007/labels/"
files = os.listdir(src_dir)
for file_name in files:
	labels = np.load(src_dir + file_name).astype(np.uint32).reshape(-1)     
	print(np.unique(labels)) 
	with open(dest_dir+"00"+file_name.replace(".npy", ".label"), 'wb') as f:
		f.write(labels.tobytes())



