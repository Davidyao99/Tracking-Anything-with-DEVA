import os
import imageio
import numpy as np

PATH = "/projects/perception/datasets/4dunderstanding/data/TUM_test2/rgbd_dataset_freiburg3_walking_halfsphere_0000_0240/deva_ram_gsam_window/Annotations/"

for filename in sorted(os.listdir(PATH)):

    res = np.load(os.path.join(PATH, filename))

    res = res == 0

    imageio.imwrite(f"./results/{filename[:-4]}.png.png", res.astype(np.uint8) * 255)