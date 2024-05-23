#!/usr/bin/env python3
import os
import cv2
import sys
import numpy as np

""" Create mask images assuming data formatted as "labelled_id.png" is available at 
one folder level below the path passed to the script
"""

LABELS = {
    (69, 244, 139): "spur",
    (255, 255, 255): "trunk",
    (87, 192, 255): "branch",
    (0, 0, 244): "leader",
}

# (0 0 0): background tree,
# (255, 90, 67): sky,
# (255 0 0) poles,
# (0 0 244) leader,

def generate_masks(files, output_dir):
    for file in files:
        img = cv2.imread(file)
        # black mask
        curr_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for color in LABELS.keys():
            color_arr = np.array(color)
            
            mask = cv2.inRange(img, color_arr - 10, color_arr + 10)
            # or with previous mask
            curr_mask = cv2.bitwise_or(mask, curr_mask)
            start_index = file.find("labelled") + len("labelled")
            # mask_name = f"mask_{LABELS[color]}_{file[start_index:]}"
            # print(f"Saving mask {mask_name} in {output_dir}")
            # cv2.imwrite(os.path.join(output_dir, mask_name), mask)
        mask_name = f"mask_all_{file[start_index:]}"
        cv2.imwrite(os.path.join(output_dir, mask_name), curr_mask)
        print(f"Saving mask {mask_name} in {output_dir}")

if __name__ == '__main__':
    dir_path = sys.argv[1]
    output_dir = dir_path
    os.makedirs(output_dir, exist_ok=True)
    for subfolder in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subfolder)
        files = [os.path.join(subdir_path, x) for x in os.listdir(subdir_path) if x.endswith(".png") and x.startswith("labelled")]
        generate_masks(files, os.path.join(output_dir, subfolder))

