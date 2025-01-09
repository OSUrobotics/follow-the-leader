#!/usr/bin/env python
import argparse
import os
import sys
from time import time

import cv2
import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode

install_path = os.path.join(os.path.expanduser("~"), "follow-the-leader-deps", "unimatch")
model_path = os.path.join(os.path.split(install_path)[0], "models", "gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth")
sys.path.append(install_path)
from unimatch.unimatch import UniMatch
from utils.flow_viz import flow_to_image, save_vis_flow_tofile

from follow_the_leader.networks.pix2pix import Pix2PixGAN

class UniMatchGANWrapper:
    def __init__(self, size, flow_path=model_path, gan_name="synthetic_flow_pix2pix"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        flow_model = UniMatch(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=False,
            task="flow",
        ).to(self.device)
        if flow_path is not None:
            checkpoint = torch.load(flow_path, map_location=self.device)
            flow_model.load_state_dict(checkpoint["model"])
        self.model = flow_model.eval()
        self.size = [320, 576]
        self.unimatch_resize = Resize(self.size, InterpolationMode.BILINEAR, antialias=True)
        
        self.gan = Pix2PixGAN(
                gan_name,
                input_nc=6,
                output_nc=1,
                output_size=size,
                epoch="best",
            )
        self.gan_resize = Resize((512, 512), antialias=True)
       
        self.orig_resize = Resize(size, antialias=True)
        self.last_img = None
        return

    def flow_forward(self, image1: np.ndarray, image2: np.ndarray):
        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)
        with torch.no_grad():
            results_dict = self.model(
                    image1.to(self.device),
                    image2.to(self.device),
                    attn_type="swin",
                    attn_splits_list=[2],
                    corr_radius_list=[-1],
                    prop_radius_list=[-1],
                    num_reg_refine=1,
                    task="flow",
                )

            pf = results_dict["flow_preds"][-1]
            flow_img = flow_to_image(pf[0].permute(1, 2, 0).cpu().numpy())
            # cv2.imshow("flow", flow_img)
            # cv2.waitKey(0)
            torch.cuda.synchronize()
            del image1, image2, results_dict, pf
            torch.cuda.empty_cache()

            return flow_img
    
    def process(self, image):
        image_tensor = self.unimatch_resize(torch.from_numpy(image).permute(2, 0, 1).float())
        if self.last_img is None:
            self.last_img = image_tensor

        rgb_flow = self.flow_forward(self.last_img, image_tensor)
        self.last_img = image_tensor

        img = np.dstack([image, cv2.resize(rgb_flow, (image.shape[1], image.shape[0]))])
        img_tensor = self.gan.process_input_numpy_array(img)
        img_tensor = self.gan_resize(img_tensor)
        seg = self.gan.forward(img_tensor)

        return rgb_flow, seg

if __name__ == "__main__":
    img1_path = "/home/grimmlins/ftl_ws/src/follow-the-leader/follow_the_leader/test/resources/color-raw-11.jpg"
    img2_path = "/home/grimmlins/ftl_ws/src/follow-the-leader/follow_the_leader/test/resources/color-raw-91.jpg"
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # get size of image
    size = img1.shape[:2]
    size = (size[1], size[0])

    model = UniMatchGANWrapper(size)
    for img in [img1, img2]:
        start = time()
        rgb_flow, res = model.process(img)
        end = time()
        print(f'Processing time: {end-start:.2f}s')
        cv2.imshow("seg", res)
        cv2.waitKey(0)
