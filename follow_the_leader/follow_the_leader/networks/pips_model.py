#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np

import pips.pips as pips
import pips.saverloader as saverloader

class PipsTracker:
    def __init__(self, model_dir=None):
        self.model = pips.Pips(stride=4).cuda()
        if model_dir is not None:
            saverloader.load(model_dir, self.model)
        self.model.eval()

    @staticmethod
    def organize_rgb_images(imgs):
        rgbs = torch.stack([torch.from_numpy(img).permute(2,0,1) for img in imgs], dim=0).unsqueeze(0)
        return rgbs

    def track_points(self, pts, imgs):

        # pts should be a N x 2 array
        pts = torch.from_numpy(np.reshape(pts, (1, *pts.shape))).cuda().float()
        rgbs = self.organize_rgb_images(imgs).cuda()

        # Is interpolation to 640x360 necessary here?
        with torch.no_grad():
            trajs = self.model(pts, rgbs, iters=6)[0][-1].squeeze(0).cpu().numpy()
        # indexed by Frame x Point x Dim (2)
        return trajs


