import os
import sys
import torch
import numpy as np
from PIL import Image
import argparse
import torch
from time import time

install_path = os.path.join(os.path.expanduser('~'), 'repos')
sys.path.append(install_path)

from flownet2pytorch.models import FlowNet2
from flownet2pytorch.utils.flow_utils import flow2img

class FlowNetWrapper:

    def __init__(self, cuda=True, weight_path=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--rgb_max", type=float, default=255.)
        parser.add_argument('--fp16', action='store_true',
                            help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
        args = parser.parse_args()
        model = FlowNet2(args)
        if cuda:
            model = model.cuda()
        if args.fp16:
            model = model.half()

        if weight_path is not None:
            model.load_state_dict(torch.load(weight_path)['state_dict'])
        model.eval()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            start = time()
            output = self.model(x)
            # Testing stuff
            torch.cuda.synchronize()
            end = time()
            # print(f'Processing time: {end-start:.2f}s')
            output = output[0].cpu()
            output = output.numpy()
            output = np.transpose(output, (1,2,0))
            output = flow2img(output)
            return output

            # return flow2img(np.transpose(output[0].cpu().numpy(), (1, 2, 0)))

