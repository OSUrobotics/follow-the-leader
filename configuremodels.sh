#!/bin/bash
downloads=~/"follow-the-leader-deps/models"
repos=~/"follow-the-leader-deps"
cp -r "$downloads/pix2pix/checkpoints" "$repos/pytorch-CycleGAN-and-pix2pix/"
mkdir -p "$repos/pips/pips/reference_model"
cp "$downloads/pips/model-000200000.pth" "$repos/pips/pips/reference_model/"
mkdir -p "$repos/flownet2pytorch/weights"
cp "$downloads/FlowNet2_checkpoint.pth.tar" "$repos/flownet2pytorch/weights"
