# Follow the leader - manipulator controller

This repository contains the code for the follow the leader pruning controller, used to scan up a primary branch via manipulator servoing. The 3D controller is also capable of creating 3D models of the branch features it detects in the environment.





## How to run the controller

First, make sure you have properly installed all the dependencies (see the Dependencies section) and built (`colcon build`) and sourced the ROS2 environment. The following commands should then start all the controllers necessary for the 3D controller:

```
# You can skip this if you set load_core:=true in the next launch file
ros2 launch follow_the_leader core_ftl_3d.launch.py

# This launch file is specifically for the UR5 - Will need to modify for your own hardware/configuration
# You can also use follow_the_leader_sim.launch.py which will launch Blender in the background for rendering
ros2 launch follow_the_leader follow_the_leader.launch.py use_fake_hardware:=false load_core:=false

# Starting the scanning service can be done by sending a /state_announcement message of message type States
# See States.msg for all values, but 0 is idle and 1 is scanning
# Alternatively connect a game controller! 
# A and B on a Nintendo Switch controller start and stop the scanning. I believe it is reversed for an XBox controller 
ros2 topic pub /state_announcement follow_the_leader_msgs/msg/States state:\ 1\ 
```

### Details

The core nodes to run are in the `core_ftl_3d.launch.py` file. The other `follow_the_leader` launch files are bringup files for launching the utilities and configurations necessary to operate on our setup with a UR5e and a RealSense camera; you can replace launching this file with whatever launch file you want that brings up your own robot. In general this package should be agnostic to the type of arm being used, so long as moveit_servo is configured and running and the camera optical frame is defined.

Once everything has started, you should be ready to run the controller. The operation of the system is governed by the state machine defined in `simple_state_manager.py`. This node offers a `/scan_start` and `/scan_stop` service to start and stop the scanning. This is equivalent to publishing a corresponding `States` message to `/state_announcement`. The state manager listens to this topic and sends out a corresponding action to all nodes governed by the state machine.

Note: There is a 2D controller file as well, but the 2D controller is in the process of being phased out and exists mostly for legacy reasons. 

## Node information

### Core nodes
#### General
- `simple_state_manager.py` - A lightweight state machine for managing the behavior of the nodes
- `image_processor.py` - Publishes the optical flow-based foreground segmentations
- `visual_servoing.py` - Given a pixel target and a tracking pixel, it will attempt to visually line up the target and tracking pixels. It will also read in the 3D estimate of the tracked pixel and use it to determine when to stop the servoing. (*Note*: Due to the refactor of the point tracker, this file currently is likely to not be working properly.) 

#### 3D Controller
- `point_tracker.py` - A node that stores in RGB images, runs PIPs when queried, and triangulates point correspondences to output 3D point locations. Can either be synchronously queried for a set of pixels, or can asynchronously send a set of target pixels to start tracking. (*Note*: The latter function may be broken at the moment)
- `curve_3d_model.py` - Builds a 3D model of the tree. Does so by reading in the foreground segmentation masks, running branch detection in the 2D mask, retrieving the corresponding 3D estimates, and stitching together the estimates to form the curve.
- `controller_3d.py` - Subscribes to the 3D model of the tree and uses this information to output velocity commands to maintain a set distance from the branch while following it in a given direction. Also handles rotating the camera around the lookat target to get multiple views of the tree.

#### 2D Controller 

This controller is being phased out in favor of the 3D one, and there is no guarantee it will be supported in the future.

- `controller.py` - Outputs velocity commands to follow the leader in the optical frame by processing the mask data and fitting a curve

### Utilities
- `utils/blender_server.py` - If testing the robot in simulation (use the `follow_the_leader_sim.launch.py` file), this file handles running a Blender instance that creates a mock tree model. It subscribes to the position of the camera and renders images as the robot moves. Note that the Blender rendering is not super fast and so it is not advisable to move the robot too fast.
- `io_manager.py` - Handles reading inputs from a game controller (`/joy`) for convenience.
- `curve_fitting.py` - Utilities for Bezier-based curve fitting.

### Obsolete (delete these later)
- `gui.py` - Provides a GUI that connects to the camera feed, allows you to click on points, and visualize the point tracking results from the point tracking node. Also allows you to test the visual servoing by selecting a single point to be tracked and right-clicking on the pixel to be aligned to. Requires the point_tracker node to be running, as well as the visual seroving node if you're testing that.

## Dependencies

This package depends on the following Python packages:
- skimage
- networkx
- scipy
- torch (see the notes below about building FlowNet2)

Aside from the usual ROS2 dependencies, this project currently makes use of a number of other repositories which need to be installed and configured properly. Unfortunately the installation process is not quite as easy as it should be.

All the following instructions assume that you have cloned each repo into a folder called `repos` located in your user home directory. Otherwise, you will need to go into the wrapper files in follow_the_leader.networks and modify the install path.

- [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch): Used to produce optical flow estimates that are used in the segmentation framework. You must build the custom layers (bash install.sh) and download the weights for the full model (the code currently assumes it is in `~/weights`). Note that this repo is particularly sensitive that your CUDA version matches the one that PyTorch is compiled with, so you may need to downgrade your CUDA if this is the case.
- [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix): Used in image_processor.py to perform segmentation of the RGB + optical flow 6-channel image. Weights are [located here](https://oregonstate.box.com/s/au4cm0o85sx8lnatmczodat958zifnox) and should be unzipped and go in the checkpoints folder in the pix2pix repository.
- [Persistent Independent Particles (PIPs)](https://github.com/aharley/pips): Used in point_tracker.py to perform point tracking. I used a modified version of the repo which has a setup.py file allowing all internal modules to be imported in Python. [TODO: Figure out how to share these modifications] First, install the requirements.txt file from pips. Then download the weights, and then run the setup.py file (`pip install -e .`) and confirm that you can run a command like `import pips.pips as pips`.