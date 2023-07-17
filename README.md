# Follow the leader - manipulator controller

This repository contains the code for the follow the leader pruning controller, used to scan up a primary branch via manipulator servoing. The 3D controller is also capable of creating 3D models of the branch features it detects in the environment.





## How to run the controller

First, make sure you have properly installed all the dependencies (see the Dependencies section) and built (`colcon build`) and sourced the ROS2 environment. The following commands should then start all the controllers necessary for the 3D controller:

```
# You can skip this if you set load_core:=true in the bringup file
ros2 launch follow_the_leader core_ftl_3d.launch.py

# This launch file is specifically for the UR3 - Will need to modify for your own hardware/configuration
ros2 launch follow_the_leader follow_the_leader.launch.py use_fake_hardware:=false load_core:=false

# Call the service - the robot should start scanning and moving!
ros2 service call /servo_3d_start std_srvs/srv/Trigger
```

### Details

The core nodes to run are in the core_ftl_3d.launch.py file. The other follow_the_leader launch files are bringup files for launching the utilities and configurations necessary to operate on a real robot. In general this package should be agnostic to the type of arm being used, so long as moveit_servo is configured and running and the camera optical frame is defined.

Note: The ABB launch file does not launch the ABB ROS controller nodes, those should be started separately. 

Once everything has started, you should be ready to run the controller. The state manager advertises /scan_start and /scan_stop services for starting and stopping the controller. If using the UR3 with buttons attached to digital inputs 0 and 1, you can run `io_manager` and use those buttons to start and stop the robot. Note that this interface may change in the future.

Note: There is a 2D controller file as well, but the 2D controller is in the process of being phased out and exists mostly for legacy reasons. 

## Node information

### Core nodes
#### General
- simple_state_manager.py - A lightweight state machine for managing the behavior of the nodes
- image_processor.py - Publishes the optical flow-based foreground segmentations
- visual_servoing.py - Given a pixel target and a tracking pixel, it will attempt to visually line up the target and tracking pixels. It will also read in the 3D estimate of the tracked pixel and use it to determine when to stop the servoing.

#### 3D Controller
- point_tracker.py - Accepts requests to start tracking pixels in an image. Runs PIPs and triangulates point correspondences to output 3D point locations.
- curve_3d_model.py - Builds a 3D model of the tree by reading in the foreground segmentation masks, running branch detection, initializing tracking points for the point tracker, retrieving the 3D estimates, and stitching together the estimates to form the curve.
- controller_3d.py - Subscribes to the 3D model of the tree and uses this information to output velocity commands to maintain a set distance from the branch while following it in a given direction.

#### 2D Controller 

This controller is being phased out in favor of the 3D one, and there is no guarantee it will be supported in the future.

- controller.py - Outputs velocity commands to follow the leader in the optical frame by processing the mask data and fitting a curve

### Utilities
- gui.py - Provides a GUI that connects to the camera feed, allows you to click on points, and visualize the point tracking results from the point tracking node. Also allows you to test the visual servoing by selecting a single point to be tracked and right-clicking on the pixel to be aligned to. Requires the point_tracker node to be running, as well as the visual seroving node if you're testing that.
- io_manager.py - Quick interface for the UR to easily run the 3D controller via a physical device connected to the UR's digital IO pins.
- curve_fitting.py - Utilities for Bezier-based curve fitting.

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