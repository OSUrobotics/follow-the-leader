# Follow the leader - manipulator controller

This repository contains the code for the follow the leader pruning controller, used to scan up a primary branch via manipulator servoing. The 3D controller is also capable of creating 3D models of the branch features it detects in the environment.

**[Link to paper on arXiv](https://arxiv.org/abs/2309.11580)**

## How to run the controller

First, make sure you have properly installed all the dependencies (see the Dependencies section) and built (`colcon build`) and sourced the ROS2 environment. The following commands should then start all the controllers necessary for the 3D controller:

```
# Real robot
ros2 launch follow_the_leader follow_the_leader.launch.py ur_type:=ur5e use_sim:=false camera_type:=d435 load_core:=true
# Fake simulated robot
ros2 launch follow_the_leader follow_the_leader.launch.py ur_type:=ur5e use_sim:=true load_core:=true launch_blender:=true

# Get Intel RealSense autoexposure running correctly
ros2 param set /camera/camera rgb_camera.enable_auto_exposure True
```

Note that these files are tailored to our specific setup with a RealSense camera and a Universal Robots arm. If you want to use this with a different setup you will need to modify this file as necessary.

The controller is activated by sending a `States` message to the `/state_announcement` topic (current useful values are 0 for idle and 1 for leader scan; see the States.msg definition for more information). For instance, you could manually start it by publishing the following message through the command line:

```
ros2 topic pub /state_announcement follow_the_leader_msgs/msg/States "{state: 1}"
```

However a much more convenient way is to use a game controller. Controller button presses are handled by `io_manager.py` (which is automatically launched by the `core_ftl_3d` launch file). By default, you can press the A (right button) and B (left button) on a Nintendo Switch controller to start and stop the scanning procedure. For other types of controllers you will need to figure out the corresponding button mappings.

For more advanced control, you will want to look at the `run_experiments.py` file in `follow_the_leader.utils`. It is just a regular Python script you can run as follows:

```
cd [ROS2_ROOT]/src/follow_the_leader/follow_the_leader/follow_the_leader/utils

# For simulation
python run_experiments.py sim

# For a real robot
python run_experiments.py ur5e
```

This file offers various additional controls that are useful for operating the system. Many of the features are hacked together for my own use (hence why this is not included as a core file), but the most important ones are:

- Home button: Sends the robot to a designated home position (check the `__main__` section)
- D-Pad Up/Down: Adjusts the speed of the controller. Useful for on the real robot due to a bug with moveit_servo where the actual speed of the robot doesn't match the specified speed (it seems to be scaled down by 10, e.g. specifying a speed of 0.5 causes the controller to move at 0.05 m/s).
- L: For simulation, resets the simulated Blender tree. (Equivalent to calling the `/initialize_tree_spindle` service.)


### Details

The core nodes to run are in the `core_ftl_3d.launch.py` file. The other `follow_the_leader` launch files are bringup files for launching the utilities and configurations necessary to operate on our setup with a UR5e and a RealSense camera; you can replace launching this file with whatever launch file you want that brings up your own robot. In general this package should be agnostic to the type of arm being used, so long as moveit_servo is configured and running and the camera optical frame is defined.

Once everything has started, you should be ready to run the controller. The operation of the system is governed by the state machine defined in `simple_state_manager.py`. This node offers a `/scan_start` and `/scan_stop` service to start and stop the scanning. This is equivalent to publishing a corresponding `States` message to `/state_announcement`. The state manager listens to this topic and sends out a corresponding `StateTransition` message to all nodes listening to the `/state_transition` topic.

Each `StateTransition` contains the time of the transition, the starting state, the ending state, and a list of `NodeAction` messages (essentially a dictionary) assigning a string action to each node. *The string actions are determined by the `simple_state_manager.py` node* inside the `self.transition_table` attribute. **It is not necessary to assign an action to each node!** (E.g. for nodes that only care about the terminal state)

## Setting up simulation

The simulated environment requires some additional assets to run, namely texture files for the tree spindle and the background. **[You can find them at this link.](https://oregonstate.box.com/s/8aam98zpwzqv4e086a76w7phwwreu7re)** The files you should set up are:

- HDRIs: Download them to `~/Pictures/HDRIs`
- Tree textures: Download them to `~/Pictures/tree_textures`. You can use your own textures, or use the provided file `Randomized Textures Used for Training.zip`. 

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

This controller is deprecated and exists only for legacy reasons.

- `controller.py` - Outputs velocity commands to follow the leader in the optical frame by processing the mask data and fitting a curve

### Utilities
- `utils/blender_server.py` - If testing the robot in simulation, this file handles running a Blender instance that creates a mock tree model. It subscribes to the position of the camera and renders images as the robot moves. Note that the Blender rendering is not super fast and so it is not advisable to move the robot too fast.
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

Aside from the usual ROS2 dependencies, this project currently makes use of a number of other repositories which need to be installed and configured properly.

**[Please download the files from this link!](https://oregonstate.box.com/s/4jwnoiy8u1dyvvce2brby5j0usf198ft)** and place them in a folder called `~/follow-the-leader-deps/models` located in your user home directory. Otherwise, you will need to go into the wrapper files in follow_the_leader.networks and modify the install path. Next, clone each repository in `~/follow-the-leader-deps/` and build/configure each package as necessary:

- [FlowNet2](https://github.com/OSUrobotics/flownet2pytorch): Used to produce optical flow estimates that are used in the segmentation framework. You must build the custom layers (bash install.sh) and download the weights for the full model (available in the linked folder as `FlowNet2_checkpoint.pth.tar`, the code currently assumes it is in `weights`). Note that this repo is particularly sensitive that your CUDA version matches the one that PyTorch is compiled with, so you may need to downgrade your CUDA if this is the case. (Or use Docker!)
- [pix2pix](https://github.com/OSUrobotics/pytorch-CycleGAN-and-pix2pix): Used in image_processor.py to perform segmentation of the RGB + optical flow 6-channel image. Weights are included (`checkpoints/synthetic_flow_pix2pix``). No need to build or install anything (these files are accessed via a path hack in `pix2pix.py`).
- [Persistent Independent Particles (PIPs)](https://github.com/OSUrobotics/pips): Used in point_tracker.py to perform point tracking. **This is a slightly modified version of PIPs which allows it to be installed with pip and imported as a module.** First, install the requirements.txt file from PIPs (`pip install -r requirements.txt`). Then run the setup.py file (`pip install -e .`) and confirm that you can run a command like `import pips.pips as pips`. 
- Run `.configure_models.sh`

## Things that need to be improved

- Hardcoded elements that would ideally be configurable: You can find these by searching `# TODO: Hardcoded`
- Especially on the real trials, there is sometimes an issue where a point is identified far into the background and the system cannot recover. Figure out what the source of this is
- Instead of the skeletonization method, we would ideally use instance segmentation output to match up side branches to leaders. This logic would go inside the `run_mask_curve_detection` method in `curve_3d_model.py`.

## What needs to be done to use this in a real pruning trial?

This framework is flexible enough to be modified for the actual pruning trials. The key logic to handle is:

- **More states**: There will need to be states for switching over to pruning mode, doing horizontal scanning with the linear slider, etc. 
- **Collision avoidance**: Technically not essential, but it would be useful to incorporate collision avoidance for forward facing branches.
- **Incorporating the cutter**: This may involve updating the segmentation framework in `image_processor.py` to produce separate masks for the branches and the cutters (only the branch mask should be sent with the ImageMaskPair to the 3D curve modeling node).
- You may wish to reimplement the GUI for easier debugging of things like point tracking failures, etc.