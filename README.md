# Follow the leader - manipulator controller

This repository contains the code for the follow the leader pruning controller, used to scan up a primary branch via manipulator servoing. 

## Dependencies




## How to run the controller

The launch files take care of running the core nodes required to activate the controller. Note: The ABB launch file does not launch the ABB ROS controller nodes, those should be started separately. In general this package should be agnostic to the type of arm being used, so long as it moveit_servo is configured and running and the camera optical frame is defined.

Once everything has started, you should be ready to run the controller. The controller node file in controller.py advertises two Trigger services, /servo_start and /servo_stop; these can be used to toggle the servoing services.

## Node information

### Core nodes
- controller.py - Processes the mask data, does the curve fitting, and outputs the corresponding velocity command to the moveit_servo node
- image_processor.py - Publishes the optical flow-based foreground segmentations
- point_tracker.py - Accepts requests to start tracking pixels in an image. Runs PIPs and triangulates point correspondences to output 3D point locations.
- visual_servoing.py - WIP

### Utilities
- gui.py - Provides a GUI that connects to the camera feed, allows you to click on points, and visualize the point tracking results from the point tracking node.
- curve_fitting.py - Utilities for Bezier-based curve fitting.
