# Scene Understanding

### CU GoodBot project

The scene understanding package for the GoodBot project for CSCI 7000. The package is composed of the Scene Understanding module, a ROS wrapper class, and custom ROS messages.

## ROS

The ROS wrapper class handles subscribing and publishing to topics for scene and image data.


#### Building

To build the package, clone it to the `src/` folder of your catkin workspace. Then navigate to the root of that workspace and run
```
$ catkin_make
```

#### Running

To run the ROS node, make sure you've sourced your workspace by running
```
$ source <your workspace root>/devel/setup.bash
```
then run
```
$ roslaunch scene_understanding scene_understanding.launch
```

Check that scene information is being published to the `/scene_info` by running
```
$ rostopic echo /scene_info
```

#### Custom ROS messages

This package defines two custom ROS messages: a DetectedObject message, and a Scene message. 

The Scene message contains:

- header -- standard ROS message header of type Header
- objects -- array of DetectedObject messages

The DetectedObject message contains:

- header -- standard ROS message header of type Header
- label -- string name of object
- bounding_box_center -- 2 element float32 array with object bounding box center in image coords
- bounding_box_corners -- 4 element float32 array with bounding box corners in image coords clockwise from bottom left
