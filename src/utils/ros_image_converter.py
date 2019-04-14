#!/usr/bin/env python

"""
Utility function to extract image data and save as a pickle file.

RGB images are 3d numpy arrays, and depth are 2d numpy arrays.
"""

import os
import sys
import rospy
import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


rospy.init_node('image_converter')

# rospy.Subscriber('/gibson_ros/camera/rgb/image',Image)
# rospy.Subscriber('gibson_ros/camera/depth/image',Image)

rgb_msg = rospy.wait_for_message('/gibson_ros/camera/rgb/image',Image)
depth_msg = rospy.wait_for_message('/gibson_ros/camera/depth/image',Image)

bridge = CvBridge()

rgb_image = bridge.imgmsg_to_cv2(rgb_msg,desired_encoding="rgb8")
depth_image = bridge.imgmsg_to_cv2(depth_msg,desired_encoding="passthrough")

with open('rgb_image_array.npy','w') as f:
    np.save(f,rgb_image)

with open('depth_image_array.npy','w') as f:
    np.save(f,depth_image)


    
    