#!/usr/bin/env python

"""
ROS wrapper for object detector / scene understanding module.
"""

import rospy
import json
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image

from scene_understanding.msg import DetectedObject, Scene
from object_detection import obstacle_detector

# TODO: change to param loaded from config file
UPDATE_RATE = 1.0 # [Hz]

class SUWrapper(object):

    def __init__(self):

        # variables for most recently received rgb and depth images
        self.recent_rgb = None
        self.recent_depth = None

        # counters for receiving images
        self.rgb_cnt = 0
        self.depth_cnt = 0

        # rates for receiving rgb and depth
        self.rgb_rate = 2*UPDATE_RATE # [Hz]
        self.depth_rate = 2*UPDATE_RATE # [Hz]
        
        rospy.init_node('scene_understanding')

        # create publishers and subscribers
        rospy.Subscriber('/kinect2/qhd/image_color', Image, self.rgb_image_cb)
        rospy.Subscriber('/kinect2/sd/image_depth', Image, self.depth_image_cb)

        pub = rospy.Publisher('/scene_info', Scene, queue_size=10)

        # create cv_bridge instance
        bridge = CvBridge()

        # TODO: instantiate scene understanding module

        # start update loop
        rate = rospy.Rate(UPDATE_RATE)
        while not rospy.is_shutdown():

            if (self.recent_rgb is not None) and (self.recent_depth is not None):

                # convert most recent rgb and depth images to numpy arrays
                rgb_array = bridge.imgmsg_to_cv2(self.recent_rgb,desired_encoding="rgb8")
                depth_array = bridge.imgmsg_to_cv2(self.recent_depth,desired_encoding="passthrough")

                # run detector update
                data = obstacle_detector(rgb_array, depth_array)

                # create msg and publish update
                msg = self.gen_scene_msg(data)
                pub.publish(msg)

            # sleep until next update
            rate.sleep()


    def rgb_image_cb(self,msg):
        """
        RGB image callback. Saves image message to be processed during detector update.
        """
        self.recent_rgb = msg

    def depth_image_cb(self,msg):
        """
        Depth image callback. Saves depth message to be processed during detector update.
        """
        self.recent_depth = msg

    def gen_scene_msg(self,data):
        """
        Generate Scene message with information from object detection.

        Inputs:

            data -- list of dictionary entries, each with label, bounding box corners,
                    bounding box center, and depth keys

        Returns:

            msg -- Scene message to be published.
        """
        msg = Scene()

        # populate message with object info
        for obj in data:
            o = DetectedObject(**obj)
            msg.objects.append(o)

        return msg

if __name__ == "__main__":
    SUWrapper()