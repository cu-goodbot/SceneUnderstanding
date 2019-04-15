#!/usr/bin/env python

"""
ROS wrapper for object detector / scene understanding module.
"""

import rospy
import json
from std_msgs.msg import String
from sensor_msgs.msg import Image

from scene_understanding.msg import DetectedObject, Scene
from src.object_detection import obstacle_detector

# TODO: change to param loaded from config file
UPDATE_RATE = 1.0 # [Hz]

class SUWrapper(object):

    def __init__(self):

        # variables for most recently received rgb and depth images
        self.recent_rgb = None
        self.recent_depth = None
        
        rospy.init_node('scene_understanding')

        # create publishers and subscribers
        rospy.Subscriber('/image_raw', Image, self.rgb_image_cb)
        rospy.Subscriber('/depth_image', Image, self.depth_image_cb)

        pub = rospy.Publisher('/scene_info', Scene, queue_size=10)

        # TODO: instantiate scene understanding module

        # start update loop
        rate = rospy.Rate(UPDATE_RATE)
        while not rospy.is_shutdown():

            # run detector update
            # data = self.detector.update(self.recent_rgb,self.recent_depth)
            data = obstacle_detector(self.recent_rgb, self.recent_depth)
            # data = [{'label': 'chair','bounding_box_corners': [-10,10,-10,10], 'bounding_box_center': [0,0], 'depth': 2.5634},
            #         {'label': 'door', 'bounding_box_corners': [-20,20,-20,20], 'bounding_box_center': [1,1], 'depth': 4.8789}]

            # create msg and publish update
            msg = self.gen_scene_msg(data)

            pub.publish(msg)

            # sleep until next update
            rate.sleep()


    def rgb_image_cb(self,msg):
        """
        RGB image callback.
        """
        pass

    def depth_image_cb(self,msg):
        """
        Depth image callback.
        """
        pass

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