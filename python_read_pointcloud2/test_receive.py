#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

class PointCloudSubscriber(object):
    def __init__(self): # -> None:
        #self.sub = rospy.Subscriber("pointcloud_topic",
        #                             PointCloud2,
        #                             self.callback, queue_size=5)
        self.sub = rospy.Subscriber("/livox/lidar2",
                                     PointCloud2,
                                     self.callback, queue_size=5)
    def callback(self, msg):
        assert isinstance(msg, PointCloud2)

        # gen=point_cloud2.read_points(msg,field_names=("x","y","z"))
        points = point_cloud2.read_points_list(
            msg, field_names=("x", "y", "z"))

        print(points)


if __name__ =='__main__':
    rospy.init_node("pointcloud_subscriber")
    PointCloudSubscriber()
    rospy.spin()
