#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped


class MapToOdomPublisher:
    def __init__(self):
        rospy.init_node('map_to_odom_publisher')

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Subscribe to the AMCL pose estimate
        rospy.Subscriber('/amcl_pose', PoseStamped, self.amcl_pose_callback)

    def amcl_pose_callback(self, msg:PoseStamped):
        # Create a transform from /map to /odom
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'odom'
        transform.transform.translation.x = msg.pose.position.x
        transform.transform.translation.y = msg.pose.position.y
        transform.transform.translation.z = msg.pose.position.z
        transform.transform.rotation = msg.pose.orientation

        # Publish the transform
        self.tf_broadcaster.sendTransform(transform)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    map_to_odom_publisher = MapToOdomPublisher()
    map_to_odom_publisher.run()
