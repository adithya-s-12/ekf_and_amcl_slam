#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
import tf

class ExtendedKalmanFilter:
    def __init__(self):
        # Initialize EKF parameters
        self.Q = np.eye(3)  # Process noise covariance matrix
        self.R_odom = np.eye(3)  # Measurement noise covariance matrix for odometry
        self.R_lidar = np.eye(360) * 0.01  # Measurement noise covariance matrix for lidar

        # Initialize state variables
        self.x = np.zeros(3)  # Initial state vector [x, y, theta]
        self.P = np.eye(3)  # Initial state covariance matrix
        self.v = 0.0  # Linear velocity
        self.w = 0.0  # Angular velocity

        # Initialize map
        self.map_resolution = 0.025
        self.map_size = 400  # Map size in meters
        self.map = np.zeros((self.map_size, self.map_size), dtype=np.int8)

        # Initialize ROS node and publishers/subscribers
        rospy.init_node('ekf_slam_node')
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)
        rospy.Subscriber('/cmd_vel', Twist, self.control_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/imu', Imu, self.imu_callback)

    def predict(self, dt):
        # Predict the next state based on motion model
        self.x[0] += self.v * np.cos(self.x[2]) * dt
        self.x[1] += self.v * np.sin(self.x[2]) * dt
        self.x[2] += self.w * dt
        self.P = self.P + self.Q

    def update(self, measurement, R):
        # Update the state estimate based on the combined measurement
        H = np.eye(3)  # Linearized observation matrix
        y = measurement - self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

    def control_callback(self, msg):
        # Callback function for velocity control commands
        self.v = msg.linear.x
        self.w = msg.angular.z

    def odom_callback(self, msg):
        # Callback function for odometry data
        pose = msg.pose.pose
        twist = msg.twist.twist
        # Add small Gaussian noise to odometry measurements
        noise = 0.1  # Adjust noise level as needed
        x_noise = pose.position.x + np.random.normal(0, noise)
        y_noise = pose.position.y + np.random.normal(0, noise)
        theta_noise = 2 * np.arctan2(pose.orientation.z, pose.orientation.w) + np.random.normal(0, noise)
        self.update(np.array([x_noise, y_noise, theta_noise]), self.R_odom)

    def imu_callback(self, msg):
        # Callback function for IMU data
        quaternion = msg.orientation
        self.update(np.array([quaternion.x, quaternion.y, quaternion.w]), self.R_odom)  # Use odometry noise covariance matrix for IMU

    def lidar_callback(self, msg):
        # Callback function for lidar data
        ranges = msg.ranges
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Get the latest odometry information
        try:
            odom_msg = rospy.wait_for_message('/odom', Odometry, timeout=1.0)
        except rospy.ROSException:
            rospy.logwarn("Failed to receive odometry message")
            return
        
        # Get the robot's position from the odometry message
        pose = odom_msg.pose.pose
        x_robot = pose.position.x
        y_robot = pose.position.y
        theta_robot = 2 * np.arctan2(pose.orientation.z, pose.orientation.w)

        for r, theta in zip(ranges, angles):
            if r < msg.range_max:
                # Transform lidar measurement to the map frame
                x_map = x_robot + r * np.cos(theta + theta_robot)
                y_map = y_robot + r * np.sin(theta + theta_robot)
                # Add small Gaussian noise to lidar measurements
                noise = 0.001  # Adjust noise level as needed
                x_noise = x_map + np.random.normal(0, noise)
                y_noise = y_map + np.random.normal(0, noise)
                self.update_map(x_noise, y_noise)

    def update_map(self, x, y):
        # Update map based on lidar measurements
        x_idx = int((y + self.map_size * self.map_resolution / 2) / self.map_resolution)
        y_idx = int((x + self.map_size * self.map_resolution / 2) / self.map_resolution)
        if 0 <= x_idx < self.map_size and 0 <= y_idx < self.map_size:
            if self.map[x_idx, y_idx] == 0:
                self.map[x_idx, y_idx] = 100  # Mark occupied cell

    def publish_map(self):
        # Publish occupancy grid map to /map topic
        grid_map = OccupancyGrid()
        grid_map.header.stamp = rospy.Time.now()
        grid_map.header.frame_id = 'map'
        grid_map.info.resolution = self.map_resolution
        grid_map.info.width = self.map_size
        grid_map.info.height = self.map_size
        grid_map.info.origin.position.x = -self.map_size * self.map_resolution / 2
        grid_map.info.origin.position.y = -self.map_size * self.map_resolution / 2
        grid_map.info.origin.orientation.w = 1.0
        grid_map.data = np.ravel(self.map).tolist()
        self.map_pub.publish(grid_map)
        rospy.loginfo("Map published")

    def run(self):
        # Run SLAM loop
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            try:
                # Perform EKF prediction step
                self.predict(0.1)

                # Publish occupancy grid map
                self.publish_map()
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Failed to get transform from map to odom")

            rate.sleep()

if __name__ == '__main__':
    ekf = ExtendedKalmanFilter()
    ekf.run()
