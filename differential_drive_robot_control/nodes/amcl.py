#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import random
import tf2_ros

class Particle:
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

class AMCLParticleFilter:
    def __init__(self):
        self.num_particles = 100
        self.particles = []

        self.map_resolution = 0.025
        self.map_size = 400  # Map size in meters
        self.map = np.zeros((self.map_size, self.map_size), dtype=np.int8)

        rospy.init_node('amcl_particle_filter')
        self.pose_pub = rospy.Publisher('/amcl_pose', PoseStamped, queue_size=1)
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        rospy.Subscriber('/cmd_vel', Twist, self.control_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)

        # Motion model parameters
        self.motion_noise_v = 0.1  # Standard deviation of velocity noise
        self.motion_noise_w = 0.1  # Standard deviation of angular velocity noise

        # Observation model parameters
        self.observation_noise = 0.1  # Standard deviation of observation noise

    def initialize_particles(self):
        for _ in range(self.num_particles):
            x = random.uniform(-self.map_size * self.map_resolution / 2, self.map_size * self.map_resolution / 2)
            y = random.uniform(-self.map_size * self.map_resolution / 2, self.map_size * self.map_resolution / 2)
            theta = random.uniform(0, 2 * np.pi)
            weight = 1.0 / self.num_particles
            self.particles.append(Particle(x, y, theta, weight))

    def motion_model(self, v, w, dt):
        for particle in self.particles:
            # Add noise to velocity commands
            v_noise = np.random.normal(0, self.motion_noise_v)
            w_noise = np.random.normal(0, self.motion_noise_w)
            
            particle.x += (v + v_noise) * np.cos(particle.theta) * dt
            particle.y += (v + v_noise) * np.sin(particle.theta) * dt
            particle.theta += (w + w_noise) * dt

    def update_particles(self, laser_msg):
        # Update particles based on laser scan data
        ranges = laser_msg.ranges
        angle_min = laser_msg.angle_min
        angle_increment = laser_msg.angle_increment
        weights = []
        for particle in self.particles:
            # Compute the likelihood of the particle generating the observed laser scan data
            likelihood = self.compute_likelihood(particle, ranges, angle_min, angle_increment)
            # Add noise to likelihood
            likelihood += np.random.normal(0, self.observation_noise)
            # Update the weight of the particle accordingly
            particle.weight = likelihood
            # Append the weight to the weights list
            weights.append(likelihood)

        # Normalize the weights
        total_weight = sum(weights)
        if total_weight != 0:
            weights = [w / total_weight for w in weights]

        # Resample particles to maintain diversity
        self.resample_particles(weights)

        # Compute the weighted average of the particles as the pose estimate
        pose_estimate = self.compute_weighted_average()

        return pose_estimate

    def compute_likelihood(self, particle, ranges, angle_min, angle_increment):
        # Placeholder for likelihood computation
        likelihood = 1.0
        return likelihood

    def resample_particles(self, weights):
        # Resample particles to maintain diversity
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=weights)
        self.particles = [self.particles[i] for i in indices]

        # Reset weights
        for particle in self.particles:
            particle.weight = 1.0 / self.num_particles

    def compute_weighted_average(self):
        # Compute the weighted average of the particles as the pose estimate
        x_sum, y_sum, theta_sum = 0.0, 0.0, 0.0
        for particle in self.particles:
            x_sum += particle.x * particle.weight
            y_sum += particle.y * particle.weight
            theta_sum += particle.theta * particle.weight

        # Normalize the sum by the total weight
        total_weight = sum(particle.weight for particle in self.particles)
        x_avg = x_sum / total_weight
        y_avg = y_sum / total_weight
        theta_avg = theta_sum / total_weight

        return Particle(x_avg, y_avg, theta_avg, 1.0)

    def control_callback(self, msg):
        # Callback function for velocity control commands
        v = msg.linear.x
        w = msg.angular.z
        dt = 0.1  # Sample time (s)

        # Update particle motion based on control commands
        self.motion_model(v, w, dt)

    def odom_callback(self, msg):
        # Callback function for odometry data
        pose = msg.pose.pose
        quaternion = pose.orientation
        _, _, theta = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])

        # Update particles with odometry data
        for particle in self.particles:
            particle.x = pose.position.x
            particle.y = pose.position.y
            particle.theta = theta

    def lidar_callback(self, msg):
        # Callback function for lidar data
        # Update particles using laser scan data
        pose_estimate = self.update_particles(msg)
        # Publish pose estimate
        self.publish_pose(pose_estimate)

        # Publish the transformation from /map to /odom
        try:
            trans = self.tf_buffer.lookup_transform("map", "odom", rospy.Time())
            self.tf_broadcaster.sendTransform(trans)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to publish transform: %s", str(e))

    def publish_pose(self, pose_estimate):
        # Publish pose estimate
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = pose_estimate.x
        pose_msg.pose.position.y = pose_estimate.y
        pose_msg.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, pose_estimate.theta)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)

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
        self.initialize_particles()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                # Check for control commands from EKF node
                control_msg = None
                if control_msg is not None:
                    self.control_callback(control_msg)

                # Publish the pose estimate received from AMCL Particle Filter node
                pose_estimate = None
                if pose_estimate is not None:
                    self.publish_pose(pose_estimate)

                # Check for laser scan data from EKF node and update particles
                laser_scan_msg = None
                if laser_scan_msg is not None:
                    self.lidar_callback(laser_scan_msg)
                    
                self.publish_map()
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("ROS Interrupt Exception")


if __name__ == '__main__':
    amcl_pf = AMCLParticleFilter()
    amcl_pf.run()
