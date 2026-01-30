#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import numpy as np
from tf_transformations import quaternion_matrix


class XsensImuOdometry(Node):

    def __init__(self):
        super().__init__('xsens_imu_odometry')

        # ================= PARAMETERS =================
        self.declare_parameter('mode', 'pure')  # 'pure' or 'demo'
        self.declare_parameter('calibration_samples', 300)
        self.declare_parameter('velocity_damping', 0.99)
        self.declare_parameter('static_velocity_threshold', 0.05)

        self.mode = self.get_parameter('mode').value
        self.calib_samples = self.get_parameter('calibration_samples').value
        self.velocity_damping = self.get_parameter('velocity_damping').value
        self.static_thresh = self.get_parameter('static_velocity_threshold').value

        if self.mode not in ['pure', 'demo']:
            self.get_logger().warn("Invalid mode, defaulting to PURE")
            self.mode = 'pure'

        self.get_logger().info(f"IMU Odometry running in [{self.mode.upper()}] mode")

        # ================= ROS INTERFACES =================
        self.sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 100)

        self.pub = self.create_publisher(
            Odometry, '/imu/odometry', 10)

        # ================= STATE =================
        self.prev_time = None
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.prev_acc_world = np.zeros(3)

        # ================= BIAS CALIBRATION =================
        self.calibrating = True
        self.calibration_data = []
        self.accel_bias = np.zeros(3)

        self.get_logger().info("Keep IMU stationary for bias calibration...")

    # ======================================================
    def imu_callback(self, msg: Imu):

        # ----- Orientation check -----
        if msg.orientation_covariance[0] == -1.0:
            return

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # ================= CALIBRATION =================
        if self.calibrating:
            acc = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            self.calibration_data.append(acc)

            if len(self.calibration_data) >= self.calib_samples:
                self.accel_bias = np.mean(self.calibration_data, axis=0)
                self.calibrating = False
                self.get_logger().info(f"Bias calibration complete: {self.accel_bias}")
            return

        # ================= TIMING =================
        if self.prev_time is None:
            self.prev_time = current_time
            return

        dt = current_time - self.prev_time
        self.prev_time = current_time

        if dt <= 0.0 or dt > 0.1:
            return

        # ================= ORIENTATION =================
        q = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]
        R = quaternion_matrix(q)[0:3, 0:3]

        # ================= ACCELERATION =================
        acc_body = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ]) - self.accel_bias

        acc_world = R @ acc_body

        # ================= INTEGRATION =================
        # Mid-point integration
        acc_mid = 0.5 * (acc_world + self.prev_acc_world)
        self.velocity += acc_mid * dt
        self.position += self.velocity * dt + 0.5 * acc_mid * dt * dt
        self.prev_acc_world = acc_world

        # ================= DEMO MODE STABILIZATION =================
        if self.mode == 'demo':
            vel_mag = np.linalg.norm(self.velocity)
            acc_mag = np.linalg.norm(acc_world)

            if vel_mag < self.static_thresh and acc_mag < 0.2:
                self.velocity *= self.velocity_damping

                if vel_mag < 0.01:
                    self.position *= 0.999

        # ================= PUBLISH ODOMETRY =================
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(self.position[0])
        odom.pose.pose.position.y = float(self.position[1])
        odom.pose.pose.position.z = float(self.position[2])
        odom.pose.pose.orientation = msg.orientation

        odom.twist.twist.linear.x = float(self.velocity[0])
        odom.twist.twist.linear.y = float(self.velocity[1])
        odom.twist.twist.linear.z = float(self.velocity[2])
        odom.twist.twist.angular = msg.angular_velocity

        self.pub.publish(odom)


def main():
    rclpy.init()
    node = XsensImuOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
