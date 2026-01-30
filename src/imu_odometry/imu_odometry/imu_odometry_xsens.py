#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

import numpy as np
from ahrs.filters import Madgwick


class ImuMadgwickOdometry(Node):

    def __init__(self):
        super().__init__('imu_madgwick_odometry')

        self.frame_id = 'odom'
        self.child_id = 'base_link'

        # ---- Thresholds ----
        self.accel_thresh = 0.12
        self.gyro_thresh = 0.03
        self.MOVE_COUNT = 5
        self.STOP_COUNT = 10

        # ---- Limits ----
        self.MAX_ACCEL = 0.4
        self.MAX_VEL = 0.25

        # ---- State ----
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        self.calib_samples = []
        self.calib_count = 300
        self.calibrating = True

        self.last_time = None
        self.motion_counter = 0
        self.static_counter = 0
        self.moving = False

        # ---- Madgwick ----
        self.madgwick = Madgwick(beta=0.03)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

        # ---- ROS ----
        self.sub = self.create_subscription(
            Imu, '/imu/data', self.imu_cb, qos_profile_sensor_data)

        self.pub = self.create_publisher(Odometry, '/imu/odometry', 10)
        self.tf_pub = TransformBroadcaster(self)

        self.get_logger().warn("KEEP IMU COMPLETELY STILL FOR CALIBRATION")

    def imu_cb(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # ---- Calibration ----
        if self.calibrating:
            self.calib_samples.append((accel, gyro))
            if len(self.calib_samples) >= self.calib_count:
                self.accel_bias = np.mean([s[0] for s in self.calib_samples], axis=0)
                self.gyro_bias = np.mean([s[1] for s in self.calib_samples], axis=0)
                self.calibrating = False
                self.last_time = t
                self.get_logger().info("CALIBRATION COMPLETE")
            return

        dt = t - self.last_time
        self.last_time = t
        if dt <= 0.0 or dt > 0.05:
            return

        accel_b = accel - self.accel_bias
        gyro_b = gyro - self.gyro_bias

        # ---- Motion detection ----
        if np.linalg.norm(accel_b) > self.accel_thresh or np.linalg.norm(gyro_b) > self.gyro_thresh:
            self.motion_counter += 1
            self.static_counter = 0
        else:
            self.static_counter += 1
            self.motion_counter = 0

        if not self.moving and self.motion_counter >= self.MOVE_COUNT:
            self.moving = True

        if self.moving and self.static_counter >= self.STOP_COUNT:
            self.moving = False
            self.velocity[:] = 0.0
            self.publish()
            return

        # ---- Orientation (Madgwick) ----
        self.q = self.madgwick.updateIMU(
            q=self.q,
            gyr=gyro_b,
            acc=accel_b,
            dt=dt
        )

        if not self.moving:
            self.velocity[:] = 0.0
            self.publish()
            return

        # ---- Controlled integration ----
        accel_world = self.rotate(self.q, accel_b)
        accel_world[1] = 0.0
        accel_world[2] = 0.0

        accel_world = np.clip(accel_world, -self.MAX_ACCEL, self.MAX_ACCEL)

        self.velocity += accel_world * dt
        self.velocity = np.clip(self.velocity, -self.MAX_VEL, self.MAX_VEL)
        self.position += self.velocity * dt

        self.publish()

    def rotate(self, q, v):
        w, x, y, z = q
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
        ])
        return R @ v

    def publish(self):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_id

        odom.pose.pose.position.x = float(self.position[0])
        odom.pose.pose.position.y = float(self.position[1])
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.w = float(self.q[0])
        odom.pose.pose.orientation.x = float(self.q[1])
        odom.pose.pose.orientation.y = float(self.q[2])
        odom.pose.pose.orientation.z = float(self.q[3])

        self.pub.publish(odom)

        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = self.child_id
        tf.transform.translation.x = self.position[0]
        tf.transform.translation.y = self.position[1]
        tf.transform.translation.z = 0.0
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf_pub.sendTransform(tf)


def main():
    rclpy.init()
    node = ImuMadgwickOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
