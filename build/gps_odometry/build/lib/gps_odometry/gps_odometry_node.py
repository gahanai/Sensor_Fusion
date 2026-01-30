#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TransformStamped
from ublox_msgs.msg import NavPVT

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler

import math


class GPSOdometryNode(Node):
    """
    GPS → ENU Odometry publisher
    - ENU frame (East, North, Up)
    - Heading from GPS when moving, position delta when slow
    - Designed to work cleanly with robot_localization
    """

    def __init__(self):
        super().__init__('gps_odometry_node')

        # ---------------- PARAMETERS ----------------
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('min_speed_for_heading', 0.2)  # m/s
        self.declare_parameter('heading_smoothing', 0.2)      # [0..1]

        self.frame_id = self.get_parameter('frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.min_speed = self.get_parameter('min_speed_for_heading').value
        self.heading_alpha = self.get_parameter('heading_smoothing').value

        # ---------------- STATE ----------------
        self.ref_lat = None
        self.ref_lon = None
        self.ref_alt = None
        self.initialized = False

        self.prev_x = None
        self.prev_y = None
        self.prev_time = None

        self.last_yaw = None

        # ---------------- ROS I/O ----------------
        self.sub = self.create_subscription(
            NavPVT,
            '/navpvt',
            self.navpvt_callback,
            10
        )

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.fix_pub = self.create_publisher(NavSatFix, '/fix', 10)

        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info('GPS Odometry Node started')

    # =========================================================
    # GPS → ENU
    # =========================================================
    def gps_to_enu(self, lat, lon, alt):
        """Accurate ECEF → ENU using WGS84"""
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = 2 * f - f ** 2

        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        ref_lat_r = math.radians(self.ref_lat)
        ref_lon_r = math.radians(self.ref_lon)

        sin_ref = math.sin(ref_lat_r)
        cos_ref = math.cos(ref_lat_r)
        sin_lon = math.sin(ref_lon_r)
        cos_lon = math.cos(ref_lon_r)

        N_ref = a / math.sqrt(1 - e2 * sin_ref ** 2)

        x_ref = (N_ref + self.ref_alt) * cos_ref * cos_lon
        y_ref = (N_ref + self.ref_alt) * cos_ref * sin_lon
        z_ref = (N_ref * (1 - e2) + self.ref_alt) * sin_ref

        sin_lat = math.sin(lat_r)
        N = a / math.sqrt(1 - e2 * sin_lat ** 2)

        x = (N + alt) * math.cos(lat_r) * math.cos(lon_r)
        y = (N + alt) * math.cos(lat_r) * math.sin(lon_r)
        z = (N * (1 - e2) + alt) * sin_lat

        dx = x - x_ref
        dy = y - y_ref
        dz = z - z_ref

        east = -sin_lon * dx + cos_lon * dy
        north = -sin_ref * cos_lon * dx - sin_ref * sin_lon * dy + cos_ref * dz
        up = cos_ref * cos_lon * dx + cos_ref * sin_lon * dy + sin_ref * dz

        return east, north, up

    # =========================================================
    # HEADING
    # =========================================================
    def compute_heading(self, msg, x, y, now):
        """Stable heading computation with smoothing"""

        yaw = None
        speed = msg.g_speed * 1e-3

        # Use GPS heading when moving
        if msg.heading != 0 and speed > self.min_speed:
            yaw = math.radians(msg.heading * 1e-5)

        # Otherwise use position delta
        elif self.prev_x is not None and self.prev_time is not None:
            dt = (now - self.prev_time).nanoseconds * 1e-9
            if dt > 0:
                dx = x - self.prev_x
                dy = y - self.prev_y
                if math.hypot(dx, dy) > 0.02:
                    yaw = math.atan2(dx, dy)

        # Fallback
        if yaw is None:
            yaw = self.last_yaw if self.last_yaw is not None else 0.0

        # Smooth yaw
        if self.last_yaw is not None:
            dyaw = math.atan2(math.sin(yaw - self.last_yaw),
                              math.cos(yaw - self.last_yaw))
            yaw = self.last_yaw + self.heading_alpha * dyaw

        self.last_yaw = yaw
        return yaw

    # =========================================================
    # CALLBACK
    # =========================================================
    def navpvt_callback(self, msg):
        if msg.fix_type < 2:
            self.get_logger().warn('No valid GPS fix', throttle_duration_sec=5)
            return

        lat = msg.lat * 1e-7
        lon = msg.lon * 1e-7
        alt = msg.height * 1e-3

        if not self.initialized:
            self.ref_lat = lat
            self.ref_lon = lon
            self.ref_alt = alt
            self.initialized = True
            self.get_logger().info('GPS reference initialized')
            return

        x, y, z = self.gps_to_enu(lat, lon, alt)
        now = self.get_clock().now()

        yaw = self.compute_heading(msg, x, y, now)
        q = quaternion_from_euler(0.0, 0.0, yaw)

        vx = msg.vel_e * 1e-3
        vy = msg.vel_n * 1e-3
        vz = -msg.vel_d * 1e-3

        stamp = now.to_msg()

        self.publish_odometry(stamp, x, y, z, q, vx, vy, vz, msg)
        self.publish_navsat(stamp, lat, lon, alt, msg)

        self.prev_x = x
        self.prev_y = y
        self.prev_time = now

    # =========================================================
    # PUBLISHERS
    # =========================================================
    def publish_odometry(self, stamp, x, y, z, q, vx, vy, vz, msg):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz

        h = msg.h_acc * 1e-3
        v = msg.v_acc * 1e-3
        s = msg.s_acc * 1e-3

        odom.pose.covariance[0] = h*h
        odom.pose.covariance[7] = h*h
        odom.pose.covariance[14] = v*v
        odom.pose.covariance[21] = 99999
        odom.pose.covariance[28] = 99999
        odom.pose.covariance[35] = 0.2

        odom.twist.covariance[0] = s*s
        odom.twist.covariance[7] = s*s
        odom.twist.covariance[14] = s*s

        self.odom_pub.publish(odom)

        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.frame_id
            t.child_frame_id = self.child_frame_id
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.tf_broadcaster.sendTransform(t)

    def publish_navsat(self, stamp, lat, lon, alt, msg):
        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = self.child_frame_id
        fix.latitude = lat
        fix.longitude = lon
        fix.altitude = alt
        fix.status.status = 0 if msg.fix_type >= 3 else -1
        fix.status.service = 1
        self.fix_pub.publish(fix)


def main(args=None):
    rclpy.init(args=args)
    node = GPSOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
