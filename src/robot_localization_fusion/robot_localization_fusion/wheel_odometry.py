#!/usr/bin/env python3
"""
Wheel Odometry Publisher
Converts vehicle speed and steering angle to odometry using Ackermann kinematics
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from vehicle_msgs.msg import VehicleMsg, SteeringAngle
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
import math


class WheelOdometryPublisher(Node):
    def __init__(self):
        super().__init__('wheel_odometry_publisher')
        
        # Parameters
        self.declare_parameter('wheelbase', 1.615)  # Distance between front and rear axle (meters)
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('publish_tf', False)  # Let EKF handle TF
        
        self.wheelbase = self.get_parameter('wheelbase').value
        self.frame_id = self.get_parameter('frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        self.publish_tf = self.get_parameter('publish_tf').value
        
        # State variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        self.current_speed_mps = 0.0
        self.current_steering_angle = 0.0
        
        self.last_time = None
        
        # Subscribers
        self.vehicle_sub = self.create_subscription(
            VehicleMsg,
            '/vehicle',
            self.vehicle_callback,
            10
        )
        
        self.steering_sub = self.create_subscription(
            SteeringAngle,
            '/steering_angle',
            self.steering_callback,
            10
        )
        
        # Publisher
        self.odom_pub = self.create_publisher(Odometry, '/wheel/odometry', 10)
        
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
        
        # Timer for odometry computation (50 Hz)
        self.timer = self.create_timer(0.02, self.compute_odometry)
        
        self.get_logger().info(f'Wheel Odometry Publisher initialized')
        self.get_logger().info(f'  Wheelbase: {self.wheelbase} m')
    
    def vehicle_callback(self, msg):
        """Update vehicle speed"""
        self.current_speed_mps = msg.vehicle_speed_mps
    
    def steering_callback(self, msg):
        """Update steering angle"""
        # Use tyre angle (already corrected for sign in your code)
        self.current_steering_angle = msg.tyre_angle_rad
    
    def compute_odometry(self):
        """Compute odometry using Ackermann kinematics"""
        
        current_time = self.get_clock().now()
        
        if self.last_time is None:
            self.last_time = current_time
            return
        
        # Calculate dt
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        
        if dt <= 0 or dt > 0.1:  # Skip if time delta is invalid
            self.last_time = current_time
            return
        
        # Ackermann steering kinematics
        # For small angles, we can use simplified model
        # For larger angles, use proper Ackermann
        
        if abs(self.current_steering_angle) < 0.001:  # Straight line
            # Simple forward kinematics
            dx = self.current_speed_mps * math.cos(self.theta) * dt
            dy = self.current_speed_mps * math.sin(self.theta) * dt
            dtheta = 0.0
        else:
            # Ackermann turning
            # Radius of curvature
            turning_radius = self.wheelbase / math.tan(self.current_steering_angle)
            
            # Angular velocity
            omega = self.current_speed_mps / turning_radius
            
            # Update heading
            dtheta = omega * dt
            
            # Update position (arc motion)
            if abs(omega) > 0.001:
                dx = turning_radius * (math.sin(self.theta + dtheta) - math.sin(self.theta))
                dy = turning_radius * (-math.cos(self.theta + dtheta) + math.cos(self.theta))
            else:
                dx = self.current_speed_mps * math.cos(self.theta) * dt
                dy = self.current_speed_mps * math.sin(self.theta) * dt
        
        # Update pose
        self.x += dx
        self.y += dy
        self.theta += dtheta
        
        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Compute velocities
        vx = self.current_speed_mps  # Linear velocity in x (body frame)
        vy = 0.0  # No lateral velocity in Ackermann model
        
        if abs(self.current_steering_angle) > 0.001:
            omega = self.current_speed_mps * math.tan(self.current_steering_angle) / self.wheelbase
        else:
            omega = 0.0
        
        # Publish odometry
        self.publish_odometry(current_time, vx, vy, omega)
        
        self.last_time = current_time
    
    def publish_odometry(self, stamp, vx, vy, omega):
        """Publish odometry message"""
        
        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        
        # Position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        
        # Orientation
        q = quaternion_from_euler(0, 0, self.theta)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        # Velocity (in body frame)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = 0.0
        
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = omega
        
        # Covariance matrices
        # Position covariance (trust wheel odometry moderately)
        odom.pose.covariance[0] = 0.1   # x
        odom.pose.covariance[7] = 0.1   # y
        odom.pose.covariance[14] = 1e6  # z (don't use)
        odom.pose.covariance[21] = 1e6  # roll (don't use)
        odom.pose.covariance[28] = 1e6  # pitch (don't use)
        odom.pose.covariance[35] = 0.2  # yaw (moderate uncertainty)
        
        # Velocity covariance (trust wheel speed)
        odom.twist.covariance[0] = 0.01  # vx (trust this!)
        odom.twist.covariance[7] = 1e6   # vy (Ackermann = no lateral)
        odom.twist.covariance[14] = 1e6  # vz (don't use)
        odom.twist.covariance[21] = 1e6  # vroll (don't use)
        odom.twist.covariance[28] = 1e6  # vpitch (don't use)
        odom.twist.covariance[35] = 0.05 # vyaw (computed from steering)
        
        self.odom_pub.publish(odom)
        
        # Publish TF if enabled (usually let EKF do this)
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp.to_msg()
            t.header.frame_id = self.frame_id
            t.child_frame_id = self.child_frame_id
            
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdometryPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
