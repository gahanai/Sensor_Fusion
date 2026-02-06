#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

class ImuCovarianceFixer(Node):
    def __init__(self):
        super().__init__('imu_covariance_fixer')
        
        self.sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.pub = self.create_publisher(Imu, '/imu/data/fixed', 10)
        
        # Reasonable covariances for Xsens MTi
        self.orientation_cov = [0.0001, 0.0, 0.0,
                                 0.0, 0.0001, 0.0,
                                 0.0, 0.0, 0.0001]
        
        self.angular_vel_cov = [0.001, 0.0, 0.0,
                                0.0, 0.001, 0.0,
                                0.0, 0.0, 0.001]
        
        self.linear_acc_cov = [0.01, 0.0, 0.0,
                               0.0, 0.01, 0.0,
                               0.0, 0.0, 0.01]
        
        self.get_logger().info('IMU Covariance Fixer started')
    
    def imu_callback(self, msg):
        # Add covariances if they're all zeros
        if all(c == 0.0 for c in msg.orientation_covariance):
            msg.orientation_covariance = self.orientation_cov
        if all(c == 0.0 for c in msg.angular_velocity_covariance):
            msg.angular_velocity_covariance = self.angular_vel_cov
        if all(c == 0.0 for c in msg.linear_acceleration_covariance):
            msg.linear_acceleration_covariance = self.linear_acc_cov
        
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = ImuCovarianceFixer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
