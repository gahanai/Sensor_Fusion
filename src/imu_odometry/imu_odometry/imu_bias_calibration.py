#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np

class IMUBiasCalibration(Node):
    """
    Calibrates IMU biases by collecting samples while stationary.
    Publishes the biases to /imu/bias topic.
    """
    
    def __init__(self):
        super().__init__('imu_bias_calibration')
        
        # Parameters
        self.declare_parameter('sample_count', 500)
        self.declare_parameter('publish_rate', 1.0)  # Hz
        
        self.sample_count = self.get_parameter('sample_count').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # State
        self.samples = []
        self.calibrating = True
        self.accel_bias = None
        self.gyro_bias = None
        
        # Subscribers
        self.sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        
        # Publishers
        self.pub = self.create_publisher(Imu, '/imu/bias', 10)
        
        # Timer (will be created after calibration)
        self.timer = None
        
        self.get_logger().warn('='*60)
        self.get_logger().warn('KEEP IMU COMPLETELY STILL!')
        self.get_logger().warn(f'Collecting {self.sample_count} samples...')
        self.get_logger().warn('='*60)
    
    def imu_callback(self, msg: Imu):
        """Collect samples for calibration"""
        if not self.calibrating:
            return
        
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
        
        self.samples.append((accel, gyro))
        
        # Progress indicator
        if len(self.samples) % 50 == 0:
            progress = (len(self.samples) / self.sample_count) * 100
            self.get_logger().info(f'Progress: {progress:.1f}% ({len(self.samples)}/{self.sample_count})')
        
        if len(self.samples) >= self.sample_count:
            self.calculate_biases()
    
    def calculate_biases(self):
        """Calculate biases from collected samples"""
        self.get_logger().info('Calculating biases...')
        
        # Extract arrays
        accels = np.array([s[0] for s in self.samples])
        gyros = np.array([s[1] for s in self.samples])
        
        # Calculate mean (bias)
        accel_bias = np.mean(accels, axis=0)
        gyro_bias = np.mean(gyros, axis=0)
        
        # Calculate standard deviation (noise level)
        accel_std = np.std(accels, axis=0)
        gyro_std = np.std(gyros, axis=0)
        
        # Find gravity axis and remove gravity from bias
        # The axis with highest magnitude (close to 9.81) is gravity
        accel_mags = np.abs(accel_bias)
        gravity_axis = np.argmax(accel_mags)
        
        if accel_mags[gravity_axis] > 8.0:  # Reasonable check for gravity
            self.get_logger().info(f'Detected gravity on axis {gravity_axis} ({["X", "Y", "Z"][gravity_axis]})')
            self.get_logger().info(f'  Value: {accel_bias[gravity_axis]:.3f} m/s²')
            
            # Remove gravity component from bias
            gravity_value = accel_bias[gravity_axis]
            if abs(gravity_value - 9.81) < abs(gravity_value + 9.81):
                accel_bias[gravity_axis] -= 9.81
            else:
                accel_bias[gravity_axis] += 9.81
        else:
            self.get_logger().warn('Could not detect gravity axis clearly!')
            self.get_logger().warn(f'Accel magnitudes: {accel_mags}')
        
        self.accel_bias = accel_bias
        self.gyro_bias = gyro_bias
        
        # Log results
        self.get_logger().info('')
        self.get_logger().info('='*60)
        self.get_logger().info('CALIBRATION COMPLETE!')
        self.get_logger().info('='*60)
        self.get_logger().info('Accelerometer Bias:')
        self.get_logger().info(f'  X: {self.accel_bias[0]:8.5f} m/s² (std: {accel_std[0]:.5f})')
        self.get_logger().info(f'  Y: {self.accel_bias[1]:8.5f} m/s² (std: {accel_std[1]:.5f})')
        self.get_logger().info(f'  Z: {self.accel_bias[2]:8.5f} m/s² (std: {accel_std[2]:.5f})')
        self.get_logger().info('')
        self.get_logger().info('Gyroscope Bias:')
        self.get_logger().info(f'  X: {self.gyro_bias[0]:8.5f} rad/s (std: {gyro_std[0]:.5f})')
        self.get_logger().info(f'  Y: {self.gyro_bias[1]:8.5f} rad/s (std: {gyro_std[1]:.5f})')
        self.get_logger().info(f'  Z: {self.gyro_bias[2]:8.5f} rad/s (std: {gyro_std[2]:.5f})')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Publishing bias at {self.publish_rate} Hz to /imu/bias')
        self.get_logger().info('='*60)
        
        # Start publishing
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_bias)
        self.calibrating = False
    
    def publish_bias(self):
        """Publish bias message"""
        if self.accel_bias is None or self.gyro_bias is None:
            return
        
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        
        msg.linear_acceleration.x = float(self.accel_bias[0])
        msg.linear_acceleration.y = float(self.accel_bias[1])
        msg.linear_acceleration.z = float(self.accel_bias[2])
        
        msg.angular_velocity.x = float(self.gyro_bias[0])
        msg.angular_velocity.y = float(self.gyro_bias[1])
        msg.angular_velocity.z = float(self.gyro_bias[2])
        
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = IMUBiasCalibration()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()