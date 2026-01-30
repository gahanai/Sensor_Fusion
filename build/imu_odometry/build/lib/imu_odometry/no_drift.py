#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

import numpy as np
from collections import deque
from ahrs.filters import Madgwick


class OdomPredictor(Node):
    """
    IMU Odometry with AGGRESSIVE drift prevention
    Fixes: Movement appears much longer than actual in RViz
    """

    def __init__(self):
        super().__init__('imu_odometry_predictor')

        # Parameters
        self.declare_parameter('max_imu_queue_length', 1000)
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('initial_z', 0.0)
        self.declare_parameter('use_2d', True)
        self.declare_parameter('use_madgwick', True)
        self.declare_parameter('madgwick_beta', 0.03)
        
        # DRIFT PREVENTION PARAMETERS
        self.declare_parameter('accel_threshold', 0.30)  # Higher threshold
        self.declare_parameter('gyro_threshold', 0.08)   # Higher threshold
        self.declare_parameter('move_count', 8)          # More samples to confirm
        self.declare_parameter('stop_count', 5)          # Fewer to stop quickly
        self.declare_parameter('max_accel', 0.15)        # Lower limit
        self.declare_parameter('max_velocity', 0.10)     # Lower limit
        self.declare_parameter('velocity_decay', 0.95)   # Add decay factor
        self.declare_parameter('zero_velocity_threshold', 0.01)  # Auto-zero threshold
        
        # Get parameters
        self.max_imu_queue_length = self.get_parameter('max_imu_queue_length').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.publish_tf_param = self.get_parameter('publish_tf').value
        initial_z = self.get_parameter('initial_z').value
        self.use_2d = self.get_parameter('use_2d').value
        self.use_madgwick = self.get_parameter('use_madgwick').value
        madgwick_beta = self.get_parameter('madgwick_beta').value
        
        self.accel_thresh = self.get_parameter('accel_threshold').value
        self.gyro_thresh = self.get_parameter('gyro_threshold').value
        self.MOVE_COUNT = self.get_parameter('move_count').value
        self.STOP_COUNT = self.get_parameter('stop_count').value
        self.max_accel = self.get_parameter('max_accel').value
        self.max_vel = self.get_parameter('max_velocity').value
        self.velocity_decay = self.get_parameter('velocity_decay').value
        self.zero_vel_thresh = self.get_parameter('zero_velocity_threshold').value

        # State variables
        self.seq_ = 0
        self.have_odom_ = False
        self.have_bias_ = False
        self.have_orientation_ = False
        self.has_imu_meas = False
        self.prev_imu_time_ = None

        # IMU queue
        self.imu_queue_ = deque(maxlen=self.max_imu_queue_length)

        # Transform
        self.position_ = np.array([0.0, 0.0, initial_z])
        self.orientation_ = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]

        # Velocities
        self.linear_velocity_ = np.zeros(3)
        self.angular_velocity_ = np.zeros(3)

        # Biases
        self.imu_linear_acceleration_bias_ = np.zeros(3)
        self.imu_angular_velocity_bias_ = np.zeros(3)

        # Timestamp
        self.estimate_timestamp_ = self.get_clock().now().to_msg()

        # Gravity
        self.gravity_magnitude = 9.81
        self.kGravity = np.array([0.0, 0.0, -self.gravity_magnitude])

        # Motion detection with moving average
        self.motion_counter = 0
        self.static_counter = 0
        self.moving = False
        self.accel_history = deque(maxlen=10)  # Track recent accelerations
        self.gyro_history = deque(maxlen=10)   # Track recent gyro values

        # Madgwick filter
        if self.use_madgwick:
            self.madgwick = Madgwick(beta=madgwick_beta)
        else:
            self.madgwick = None

        # Initial orientation calibration
        self.init_calib_samples = []
        self.init_calib_count = 100
        self.calibrating_orientation = not self.use_madgwick
        self.init_gravity_vector = None
        self.init_rotation = None

        # QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )

        # Subscribers
        self.imu_sub_ = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, qos_profile)
        
        self.imu_bias_sub_ = self.create_subscription(
            Imu, '/imu/bias', self.imu_bias_callback, qos_profile)

        # Publishers
        self.odom_pub_ = self.create_publisher(Odometry, '/imu/odometry', 100)
        self.tf_broadcaster_ = TransformBroadcaster(self)

        self.get_logger().info('='*60)
        self.get_logger().info('IMU Odometry with AGGRESSIVE Drift Prevention')
        self.get_logger().info('='*60)
        self.get_logger().info(f'  Mode: {"2D" if self.use_2d else "3D"}')
        self.get_logger().info(f'  Orientation: {"Madgwick" if self.use_madgwick else "Integration"}')
        self.get_logger().info(f'  Accel threshold: {self.accel_thresh:.3f} m/s²')
        self.get_logger().info(f'  Gyro threshold: {self.gyro_thresh:.3f} rad/s')
        self.get_logger().info(f'  Max velocity: {self.max_vel:.3f} m/s')
        self.get_logger().info(f'  Velocity decay: {self.velocity_decay:.3f}')
        self.get_logger().info('='*60)
        
        if self.calibrating_orientation:
            self.get_logger().warn('  KEEP IMU STILL FOR CALIBRATION')

    def imu_callback(self, msg: Imu):
        """Main IMU callback"""
        
        # Extract measurements
        imu_linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        
        imu_angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Get orientation from IMU if available
        if msg.orientation_covariance[0] != -1.0:
            self.orientation_ = np.array([
                msg.orientation.w,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z
            ])
            self.have_orientation_ = True
            self.calibrating_orientation = False
            
            self.imu_queue_.append(msg)
            try:
                self.integrate_imu_data(msg, imu_linear_acceleration, imu_angular_velocity)
            except Exception as e:
                self.get_logger().error(f'Integration failed: {e}')
                return
            
            self.publish_odometry()
            self.publish_tf()
            return

        # Skip calibration if using Madgwick
        if self.use_madgwick and self.calibrating_orientation:
            self.calibrating_orientation = False

        # Calibrate initial orientation
        if self.calibrating_orientation:
            self.init_calib_samples.append(imu_linear_acceleration)
            if len(self.init_calib_samples) >= self.init_calib_count:
                self.init_gravity_vector = np.mean(self.init_calib_samples, axis=0)
                gravity_mag = np.linalg.norm(self.init_gravity_vector)
                self.init_gravity_vector = self.init_gravity_vector / gravity_mag
                
                target_gravity = np.array([0.0, 0.0, -1.0])
                self.init_rotation = self.rotation_between_vectors(
                    self.init_gravity_vector, target_gravity)
                
                # Convert to [w, x, y, z]
                self.init_rotation = np.array([
                    self.init_rotation[3],
                    self.init_rotation[0],
                    self.init_rotation[1],
                    self.init_rotation[2]
                ])
                
                self.get_logger().info('Calibration complete!')
                self.calibrating_orientation = False
            return

        # Add to queue
        self.imu_queue_.append(msg)

        # Integrate
        try:
            self.integrate_imu_data(msg, imu_linear_acceleration, imu_angular_velocity)
        except Exception as e:
            self.get_logger().error(f'Integration failed: {e}')
            return

        # Publish
        self.publish_odometry()
        self.publish_tf()

    def imu_bias_callback(self, msg: Imu):
        """Bias callback"""
        self.imu_linear_acceleration_bias_ = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        
        self.imu_angular_velocity_bias_ = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        self.have_bias_ = True
        self.get_logger().info('IMU bias updated')

    def integrate_imu_data(self, msg: Imu, imu_accel, imu_gyro):
        """Integration with AGGRESSIVE drift prevention"""
        
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # First measurement
        if self.prev_imu_time_ is None:
            self.prev_imu_time_ = current_time
            self.estimate_timestamp_ = msg.header.stamp
            return

        # Calculate time delta
        delta_time = current_time - self.prev_imu_time_
        
        if delta_time <= 0.0 or delta_time > 0.1:
            self.prev_imu_time_ = current_time
            return

        # Apply biases
        accel_b = imu_accel - self.imu_linear_acceleration_bias_
        gyro_b = imu_gyro - self.imu_angular_velocity_bias_

        # ========== ORIENTATION ESTIMATION ==========
        
        if not self.have_orientation_:
            if self.use_madgwick and self.madgwick is not None:
                self.orientation_ = self.madgwick.updateIMU(
                    q=self.orientation_,
                    gyr=gyro_b,
                    acc=accel_b,
                    dt=delta_time
                )
                self.orientation_ = self.normalize_quaternion(self.orientation_)
            
            elif self.init_rotation is not None:
                if not hasattr(self, 'orientation_initialized'):
                    self.orientation_ = self.init_rotation
                    self.orientation_initialized = True
                
                delta_angle = delta_time * (gyro_b + self.angular_velocity_) / 2.0
                delta_q = self.axis_angle_to_quaternion_wxyz(delta_angle)
                self.orientation_ = self.quaternion_multiply_wxyz(self.orientation_, delta_q)
                self.orientation_ = self.normalize_quaternion(self.orientation_)
            
            else:
                delta_angle = delta_time * (gyro_b + self.angular_velocity_) / 2.0
                delta_q = self.axis_angle_to_quaternion_wxyz(delta_angle)
                self.orientation_ = self.quaternion_multiply_wxyz(self.orientation_, delta_q)
                self.orientation_ = self.normalize_quaternion(self.orientation_)
        
        self.angular_velocity_ = gyro_b

        # ========== LINEAR VELOCITY/POSITION INTEGRATION ==========
        
        # Get rotation matrix
        R = self.quaternion_to_rotation_matrix_wxyz(self.orientation_)
        
        # Gravity compensation
        gravity_in_body = R.T @ self.kGravity
        accel_true_body = accel_b + gravity_in_body
        accel_world = R @ accel_true_body
        
        # Apply 2D constraints
        if self.use_2d:
            accel_world[1] = 0.0
            accel_world[2] = 0.0
        
        # ========== AGGRESSIVE MOTION DETECTION ==========
        
        if self.use_2d:
            # Add to history
            accel_mag = np.linalg.norm(accel_world)
            gyro_mag = np.linalg.norm(gyro_b)
            
            self.accel_history.append(accel_mag)
            self.gyro_history.append(gyro_mag)
            
            # Use moving average to reduce noise
            avg_accel = np.mean(list(self.accel_history))
            avg_gyro = np.mean(list(self.gyro_history))
            
            # Motion detection with hysteresis
            if avg_accel > self.accel_thresh or avg_gyro > self.gyro_thresh:
                self.motion_counter += 1
                self.static_counter = 0
            else:
                self.static_counter += 1
                self.motion_counter = max(0, self.motion_counter - 1)  # Decay

            # Start moving
            if not self.moving and self.motion_counter >= self.MOVE_COUNT:
                self.moving = True
                self.get_logger().info(f'Motion START (accel={avg_accel:.3f}, gyro={avg_gyro:.3f})')

            # Stop moving
            if self.moving and self.static_counter >= self.STOP_COUNT:
                self.moving = False
                self.linear_velocity_ = np.zeros(3)
                self.get_logger().info(f'Motion STOP - velocity zeroed')
                self.prev_imu_time_ = current_time
                self.estimate_timestamp_ = msg.header.stamp
                self.have_odom_ = True
                return

            # Only integrate if moving
            if self.moving:
                # Aggressive acceleration limiting
                accel_world = np.clip(accel_world, -self.max_accel, self.max_accel)
                
                # Apply dead zone (ignore very small accelerations)
                dead_zone = 0.05
                accel_world = np.where(np.abs(accel_world) < dead_zone, 0.0, accel_world)
                
                # Update velocity
                self.linear_velocity_ += accel_world * delta_time
                
                # Apply velocity decay (reduces accumulated errors)
                self.linear_velocity_ *= self.velocity_decay
                
                # Aggressive velocity limiting
                self.linear_velocity_ = np.clip(self.linear_velocity_, -self.max_vel, self.max_vel)
                
                # Auto-zero very small velocities
                self.linear_velocity_ = np.where(
                    np.abs(self.linear_velocity_) < self.zero_vel_thresh, 
                    0.0, 
                    self.linear_velocity_
                )
                
                # Update position
                self.position_ += self.linear_velocity_ * delta_time
                
            else:
                # Not moving - ensure zero velocity
                self.linear_velocity_ = np.zeros(3)
        
        else:
            # 3D mode
            self.linear_velocity_ += accel_world * delta_time
            self.linear_velocity_ = np.clip(self.linear_velocity_, -self.max_vel, self.max_vel)
            self.position_ += self.linear_velocity_ * delta_time

        # Update timestamps
        self.prev_imu_time_ = current_time
        self.estimate_timestamp_ = msg.header.stamp
        self.have_odom_ = True

    def publish_odometry(self):
        """Publish odometry"""
        if not self.have_odom_:
            return
        
        msg = Odometry()
        msg.header.frame_id = self.odom_frame_id
        msg.header.stamp = self.estimate_timestamp_
        msg.child_frame_id = self.base_frame_id

        # Position
        msg.pose.pose.position.x = float(self.position_[0])
        msg.pose.pose.position.y = float(self.position_[1]) if not self.use_2d else 0.0
        msg.pose.pose.position.z = float(self.position_[2])
        
        # Orientation
        msg.pose.pose.orientation.w = float(self.orientation_[0])
        msg.pose.pose.orientation.x = float(self.orientation_[1])
        msg.pose.pose.orientation.y = float(self.orientation_[2])
        msg.pose.pose.orientation.z = float(self.orientation_[3])

        # Velocity
        msg.twist.twist.linear.x = float(self.linear_velocity_[0])
        msg.twist.twist.linear.y = float(self.linear_velocity_[1])
        msg.twist.twist.linear.z = float(self.linear_velocity_[2])
        
        msg.twist.twist.angular.x = float(self.angular_velocity_[0])
        msg.twist.twist.angular.y = float(self.angular_velocity_[1])
        msg.twist.twist.angular.z = float(self.angular_velocity_[2])

        # Covariance
        msg.pose.covariance = [0.01 if i % 7 == 0 else 0.0 for i in range(36)]
        msg.twist.covariance = [0.01 if i % 7 == 0 else 0.0 for i in range(36)]

        self.odom_pub_.publish(msg)

    def publish_tf(self):
        """Publish transform"""
        if not self.have_odom_ or not self.publish_tf_param:
            return
        
        tf = TransformStamped()
        tf.header.frame_id = self.odom_frame_id
        tf.header.stamp = self.estimate_timestamp_
        tf.child_frame_id = self.base_frame_id
        
        tf.transform.translation.x = float(self.position_[0])
        tf.transform.translation.y = float(self.position_[1]) if not self.use_2d else 0.0
        tf.transform.translation.z = float(self.position_[2])
        
        tf.transform.rotation.w = float(self.orientation_[0])
        tf.transform.rotation.x = float(self.orientation_[1])
        tf.transform.rotation.y = float(self.orientation_[2])
        tf.transform.rotation.z = float(self.orientation_[3])

        self.tf_broadcaster_.sendTransform(tf)

    # ==================== UTILITY FUNCTIONS ====================
    
    @staticmethod
    def rotation_between_vectors(v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        
        if dot < -0.999999:
            return np.array([0.0, 1.0, 0.0, 0.0])
        
        s = np.sqrt((1 + dot) * 2)
        inv_s = 1 / s
        
        return np.array([
            cross[0] * inv_s,
            cross[1] * inv_s,
            cross[2] * inv_s,
            s * 0.5
        ])
    
    @staticmethod
    def quaternion_multiply_wxyz(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def quaternion_to_rotation_matrix_wxyz(q):
        w, x, y, z = q
        
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ])
    
    @staticmethod
    def axis_angle_to_quaternion_wxyz(axis_angle):
        angle = np.linalg.norm(axis_angle)
        
        if angle < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        axis = axis_angle / angle
        half_angle = angle / 2.0
        
        return np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle)
        ])
    
    @staticmethod
    def normalize_quaternion(q):
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm


def main(args=None):
    rclpy.init(args=args)
    
    node = OdomPredictor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()