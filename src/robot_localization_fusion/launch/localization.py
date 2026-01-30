from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('robot_localization_fusion'),
        'config',
        'ekf_localization.yaml'
    )
    
    return LaunchDescription([
        # Wheel Odometry
        Node(
            package='robot_localization_fusion',
            executable='wheel_odometry_publisher',
            name='wheel_odometry_publisher',
            parameters=[{'wheelbase': 1.615}]  # UPDATE THIS
        ),
        
        # Local EKF
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_local',
            parameters=[config_file],
            remappings=[('/odometry/filtered', '/odometry/local')]
        ),
        
        # Global EKF
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_global',
            parameters=[config_file],
            remappings=[('/odometry/filtered', '/odometry/global')]
        ),
    ])