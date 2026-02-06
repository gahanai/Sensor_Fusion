from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    ekf_config = PathJoinSubstitution([
        FindPackageShare('sensor_fusion_localization'),
        'config',
        'ekf.yaml'
    ])
    
    navsat_config = PathJoinSubstitution([
        FindPackageShare('sensor_fusion_localization'),
        'config',
        'navsat.yaml'
    ])
    
    return LaunchDescription([
        
        # Static TF: base_link → imu_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu_broadcaster',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link']
        ),
        
        # Static TF: base_link → gps
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_gps_broadcaster',
            arguments=['0', '0', '0.5', '0', '0', '0', 'base_link', 'gps']
        ),
        
        # EKF Filter Node
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_config],  # ← Loading ekf.yaml
            remappings=[
                ('odometry/filtered', 'odometry/local')
            ]
        ),
        
        # Navsat Transform Node
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform_node',
            output='screen',
            parameters=[navsat_config],  # ← THIS MUST BE HERE!
            remappings=[
                ('imu', '/imu/data'),
                ('gps/fix', '/fix'),
                ('odometry/filtered', 'odometry/local')
            ]
        ),
    ])