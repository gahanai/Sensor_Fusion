from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    ekf_local_config = PathJoinSubstitution([
        FindPackageShare('sensor_fusion_localization'),
        'config',
        'ekf_local.yaml'
    ])
    
    ekf_global_config = PathJoinSubstitution([
        FindPackageShare('sensor_fusion_localization'),
        'config',
        'ekf_global.yaml'
    ])
    
    navsat_config = PathJoinSubstitution([
        FindPackageShare('sensor_fusion_localization'),
        'config',
        'navsat_dual_ekf.yaml'
    ])
    
    return LaunchDescription([
        
        # ========== STATIC TRANSFORMS ==========
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu',
            arguments=['--x', '0', '--y', '0', '--z', '0',
                       '--roll', '0', '--pitch', '0', '--yaw', '0',
                       '--frame-id', 'base_link', '--child-frame-id', 'imu_link']
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_gps',
            arguments=['--x', '0', '--y', '0', '--z', '0.5',
                       '--roll', '0', '--pitch', '0', '--yaw', '0',
                       '--frame-id', 'base_link', '--child-frame-id', 'gps']
        ),
        
        # ========== EKF LOCAL (IMU only, odom→base_link) ==========
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_local_filter_node',
            output='screen',
            parameters=[ekf_local_config],
            remappings=[
                ('odometry/filtered', '/odometry/local')
            ]
        ),
        
        # ========== NAVSAT TRANSFORM (GPS→map coordinates) ==========
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform_node',
            output='screen',
            parameters=[navsat_config],
            remappings=[
                ('imu/data', '/imu/data'),
                ('gps/fix', '/fix'),
                ('odometry/filtered', '/odometry/local')  # Gets IMU odometry from ekf_local
            ]
        ),
        
        # ========== EKF GLOBAL (GPS, map→odom correction) ==========
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_global_filter_node',
            output='screen',
            parameters=[ekf_global_config],
            remappings=[
                ('odometry/filtered', '/odometry/global')
            ]
        ),
    ])