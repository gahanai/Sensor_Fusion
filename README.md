Launch IMU Driver
ros2 launch xsens_mti_ros2_driver display.launch.py

Run IMU Odometry Node
python3 odom.py

Launch GPS Driver
ros2 launch ublox_gps ublox_gps_node_zedf9p-launch.py

Run GPS + Odometry Node
ros2 run gps_odometry gps_odometry_node.py

Launch EKF Sensor Fusion
ros2 launch robot_localization_fusion localization.py

