### TO LAUNCH IMU DRIVER :

ros2 launch xsens_mti_ros2_driver display.launch.py 

### IN NEXT TERMINAL RUN GPS Driver :

ros2 launch ublox_gps ublox_gps_node_zedf9p-launch.py

### IN NEXT TERMINAL RUN Launch EKF Sensor Fusion

ros2 launch sensor_fusion_localization ekf_navsat.launch.py


