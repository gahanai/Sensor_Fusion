# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/dell/sensor_fusion/src/xsens_mti_ros2_driver/lib/xspublic")
  file(MAKE_DIRECTORY "/home/dell/sensor_fusion/src/xsens_mti_ros2_driver/lib/xspublic")
endif()
file(MAKE_DIRECTORY
  "/home/dell/sensor_fusion/build/xsens_mti_ros2_driver/xspublic/src/xspublic-build"
  "/home/dell/sensor_fusion/build/xsens_mti_ros2_driver/xspublic"
  "/home/dell/sensor_fusion/build/xsens_mti_ros2_driver/xspublic/tmp"
  "/home/dell/sensor_fusion/build/xsens_mti_ros2_driver/xspublic/src/xspublic-stamp"
  "/home/dell/sensor_fusion/build/xsens_mti_ros2_driver/xspublic/src"
  "/home/dell/sensor_fusion/build/xsens_mti_ros2_driver/xspublic/src/xspublic-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/dell/sensor_fusion/build/xsens_mti_ros2_driver/xspublic/src/xspublic-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/dell/sensor_fusion/build/xsens_mti_ros2_driver/xspublic/src/xspublic-stamp${cfgdir}") # cfgdir has leading slash
endif()
