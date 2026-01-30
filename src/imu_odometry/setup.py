from setuptools import setup, find_packages

package_name = 'imu_odometry'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    
    install_requires=['setuptools'],
    zip_safe=True,
    
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='IMU-only odometry publisher',
    license='Apache-2.0',
    
    # Remove tests_require and add test dependencies in a different way
    # tests_require=['pytest'],  # Remove this line
    
    entry_points={
        'console_scripts': [
            'odom = imu_odometry.odom:main',
            'imu_odometry_xsens = imu_odometry.imu_odometry_xsens:main',
        ],
    },
)