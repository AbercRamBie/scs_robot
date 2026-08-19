from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'semantic_comm_runtime'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'assets/world'),
            glob('assets/world/*.sdf')),
        (os.path.join('share', package_name, 'assets/robot'),
            glob('assets/robot/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='subash',
    maintainer_email='subashram773@gmail.com',
    description='Semantic communication robot simulation',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'encoder_node = semantic_comm_runtime.encoder_node:main',
            'channel_node = semantic_comm_runtime.channel_node:main',
            'decoder_node = semantic_comm_runtime.decoder_node:main',
            'nav_node     = semantic_comm_runtime.nav_node:main',
            'joy_control_node = semantic_comm_runtime.joy_control_node:main',            
            'cmd_vel_stamper_node = semantic_comm_runtime.cmd_vel_stamper_node:main',
            'robot_driver_node = semantic_comm_runtime.robot_driver_node:main',        
            'startup_scan_node = semantic_comm_runtime.startup_scan_node:main',
            'vision_node = semantic_comm_runtime.vision_node:main',],
    },
)
