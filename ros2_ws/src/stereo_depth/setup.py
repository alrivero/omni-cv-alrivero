from setuptools import setup
import os
from glob import glob

package_name = 'stereo_depth'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alfredo Rivero',
    maintainer_email='alfredo.rivero@outlook.com',
    description='Stereo depth estimation node',
    license='MIT',
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
            'stereo_depth_node = stereo_depth.stereo_depth_node:main',
        ],
    },
)