from setuptools import find_packages, setup

package_name = 'unidepth_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alrivero',
    maintainer_email='placeholder@example.com',
    description='ROS2 node for monocular metric depth using UniDepthV2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'unidepth_node = unidepth_ros.unidepth_node:main',
        ],
    },
)
