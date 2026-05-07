from setuptools import setup
from glob import glob

package_name = 'waft_stereo'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alfredo Rivero',
    maintainer_email='alfredo.rivero@outlook.com',
    description='Deep learning stereo depth node using WAFT-Stereo',
    license='MIT',
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    entry_points={
        'console_scripts': [
            'waft_stereo_node = waft_stereo.waft_stereo_node:main',
            'depth_selector_node = waft_stereo.depth_selector_node:main',
        ],
    },
)
