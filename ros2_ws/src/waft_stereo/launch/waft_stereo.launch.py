from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='waft_stereo',
            executable='waft_stereo_node',
            name='waft_stereo',
            output='screen',
            parameters=[{
                # Model variant — swap to DAv2B-4 or DAv2L-5 for higher accuracy
                'config_file': 'configs/SynLarge/DAv2S-4.yaml',
                'hf_repo':     'MemorySlices/WAFT-Stereo',
                'hf_filename': 'SynLarge/DAv2S-4.pth',
                'device':      'cuda',
            }],
        ),
    ])
