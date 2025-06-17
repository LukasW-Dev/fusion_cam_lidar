import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    config_path = os.path.expanduser('~/ros2_ws/src/fusion_cam_lidar/config/warthog.yaml')

    return LaunchDescription([
        Node(
            package='fusion_cam_lidar',
            executable='fusion_node',
            name='fusion_node',
            output='screen',
            parameters=[config_path]
        )
    ])
