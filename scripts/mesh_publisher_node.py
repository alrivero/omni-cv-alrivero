#!/usr/bin/env python3
"""
=====================================================================
 * MIT License
 *
 * Copyright (c) 2025 Omni Instrument Inc.
 * ...
 * =====================================================================
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from visualization_msgs.msg import Marker


class MeshPublisherNode(Node):
    def __init__(self):
        super().__init__('mesh_publisher')

        self.declare_parameter('mesh_path', os.path.join(os.path.expanduser('~'), 'output', 'mesh.stl'))
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('publish_rate', 1.0)

        mesh_path = self.get_parameter('mesh_path').value
        frame_id  = self.get_parameter('frame_id').value
        rate      = self.get_parameter('publish_rate').value

        if not os.path.isfile(mesh_path):
            self.get_logger().error(f'Mesh file not found: {mesh_path}')
            raise FileNotFoundError(mesh_path)

        # Transient local so RViz2 receives the marker on late-join
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._pub = self.create_publisher(Marker, '/mesh_marker', qos)

        self._marker = Marker()
        self._marker.header.frame_id  = frame_id
        self._marker.ns               = 'mesh'
        self._marker.id               = 0
        self._marker.type             = Marker.MESH_RESOURCE
        self._marker.action           = Marker.ADD
        self._marker.mesh_resource    = f'file://{mesh_path}'
        self._marker.mesh_use_embedded_materials = False
        self._marker.scale.x          = 1.0
        self._marker.scale.y          = 1.0
        self._marker.scale.z          = 1.0
        self._marker.color.r          = 0.75
        self._marker.color.g          = 0.75
        self._marker.color.b          = 0.75
        self._marker.color.a          = 1.0

        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(f'Publishing mesh: {mesh_path}  frame: {frame_id}')

    def _publish(self):
        self._marker.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._marker)


def main(args=None):
    rclpy.init(args=args)
    node = MeshPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
