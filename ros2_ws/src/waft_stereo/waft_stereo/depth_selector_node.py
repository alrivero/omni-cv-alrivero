import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, CameraInfo


class DepthSelectorNode(Node):
    """
    Republishes one of several depth sources to the canonical topic consumed
    by the rest of the pipeline.  Switch sources at runtime:

        ros2 param set /depth_selector method waft
        ros2 param set /depth_selector method sgbm
    """

    SOURCES = {
        'sgbm': {
            'depth': '/zed/zedxm/depth/sgbm/depth_registered',
            'info':  '/zed/zedxm/depth/sgbm/depth_registered/camera_info',
        },
        'waft': {
            'depth': '/zed/zedxm/depth/waft/depth_registered',
            'info':  '/zed/zedxm/depth/waft/depth_registered/camera_info',
        },
        'unidepth': {
            'depth': '/zed/zedxm/depth/unidepth/depth_registered',
            'info':  '/zed/zedxm/depth/unidepth/depth_registered/camera_info',
        },
    }

    def __init__(self):
        super().__init__('depth_selector')
        self.declare_parameter('method', 'sgbm')

        self.depth_pub = self.create_publisher(
            Image, '/zed/zedxm/depth/depth_registered', 10)
        self.info_pub = self.create_publisher(
            CameraInfo, '/zed/zedxm/depth/depth_registered/camera_info', 10)

        self._depth_sub = None
        self._info_sub = None

        self.add_on_set_parameters_callback(self._on_param_change)
        self._apply_method(self.get_parameter('method').value)

    def _apply_method(self, method: str) -> bool:
        if method not in self.SOURCES:
            self.get_logger().error(
                f'Unknown depth method "{method}". Valid options: {list(self.SOURCES)}')
            return False

        if self._depth_sub:
            self.destroy_subscription(self._depth_sub)
        if self._info_sub:
            self.destroy_subscription(self._info_sub)

        src = self.SOURCES[method]
        self._depth_sub = self.create_subscription(
            Image, src['depth'], self.depth_pub.publish, 10)
        self._info_sub = self.create_subscription(
            CameraInfo, src['info'], self.info_pub.publish, 10)

        self.get_logger().info(f'Depth source → {method}')
        return True

    def _on_param_change(self, params):
        ok = True
        for p in params:
            if p.name == 'method':
                ok = self._apply_method(p.value)
        return SetParametersResult(successful=ok)


def main(args=None):
    rclpy.init(args=args)
    node = DepthSelectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
