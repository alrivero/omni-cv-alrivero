import os
import sys
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

_this_dir = os.path.dirname(os.path.abspath(__file__))
_UNIDEPTH_ROOT_DEFAULT = os.path.normpath(
    os.path.join(_this_dir, '..', '..', '..', 'external', 'UniDepth')
)


class UniDepthNode(Node):
    def __init__(self):
        super().__init__('unidepth_node')

        self.declare_parameter('unidepth_root', _UNIDEPTH_ROOT_DEFAULT)
        self.declare_parameter('hf_repo', 'lpiccinelli/unidepth-v2-vitl14')
        self.declare_parameter('device', 'cuda')

        unidepth_root = self.get_parameter('unidepth_root').value
        hf_repo       = self.get_parameter('hf_repo').value
        device_str    = self.get_parameter('device').value

        if unidepth_root not in sys.path:
            sys.path.insert(0, unidepth_root)

        from unidepth.models import UniDepthV2
        from unidepth.utils.camera import Pinhole

        self._Pinhole = Pinhole

        self.device = torch.device(device_str)
        self.get_logger().info(f'Loading UniDepthV2 from {hf_repo} ...')

        self.model = UniDepthV2.from_pretrained(hf_repo)
        self.model.interpolation_mode = 'bilinear'
        self.model = self.model.to(self.device).eval()

        self.get_logger().info('UniDepthV2 ready')

        self._use_autocast = self.device.type == 'cuda'
        self.bridge = CvBridge()

        # Only left camera needed — monocular model
        self.img_sub  = Subscriber(self, Image,      '/zed/zedxm/left/color/rect/image')
        self.info_sub = Subscriber(self, CameraInfo, '/zed/zedxm/left/color/rect/image/camera_info')

        self.sync = ApproximateTimeSynchronizer(
            [self.img_sub, self.info_sub],
            queue_size=30,
            slop=0.15,
        )
        self.sync.registerCallback(self.callback)

        self.depth_pub = self.create_publisher(
            Image, '/zed/zedxm/depth/unidepth/depth_registered', 10)
        self.info_pub  = self.create_publisher(
            CameraInfo, '/zed/zedxm/depth/unidepth/depth_registered/camera_info', 10)

        self.get_logger().info('UniDepthNode started')

    def _build_camera(self, info_msg):
        fx, fy = info_msg.k[0], info_msg.k[4]
        cx, cy = info_msg.k[2], info_msg.k[5]
        K = torch.tensor(
            [[fx,  0., cx],
             [0.,  fy, cy],
             [0.,  0., 1.]],
            dtype=torch.float32
        ).unsqueeze(0)
        return self._Pinhole(K=K)

    def callback(self, img_msg, info_msg):
        rgb = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')

        # (H, W, 3) uint8 → (3, H, W) — UniDepthV2 normalises internally
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        camera = self._build_camera(info_msg)

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self._use_autocast,
            ):
                predictions = self.model.infer(rgb_t, camera)

        # (1, 1, H, W) → (H, W) float32 metres
        depth = predictions['depth'].squeeze().cpu().float().numpy()

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header = img_msg.header
        self.depth_pub.publish(depth_msg)

        info_msg.header = img_msg.header
        self.info_pub.publish(info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = UniDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
