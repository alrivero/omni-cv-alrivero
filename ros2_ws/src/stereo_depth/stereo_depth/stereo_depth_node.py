import cv2
import rclpy
import tf2_ros
import numpy as np

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth')

        self.bridge = CvBridge()
        self.baseline = None
        self.fx = None
        self.cx = None
        self.cy = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.left_sub = Subscriber(self, Image, '/zed/zedxm/left/gray/rect/image')
        self.right_sub = Subscriber(self, Image, '/zed/zedxm/right/gray/rect/image')
        self.info_sub = Subscriber(self, CameraInfo, '/zed/zedxm/left/gray/rect/image/camera_info')

        # Synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub, self.info_sub],
            queue_size=10,
            slop=0.05
        )
        self.sync.registerCallback(self.callback)

        # Publishers
        self.depth_pub = self.create_publisher(Image, '/zed/zedxm/depth/sgbm/depth_registered', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/zed/zedxm/depth/sgbm/depth_registered/camera_info', 10)

        # SGBM
        channels = 1
        block_size = 5
        p1_multiplier = 8
        p2_multiplier = 32

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=block_size,
            P1=p1_multiplier * channels * block_size ** 2,
            P2=p2_multiplier * channels * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.get_logger().info('StereoDepthNode started')

    def lookup_baseline(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'zed_left_camera_frame_optical',
                'zed_right_camera_frame_optical',
                rclpy.time.Time()
            )
            t = transform.transform.translation
            self.baseline = abs(t.x)
            self.get_logger().info(f'Baseline: {self.baseline:.4f} m')
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')

    def callback(self, left_msg, right_msg, info_msg):

        if self.baseline is None:
            self.lookup_baseline()
            if self.baseline is None:
                return

        self.fx = info_msg.k[0]
        self.cx = info_msg.k[2]
        self.cy = info_msg.k[5]

        left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
        right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='mono8')

        raw_disp = self.stereo.compute(left, right).astype(np.float32) / 16.0

        depth = np.zeros_like(raw_disp)
        valid = raw_disp > 0
        depth[valid] = (self.fx * self.baseline) / raw_disp[valid]

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header = left_msg.header

        self.depth_pub.publish(depth_msg)

        info_msg.header = left_msg.header
        self.info_pub.publish(info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()