import os
import sys
import rclpy
import tf2_ros
import numpy as np
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Three levels up from this file lands at ros2_ws/
_this_dir = os.path.dirname(os.path.abspath(__file__))
_WAFT_ROOT_DEFAULT = os.path.normpath(os.path.join(_this_dir, '..', '..', '..', 'external', 'WAFT-Stereo'))


class WAFTStereoNode(Node):
    def __init__(self):
        super().__init__('waft_stereo')

        self.declare_parameter('waft_root', _WAFT_ROOT_DEFAULT)
        self.declare_parameter('config_file', 'configs/SynLarge/DAv2S-4.yaml')
        self.declare_parameter('hf_repo', 'MemorySlices/WAFT-Stereo')
        self.declare_parameter('hf_filename', 'SynLarge/DAv2S-4.pth')
        self.declare_parameter('device', 'cuda')

        waft_root   = self.get_parameter('waft_root').value
        config_rel  = self.get_parameter('config_file').value
        hf_repo     = self.get_parameter('hf_repo').value
        hf_filename = self.get_parameter('hf_filename').value
        device_str  = self.get_parameter('device').value

        # Make WAFT-Stereo importable (it uses project-relative imports)
        if waft_root not in sys.path:
            sys.path.insert(0, waft_root)

        from bridgedepth.config import get_cfg
        from algorithms.waft import WAFT
        from huggingface_hub import hf_hub_download
        from peft import PeftModel

        config_path = os.path.join(waft_root, config_rel)
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.freeze()

        # DAv2 backbone is loaded by relative path inside dav2.py; must be run
        # from the WAFT-Stereo root so 'depth-anything-ckpts/...' resolves correctly.
        os.chdir(waft_root)

        # Auto-download the DAv2 backbone checkpoint if not already cached.
        # dav2.py uses a CWD-relative path, so we must chdir to waft_root first.
        _DAV2_REPOS = {
            'vitl': 'depth-anything/Depth-Anything-V2-Large',
            'vitb': 'depth-anything/Depth-Anything-V2-Base',
            'vits': 'depth-anything/Depth-Anything-V2-Small',
        }
        dav2_arch = cfg.WAFT.FEATURE_ENCODER.ARCH  # e.g. 'vitl'
        dav2_ckpt_dir = os.path.join(waft_root, 'depth-anything-ckpts')
        dav2_ckpt_path = os.path.join(dav2_ckpt_dir, f'depth_anything_v2_{dav2_arch}.pth')
        if not os.path.exists(dav2_ckpt_path):
            dav2_hf_repo = _DAV2_REPOS.get(dav2_arch, _DAV2_REPOS['vitl'])
            self.get_logger().info(
                f'DAv2 backbone not found at {dav2_ckpt_path}. '
                f'Downloading from {dav2_hf_repo} ...'
            )
            os.makedirs(dav2_ckpt_dir, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=dav2_hf_repo, filename=f'depth_anything_v2_{dav2_arch}.pth'
            )
            import shutil
            shutil.copy2(downloaded, dav2_ckpt_path)
            self.get_logger().info(f'DAv2 backbone saved to {dav2_ckpt_path}')
        else:
            self.get_logger().info(f'DAv2 backbone found: {dav2_ckpt_path}')

        self.get_logger().info(f'Fetching WAFT weights: {hf_repo}/{hf_filename}')
        ckpt_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)

        self.device = torch.device(device_str)
        self.model = WAFT(cfg)
        self.model.eval()
        self.model = self.model.to(self.device)

        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.model.load_state_dict(weights, strict=False)

        # Fuse LoRA weights into base model for faster inference
        for _, module in self.model.named_modules():
            if isinstance(module, PeftModel):
                module.merge_and_unload()

        self._use_autocast = self.device.type == 'cuda'
        self.get_logger().info('WAFT-Stereo model ready')

        self.bridge = CvBridge()
        self.baseline = None
        self.fx = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.left_sub = Subscriber(self, Image, '/zed/zedxm/left/color/rect/image')
        self.right_sub = Subscriber(self, Image, '/zed/zedxm/right/color/rect/image')
        self.info_sub = Subscriber(self, CameraInfo, '/zed/zedxm/left/color/rect/image/camera_info')

        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub, self.info_sub],
            queue_size=600,
            slop=0.15,
        )
        self.sync.registerCallback(self.callback)

        self.depth_pub = self.create_publisher(Image, '/zed/zedxm/depth/waft/depth_registered', 10)
        self.info_pub  = self.create_publisher(CameraInfo, '/zed/zedxm/depth/waft/depth_registered/camera_info', 10)

        self.get_logger().info('WAFTStereoNode started')

    def _lookup_baseline(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'zed_left_camera_frame_optical',
                'zed_right_camera_frame_optical',
                rclpy.time.Time(),
            )
            self.baseline = abs(tf.transform.translation.x)
            self.get_logger().info(f'Baseline: {self.baseline:.4f} m')
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')

    def callback(self, left_msg, right_msg, info_msg):
        if self.baseline is None:
            self._lookup_baseline()
            if self.baseline is None:
                return

        self.fx = info_msg.k[0]

        left  = self.bridge.imgmsg_to_cv2(left_msg,  desired_encoding='rgb8')
        right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='rgb8')

        # (H, W, 3) uint8 -> (1, 3, H, W) float32 on device
        img1 = torch.as_tensor(left,  dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img2 = torch.as_tensor(right, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self._use_autocast):
                result = self.model({'img1': img1, 'img2': img2})

        disp = result['disp_pred'].squeeze(0).cpu().numpy().astype(np.float32)

        depth = np.zeros_like(disp)
        valid = disp > 0
        depth[valid] = (self.fx * self.baseline) / disp[valid]

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header = left_msg.header
        self.depth_pub.publish(depth_msg)

        info_msg.header = left_msg.header
        self.info_pub.publish(info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = WAFTStereoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
