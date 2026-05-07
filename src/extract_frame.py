"""
Extract a single color frame by index from the VIO_stripped MCAP bag and save as PNG.

Usage:
    python3 src/extract_frame.py 0          # first frame
    python3 src/extract_frame.py 100        # 101st frame
    python3 src/extract_frame.py 50 --side right   # right camera
    python3 src/extract_frame.py 50 --out /tmp/frame.png
"""

import argparse
import os
import sys

import numpy as np
import cv2
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


BAG_PATH = os.path.join(
    os.path.expanduser("~"), "Documents", "CV_project", "dataset", "VIO_stripped"
)

TOPIC_MAP = {
    "left":  "/zed/zedxm/left/color/rect/image",
    "right": "/zed/zedxm/right/color/rect/image",
}


def decode_image(ros_msg) -> np.ndarray:
    raw = np.frombuffer(ros_msg.data, dtype=np.uint8)
    enc = ros_msg.encoding.lower()

    if enc in ("rgb8", "bgr8", "mono8", "bgra8", "rgba8"):
        channels = 1 if enc == "mono8" else (4 if enc in ("bgra8", "rgba8") else 3)
        img = raw.reshape((ros_msg.height, ros_msg.width, channels))
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif enc == "bgra8":
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif enc == "rgba8":
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif enc in ("bayer_rggb8", "bayer_bggr8", "bayer_gbrg8", "bayer_grbg8"):
        img = raw.reshape((ros_msg.height, ros_msg.width))
        codes = {
            "bayer_rggb8": cv2.COLOR_BayerRG2BGR,
            "bayer_bggr8": cv2.COLOR_BayerBG2BGR,
            "bayer_gbrg8": cv2.COLOR_BayerGB2BGR,
            "bayer_grbg8": cv2.COLOR_BayerGR2BGR,
        }
        img = cv2.cvtColor(img, codes[enc])
    else:
        raise ValueError(f"Unsupported encoding: {ros_msg.encoding}")

    return img


def main():
    parser = argparse.ArgumentParser(description="Extract a color frame from the VIO bag")
    parser.add_argument("index", type=int, help="Zero-based frame index")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--bag", default=BAG_PATH, help="Path to bag directory")
    parser.add_argument("--out", default=None, help="Output PNG path (default: frame_<index>.png)")
    args = parser.parse_args()

    mcap_files = sorted(
        os.path.join(args.bag, f) for f in os.listdir(args.bag) if f.endswith(".mcap")
    )
    if not mcap_files:
        print(f"No .mcap files found in {args.bag}")
        sys.exit(1)

    topic = TOPIC_MAP[args.side]
    out_path = args.out or f"frame_{args.index}_{args.side}.png"

    print(f"Scanning topic {topic} for frame {args.index} ...")

    found = False
    for mcap_path in mcap_files:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for i, (_, channel, _, ros_msg) in enumerate(
                reader.iter_decoded_messages(topics=[topic])
            ):
                if i == args.index:
                    img = decode_image(ros_msg)
                    cv2.imwrite(out_path, img)
                    stamp = ros_msg.header.stamp
                    print(f"Saved frame {args.index} → {out_path}  ({img.shape[1]}x{img.shape[0]})")
                    print(f"Timestamp  : sec={stamp.sec}  nanosec={stamp.nanosec}")
                    print(f"\nTo seek here during playback:")
                    print(f"  ros2 service call /rosbag2_player/seek rosbag2_interfaces/srv/Seek \"{{time: {{sec: {stamp.sec}, nanosec: {stamp.nanosec}}}}}\"")
                    print(f"  ros2 service call /rosbag2_player/pause rosbag2_interfaces/srv/Pause {{}}")
                    found = True
                    break
        if found:
            break

    if not found:
        print(f"Index {args.index} out of range — bag has fewer than {args.index + 1} frames on {topic}")
        sys.exit(1)


if __name__ == "__main__":
    main()
