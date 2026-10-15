#!/usr/bin/env python3
# enee408i lab 4 - part c: chase a colored object w/ the camera (ROS 2 Humble)
# pipeline: ros image -> cv_bridge -> bgr -> hsv -> color mask -> biggest contour -> centroid
# then a p-controller steers to keep the centroid in the middle of the frame while creeping forward.
# if it can't see the target it just spins slowly until it finds it again
#
# params (set them on the command line with --ros-args -p):
#   image_topic:=/some/topic    which camera topic to listen to (check ros2 topic list)
#   color:=red                  red / green / blue / yellow
#   show_debug:=true            pops up a window w/ the mask + centroid (needs a display, not over plain ssh)
#
# ex: python3 part_c_vision_tracker.py --ros-args -p color:=green -p show_debug:=true
#     (or ros2 run <your_pkg> part_c_vision_tracker --ros-args -p color:=green)

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


# hsv ranges in opencv units (h: 0-179, s/v: 0-255)
# red wraps around h=0 so it needs two ranges that get or'd together.
# these are decent starting points but you'll probably have to tune them for the lab lighting
COLOR_RANGES = {
    'red':    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    'green':  [((40, 70, 50), (85, 255, 255))],
    'blue':   [((100, 150, 50), (130, 255, 255))],
    'yellow': [((20, 100, 100), (35, 255, 255))],
}

FORWARD_VEL = 0.05      # m/s, slow constant forward while tracking
SEARCH_ANG_VEL = 0.3    # rad/s, slow spin when the target isn't in view
KP = 0.8                # error is normalized to [-1, 1] so this is basically the max turn rate
MAX_ANG_VEL = 1.0       # cap so it doesn't whip around
MIN_AREA = 300          # px^2, contours smaller than this are just noise
STOP_AREA_FRAC = 0.25   # if the blob takes up this much of the frame we're right on it, stop driving forward
                        # (set this above 1.0 if you want it to always keep going forward)
TARGET_TIMEOUT = 0.5    # sec w/o a detection before we say the target is lost


def build_mask(frame_bgr, ranges):
    # bgr -> hsv, then threshold for each range and combine them
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    # open gets rid of little specks, close fills small holes in the blob
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_target(mask):
    # returns (cx, cy, area, contour) for the biggest blob, or None if nothing good is there
    # the [-2] grabs the contour list on opencv 3 (returns 3 things) and 4+ (returns 2)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    if area < MIN_AREA:
        return None

    # centroid from image moments: cx = m10/m00, cy = m01/m00
    m = cv2.moments(biggest)
    if m['m00'] == 0:
        return None
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']
    return cx, cy, area, biggest


class VisionTracker(Node):
    def __init__(self):
        super().__init__('turtlebot3_vision_tracker')

        # these need to be called from the p
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('color', 'blue')
        self.declare_parameter('show_debug', False)

        image_topic = self.get_parameter('image_topic').value
        color = str(self.get_parameter('color').value).lower()
        self.show_debug = bool(self.get_parameter('show_debug').value)

        if color not in COLOR_RANGES:
            self.get_logger().warn("don't know color '%s', falling back to red" % color)
            color = 'red'
        self.ranges = COLOR_RANGES[color]

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # sensor-data qos (best effort, depth 5) so we match camera drivers that publish best effort
        # and always work on recent frames instead of piling up a backlog
        self.create_subscription(Image, image_topic, self.image_callback, qos_profile_sensor_data)

        # latest detection as one tuple (err, area_frac, time_sec) so the control loop
        # never reads half of an old detection and half of a new one
        self.target = None
        self.was_tracking = None

        # control runs on its own 10 hz timer instead of inside the image callback,
        # that way the robot still goes into search mode even if frames stop coming in
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("tracking %s on %s" % (color, image_topic))

    def now_sec(self):
        return self.get_clock().now().seconds_nanoseconds()[0]

    def image_callback(self, msg: Image):
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().warn("cv_bridge error: %s" % e, throttle_duration_sec=2.0)
            return

        h, w = frame.shape[:2]
        mask = build_mask(frame, self.ranges)
        result = find_target(mask)

        if result is not None:
            cx, cy, area, contour = result
            # normalized horizontal error: -1 = far left edge, 0 = dead center, +1 = far right edge
            err = (cx - w / 2.0) / (w / 2.0)
            self.target = (err, area / float(w * h), self.now_sec())

        if self.show_debug:
            debug = frame.copy()
            cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
            if result is not None:
                cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)
                cv2.circle(debug, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            cv2.imshow('tracker', debug)
            cv2.imshow('mask', mask)
            cv2.waitKey(1)

    def compute_cmd(self):
        # returns (twist, tracking) where tracking is True if we can currently see the target
        twist = Twist()
        target = self.target

        if target is not None and self.now_sec() - target[2] < TARGET_TIMEOUT:
            err, area_frac, _ = target
            # p-controller: object right of center (err > 0) -> turn right -> negative angular.z
            # if your camera is mounted flipped and it turns the wrong way, flip this sign
            twist.angular.z = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -KP * err))
            twist.linear.x = 0.0 if area_frac > STOP_AREA_FRAC else FORWARD_VEL
            return twist, True
        else:
            # lost it (or never saw it), slow spin to look around
            twist.linear.x = 0.0
            twist.angular.z = SEARCH_ANG_VEL
            return twist, False

    def control_loop(self):
        twist, tracking = self.compute_cmd()
        if tracking != self.was_tracking:
            self.get_logger().info("target found, tracking" if tracking else "no target, searching")
            self.was_tracking = tracking
        self.pub.publish(twist)

    def stop_robot(self):
        # a single zero twist can get dropped on shutdown, so send a few
        try:
            for _ in range(3):
                self.pub.publish(Twist())
                time.sleep(0.05)
        except Exception:
            pass
        if self.show_debug:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = VisionTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()