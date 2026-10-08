#!/usr/bin/env python3
# enee408i lab 4 - part c: chase a colored object w/ the camera
# pipeline: ros image -> cv_bridge -> bgr -> hsv -> color mask -> biggest contour -> centroid
# then a p-controller steers to keep the centroid in the middle of the frame while creeping forward.
# if it can't see the target it just spins slowly until it finds it again
#
# params (set them on the command line):
#   _image_topic:=/some/topic   which camera topic to listen to (check rostopic list)
#   _color:=red                 red / green / blue / yellow
#   _show_debug:=true           pops up a window w/ the mask + centroid (needs a display, not over plain ssh)
#
# ex: rosrun <your_pkg> part_c_vision_tracker.py _color:=green _show_debug:=true

import rospy
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


class VisionTracker:
    def __init__(self):
        rospy.init_node('turtlebot3_vision_tracker', anonymous=True)

        # real turtlebot3 w/ the pi camera publishes raw frames on /raspicam_node/image
        # (in gazebo w/ the waffle_pi it's /camera/rgb/image_raw)
        image_topic = rospy.get_param('~image_topic', '/raspicam_node/image')
        color = rospy.get_param('~color', 'red').lower()
        self.show_debug = rospy.get_param('~show_debug', False)

        if color not in COLOR_RANGES:
            rospy.logwarn("don't know color '%s', falling back to red" % color)
            color = 'red'
        self.ranges = COLOR_RANGES[color]

        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # queue_size=1 + big buff_size so we always process the newest frame and don't lag behind
        rospy.Subscriber(image_topic, Image, self.image_cb, queue_size=1, buff_size=2 ** 24)

        # latest detection as one tuple (err, area_frac, time) so the control loop
        # never reads half of an old detection and half of a new one
        self.target = None

        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("tracking %s on %s" % (color, image_topic))

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logwarn_throttle(2.0, "cv_bridge error: %s" % e)
            return

        h, w = frame.shape[:2]
        mask = build_mask(frame, self.ranges)
        result = find_target(mask)

        if result is not None:
            cx, cy, area, contour = result
            # normalized horizontal error: -1 = far left edge, 0 = dead center, +1 = far right edge
            err = (cx - w / 2.0) / (w / 2.0)
            self.target = (err, area / float(w * h), rospy.get_time())

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

        if target is not None and rospy.get_time() - target[2] < TARGET_TIMEOUT:
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

    def shutdown(self):
        try:
            self.pub.publish(Twist())
        except Exception:
            pass
        if self.show_debug:
            cv2.destroyAllWindows()

    def run(self):
        # control runs on its own 10 hz loop instead of inside the image callback,
        # that way the robot still goes into search mode even if frames stop coming in
        rate = rospy.Rate(10)
        was_tracking = None
        while not rospy.is_shutdown():
            twist, tracking = self.compute_cmd()
            if tracking != was_tracking:
                rospy.loginfo("target found, tracking" if tracking else "no target, searching")
                was_tracking = tracking
            self.pub.publish(twist)
            rate.sleep()


if __name__ == '__main__':
    try:
        VisionTracker().run()
    except rospy.ROSInterruptException:
        pass
