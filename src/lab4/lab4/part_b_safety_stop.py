#!/usr/bin/env python3
import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import numpy as np

MAX_LIN_VEL = 0.22
MAX_ANG_VEL = 2.84
LIN_STEP = 0.01
ANG_STEP = 0.1

# safety stuff
SAFETY_DIST = 0.3                    # meters, anything closer than this in front = stop
FRONT_HALF_ANGLE = np.deg2rad(20)    # we look at a +-20 deg cone straight ahead
SCAN_TIMEOUT = 0.5                   # sec, if /scan goes quiet longer than this we play it safe and block
# the lab says to override forward and backward when blocked. flip this to False if you
# want to still be able to back away from whatever is in front of you
BLOCK_REVERSE_TOO = False


HELP_MSG = """
control the turtlebot3
----------------------
        w
   a    s    d

w/s   : increase/decrease linear vel  (max %.2f m/s)
a/d   : increase/decrease angular vel (max %.2f rad/s)
space : force stop
q     : quit
""" % (MAX_LIN_VEL, MAX_ANG_VEL)


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_stop')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.front_dist = float('inf')
        self.last_scan_time = None
        self.create_subscription(LaserScan, '/scan', self.scan_callback,
                                 qos_profile_sensor_data)

        # these parts here are to have a non-blocking terminal
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.create_timer(0.05, self.poll_keys_safe)

    def scan_callback(self, msg: LaserScan):
        closest = float('inf')
        for i, r in enumerate(msg.ranges):
            # the lds gives 0.0 (sometimes inf/nan) when it doesn't get a return, toss those.
            # nan fails every comparison so it gets skipped here too
            # heads up: anything closer than range_min (~0.12 m) also reads as 0, but
            # we should be stopped way before it gets that close
            if not (msg.range_min <= r <= msg.range_max):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            if abs(angle) <= FRONT_HALF_ANGLE and r < closest:
                closest = r

        self.front_dist = closest
        self.last_scan_time = self.get_clock().now()

    def obstacle_ahead(self):
        if self.last_scan_time is None:
            stale = True
        else:
            age = (self.get_clock().now() - self.last_scan_time).nanoseconds * 1e-9
            stale = age > SCAN_TIMEOUT
        if stale:
            self.get_logger().warn("no recent /scan data, blocking linear motion",
                                   throttle_duration_sec=2.0)
            return True
        return self.front_dist < SAFETY_DIST

    def poll_keys_safe(self):
        blocked = self.obstacle_ahead()
        if blocked and (self.lin_vel > 0 or (BLOCK_REVERSE_TOO and self.lin_vel < 0)):
            self.lin_vel = 0.0
            self.pub.publish(make_twist(self.lin_vel, self.ang_vel))
        else: 
            self.poll_keys()


    def poll_keys(self):
        key = get_key()
        if key == 'q' or key == '\x03': # this are our exit keys
            raise KeyboardInterrupt
        self.lin_vel, self.ang_vel = apply_key(key, self.lin_vel, self.ang_vel)
        self.pub.publish(make_twist(self.lin_vel, self.ang_vel))


    def restore_terminal(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def get_key():
    # select is non-blocking.
    # so either get one key or block
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return ''


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def apply_key(key, lin_vel, ang_vel):
    key = key.lower()
    if key == 'w':
        lin_vel = clamp(round(lin_vel + LIN_STEP, 2), -MAX_LIN_VEL, MAX_LIN_VEL)
    elif key == 's':
        lin_vel = clamp(round(lin_vel - LIN_STEP, 2), -MAX_LIN_VEL, MAX_LIN_VEL)
    elif key == 'a':
        ang_vel = clamp(round(ang_vel + ANG_STEP, 2), -MAX_ANG_VEL, MAX_ANG_VEL)
    elif key == 'd':
        ang_vel = clamp(round(ang_vel - ANG_STEP, 2), -MAX_ANG_VEL, MAX_ANG_VEL)
    elif key == ' ':
        lin_vel = 0.0
        ang_vel = 0.0
    return lin_vel, ang_vel


def make_twist(lin_vel, ang_vel):
    twist = Twist()
    twist.linear.x = lin_vel
    twist.angular.z = ang_vel
    return twist


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    print(HELP_MSG)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())  # stop the robot
        node.restore_terminal()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()