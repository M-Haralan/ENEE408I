#!/usr/bin/env python3
import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

MAX_LIN_VEL = 0.22
MAX_ANG_VEL = 2.84
LIN_STEP = 0.01
ANG_STEP = 0.1

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
        super().__init__('part_a_teleop')
        # self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        self.create_timer(0.05, self.poll_keys)

    # def scan_callback(self, msg: LaserScan):
    #     if msg.

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