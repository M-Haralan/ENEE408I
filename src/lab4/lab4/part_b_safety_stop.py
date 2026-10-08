#!/usr/bin/env python3
# enee408i lab 4 - part b: same keyboard teleop as part a but w/ a lidar safety stop on top
# w/s = speed up / slow down, a/d = turn left / right, space = stop, q = quit
#
# if something shows up within SAFETY_DIST in front of the robot, linear vel gets forced to 0
# no matter what you're pressing. turning still works so you can rotate away from it

import sys
import math
import select
import termios
import tty

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

# burger limits, if you're on a waffle change these to 0.26 / 1.82
MAX_LIN_VEL = 0.22
MAX_ANG_VEL = 2.84

LIN_STEP = 0.01
ANG_STEP = 0.1

# safety stuff
SAFETY_DIST = 0.3                    # meters, anything closer than this in front = stop
FRONT_HALF_ANGLE = math.radians(20)  # we look at a +-20 deg cone straight ahead
SCAN_TIMEOUT = 0.5                   # sec, if /scan goes quiet longer than this we play it safe and block
# the lab says to override forward AND backward when blocked. flip this to False if you
# want to still be able to back away from whatever is in front of you
BLOCK_REVERSE_TOO = True

HELP_MSG = """
control the turtlebot3 (w/ lidar safety stop)
---------------------------------------------
        w
   a    s    d

w/s   : increase/decrease linear vel  (max %.2f m/s)
a/d   : increase/decrease angular vel (max %.2f rad/s)
space : force stop
q     : quit

linear vel gets locked to 0 if anything is within %.2f m in front
""" % (MAX_LIN_VEL, MAX_ANG_VEL, SAFETY_DIST)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def get_key():
    # non-blocking read, gives back '' if nothing was pressed
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        return sys.stdin.read(1)
    return ''


class SafeTeleop:
    def __init__(self):
        rospy.init_node('turtlebot3_safe_teleop', anonymous=True)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # queue_size=1 so we always act on the newest scan, not some old one sitting in a queue
        rospy.Subscriber('/scan', LaserScan, self.scan_cb, queue_size=1)

        self.lin_vel = 0.0   # what the keyboard is asking for
        self.ang_vel = 0.0
        self.front_dist = float('inf')
        self.last_scan_time = None
        self.was_blocked = False

        rospy.on_shutdown(self.stop_robot)

    def scan_cb(self, msg):
        # on the turtlebot3 lds, angle_min is 0 so index 0 is straight ahead and angles go ccw.
        # going off the actual angle instead of hardcoding indices so this still works
        # in gazebo or if the scan is set up a little different
        closest = float('inf')
        for i, r in enumerate(msg.ranges):
            # the lds gives 0.0 (sometimes inf/nan) when it doesn't get a return, toss those.
            # nan fails every comparison so it gets skipped here too
            # heads up: anything closer than range_min (~0.12 m) also reads as 0, but
            # we should be stopped way before it gets that close
            if not (msg.range_min <= r <= msg.range_max):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            # wrap to [-pi, pi] so the cone check works no matter how the angles are laid out
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if abs(angle) <= FRONT_HALF_ANGLE and r < closest:
                closest = r

        self.front_dist = closest
        self.last_scan_time = rospy.get_time()

    def obstacle_ahead(self):
        # no scan yet or the scan went stale = we're driving blind, so treat it as blocked
        if self.last_scan_time is None or rospy.get_time() - self.last_scan_time > SCAN_TIMEOUT:
            rospy.logwarn_throttle(2.0, "no recent /scan data, holding linear vel at 0 to be safe")
            return True
        return self.front_dist < SAFETY_DIST

    def apply_key(self, key):
        key = key.lower()
        if key == 'w':
            self.lin_vel = clamp(round(self.lin_vel + LIN_STEP, 2), -MAX_LIN_VEL, MAX_LIN_VEL)
        elif key == 's':
            self.lin_vel = clamp(round(self.lin_vel - LIN_STEP, 2), -MAX_LIN_VEL, MAX_LIN_VEL)
        elif key == 'a':
            self.ang_vel = clamp(round(self.ang_vel + ANG_STEP, 2), -MAX_ANG_VEL, MAX_ANG_VEL)
        elif key == 'd':
            self.ang_vel = clamp(round(self.ang_vel - ANG_STEP, 2), -MAX_ANG_VEL, MAX_ANG_VEL)
        elif key == ' ':
            self.lin_vel = 0.0
            self.ang_vel = 0.0
        else:
            return False
        return True

    def safe_twist(self):
        # takes what the keyboard wants and clamps it down if there's something in the way
        twist = Twist()
        twist.angular.z = self.ang_vel
        twist.linear.x = self.lin_vel

        blocked = self.obstacle_ahead()
        if blocked and (self.lin_vel > 0 or (BLOCK_REVERSE_TOO and self.lin_vel < 0)):
            twist.linear.x = 0.0
            # also zero out the commanded speed, otherwise the second the obstacle moves
            # the robot would just take off again at whatever speed you had before
            self.lin_vel = 0.0

        # only print when we go into / out of the blocked state so it doesn't spam
        if blocked and not self.was_blocked:
            if self.front_dist < float('inf'):
                print("obstacle %.2f m ahead, linear vel locked to 0" % self.front_dist)
            else:
                print("no lidar data, linear vel locked to 0")
        elif not blocked and self.was_blocked:
            print("path clear, linear control is back")
        self.was_blocked = blocked

        return twist

    def stop_robot(self):
        try:
            self.pub.publish(Twist())
        except Exception:
            pass

    def run(self):
        rate = rospy.Rate(10)  # 10 hz
        old_settings = termios.tcgetattr(sys.stdin)
        print(HELP_MSG)

        try:
            tty.setcbreak(sys.stdin.fileno())

            while not rospy.is_shutdown():
                # drain all the waiting keys so held-down keys don't lag behind
                key = get_key()
                changed = False
                while key:
                    if key.lower() == 'q':
                        self.stop_robot()
                        return
                    if self.apply_key(key):
                        changed = True
                    key = get_key()

                twist = self.safe_twist()
                if changed:
                    print("linear: %.2f m/s   angular: %.2f rad/s   (front: %.2f m)"
                          % (twist.linear.x, twist.angular.z, self.front_dist))

                self.pub.publish(twist)
                rate.sleep()
        finally:
            self.stop_robot()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == '__main__':
    try:
        SafeTeleop().run()
    except rospy.ROSInterruptException:
        pass
