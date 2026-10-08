import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import os
import select
import sys

class TurtleBotAutonomousMove(Node):
    def __init__(self):
        super().__init__('turtlebot3_autonomous_move')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_timer(0.05, self.timer_callback)

    


def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotAutonomousMove()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot before exiting
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()