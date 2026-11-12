"""Simulation-only TurtleBot3 D* Lite navigation and conservative frontier exploration.

Run alongside Cartographer and optionally dstar_cartographer.py (read-only monitor).
One controller must own /cmd_vel. Ctrl+C stops the robot. Use only in simulation.
"""
import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

from dstar_cartographer import DStarLite, find_frontiers, render_active_path


class DStarNavigation(Node):
    # Robot center clearance on the occupancy grid; tune for your robot/world.
    CLEARANCE_M = 0.20  # Confirmed occupied cells: keep full robot clearance.
    UNKNOWN_CLEARANCE_M = 0.10  # Unknown cells: don't inflate as if solid walls.
    START_BLOCKED_RETRY_SECONDS = 2.0
    FRONT_STOP_M = 0.31
    REPLAN_SECONDS = 1.0
    SEGMENT_METERS = 0.55
    RETRY_SECONDS = 6.0
    MAX_RECOVERY_ATTEMPTS = 12
    PROGRESS_TIMEOUT = 16.0
    PROGRESS_METERS = 0.07
    WAYPOINT_TOLERANCE = 0.11
    FRONTIER_TOLERANCE = 0.16
    MAX_LINEAR = 0.08
    MAX_ANGULAR = 0.35

    def __init__(self):
        super().__init__('dstar_navigation')
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        # Different RViz versions/configurations publish on either topic.
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/dstar_active_path', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_msg = None
        self.map_at = 0.0
        self.scan_at = 0.0
        self.front_distance = None
        self.goal_xy = None
        self.mode = None
        self.target_xy = None
        self.waypoints = []
        self.path_index = 0
        self.plan_at = 0.0
        self.last_notice = None
        self.best_distance = math.inf
        self.progress_at = time.monotonic()
        self.failed_xy = []  # World coordinates survive occupancy-grid origin changes.
        self.reached_xy = []
        self.recovery_count = 0
        self.waiting_for_map_change = False
        self.waiting_signature = None
        self.waiting_since = 0.0
        self.last_display = 0.0
        self.last_map_revision = None
        self.last_pose_xy = None
        self.last_motion_at = time.monotonic()
        self.blocked_scan_count = 0
        self.last_heading = None
        self.last_heading_at = time.monotonic()
        self.pause_until = 0.0
        self.retry_rounds = 0
        self.last_start_blocked_notice = 0.0
        self.max_retry_rounds = 3
        self.timer = self.create_timer(0.2, self.control_step)
        print('\n--- D* LITE NAVIGATION + FRONTIER EXPLORATION (REVISED) ---')
        print('RViz: 2D Goal Pose. /goal_pose and /move_base_simple/goal supported.')
        print('Only this node should publish /cmd_vel. No automatic 360-degree scan.')
        print('Goal-directed exploration: prefer visible progress, then safe detours around walls.')

    def stop(self):
        if rclpy.ok():
            self.cmd_pub.publish(Twist())

    def notice(self, message):
        if self.last_notice != message:
            print(message, flush=True)
            self.last_notice = message

    def publish_path(self, pose=None):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        points = ([] if pose is None else [pose[:2]]) + self.waypoints[self.path_index:]
        for x, y in points:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.path_pub.publish(msg)

    def reset_path(self):
        self.mode = None
        self.target_xy = None
        self.waypoints = []
        self.path_index = 0
        self.plan_at = 0.0
        self.best_distance = math.inf
        self.progress_at = time.monotonic()
        self.publish_path()

    def goal_callback(self, msg):
        if msg.header.frame_id.lstrip('/') != 'map':
            self.notice('Goal rejected: RViz Fixed Frame must be map.')
            return
        xy = (msg.pose.position.x, msg.pose.position.y)
        if not all(math.isfinite(v) for v in xy):
            self.notice('Goal rejected: non-finite coordinates.')
            return
        # Avoid duplicate delivery if two goal publishers are bridged.
        if self.goal_xy is not None and math.dist(self.goal_xy, xy) < 0.001:
            self.notice('Goal reselected: retrying from current position.')
        else:
            print('\nNew goal (meters):', tuple(round(v, 2) for v in xy), flush=True)
        self.goal_xy = xy
        self.failed_xy.clear()
        self.reached_xy.clear()
        self.recovery_count = 0
        self.waiting_for_map_change = False
        self.pause_until = 0.0
        self.retry_rounds = 0
        self.last_pose_xy = None
        self.last_motion_at = time.monotonic()
        self.blocked_scan_count = 0
        self.last_heading = None
        self.last_heading_at = time.monotonic()
        self.reset_path()
        self.stop()
        self.last_notice = None

    def map_callback(self, msg):
        if msg.info.width and msg.info.height and msg.info.resolution > 0:
            self.map_msg = msg
            self.map_at = time.monotonic()

    def scan_callback(self, msg):
        # Positive infinity generally means no return within sensor range,
        # not an obstacle. Treat that as max range rather than invalid.
        forward = []
        for i, reading in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if abs(angle) > math.radians(25):
                continue
            if math.isinf(reading) and reading > 0:
                forward.append(msg.range_max)
            elif math.isfinite(reading) and msg.range_min <= reading <= msg.range_max:
                forward.append(reading)
        self.front_distance = min(forward) if forward else None
        self.scan_at = time.monotonic()
        if self.front_distance is not None and self.front_distance < self.FRONT_STOP_M:
            self.blocked_scan_count = min(10, self.blocked_scan_count + 1)
        else:
            self.blocked_scan_count = 0

    def pose(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return None
        p, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y),
                         1 - 2 * (q.y*q.y + q.z*q.z))
        return (p.x, p.y, yaw)

    @staticmethod
    def wrap(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def cell_of(xy, msg):
        info = msg.info
        return (math.floor((xy[0] - info.origin.position.x) / info.resolution),
                math.floor((xy[1] - info.origin.position.y) / info.resolution))

    @staticmethod
    def world_of(cell, msg):
        info = msg.info
        return (info.origin.position.x + (cell[0] + 0.5)*info.resolution,
                info.origin.position.y + (cell[1] + 0.5)*info.resolution)

    @staticmethod
    def inside(cell, width, height):
        return 0 <= cell[0] < width and 0 <= cell[1] < height

    def make_grids(self, msg):
        w, h = msg.info.width, msg.info.height
        raw = [[-1 if msg.data[y*w+x] == -1 else
                (1 if msg.data[y*w+x] >= 50 else 0)
                for x in range(w)] for y in range(h)]
        safe = [[0]*w for _ in range(h)]
        # Do not treat every unknown cell as a confirmed wall. The original
        # implementation inflated both by 20 cm, so the robot's own position
        # became "blocked" as the SLAM frontier advanced around it.
        # Unknown cells themselves remain forbidden for planning.
        occupied_radius = max(1, math.ceil(self.CLEARANCE_M/msg.info.resolution))
        unknown_radius = max(1, math.ceil(self.UNKNOWN_CLEARANCE_M/msg.info.resolution))
        occupied_offsets = [(dx, dy) for dy in range(-occupied_radius, occupied_radius+1)
                            for dx in range(-occupied_radius, occupied_radius+1)
                            if dx*dx+dy*dy <= occupied_radius*occupied_radius]
        unknown_offsets = [(dx, dy) for dy in range(-unknown_radius, unknown_radius+1)
                           for dx in range(-unknown_radius, unknown_radius+1)
                           if dx*dx+dy*dy <= unknown_radius*unknown_radius]
        for y in range(h):
            for x in range(w):
                if raw[y][x] != 0:
                    offsets = occupied_offsets if raw[y][x] == 1 else unknown_offsets
                    for dx, dy in offsets:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < w and 0 <= ny < h:
                            safe[ny][nx] = 1
        return raw, safe

    @staticmethod
    def distances(safe, start):
        h, w = len(safe), len(safe[0])
        if not (0 <= start[0] < w and 0 <= start[1] < h):
            return {}
        if safe[start[1]][start[0]]:
            return {}
        result = {start: 0}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                n = (x+dx, y+dy)
                if (0 <= n[0] < w and 0 <= n[1] < h
                        and safe[n[1]][n[0]] == 0 and n not in result):
                    result[n] = result[(x,y)] + 1
                    queue.append(n)
        return result

    @staticmethod
    def line_clear(a, b, safe):
        h, w = len(safe), len(safe[0])
        steps = max(1, 3*max(abs(a[0]-b[0]), abs(a[1]-b[1])))
        for i in range(steps+1):
            f = i/steps
            x = round(a[0] + (b[0]-a[0])*f)
            y = round(a[1] + (b[1]-a[1])*f)
            if not (0 <= x < w and 0 <= y < h) or safe[y][x]:
                return False
        return True

    def simplify(self, cells, safe, resolution):
        if len(cells) < 3:
            return cells
        output = [cells[0]]
        index = 0
        limit = max(2, int(0.55/resolution))
        while index < len(cells)-1:
            last = index+1
            for i in range(index+2, min(len(cells), index+limit+1)):
                if self.line_clear(cells[index], cells[i], safe):
                    last = i
                else:
                    break
            output.append(cells[last])
            index = last
        return output

    def excluded(self, xy):
        return any(math.dist(xy, old) < 0.38 for old in self.failed_xy) or any(
            math.dist(xy, old) < 0.27 for old in self.reached_xy)

    def choose_frontier(self, raw, safe, start, goal, msg, reachable):
        """Select a reachable, goal-directed observation point, not a map-coverage tour.

        Unknown cells are never traversable. Sideways detours remain possible
        when the direct approach is blocked by a wall.
        """
        frontiers = find_frontiers(raw)
        if not frontiers:
            return None
        res = msg.info.resolution
        radius = max(1, math.ceil(0.45 / res))
        max_travel = math.ceil(2.5 / res)
        candidates = set()
        for fx, fy in frontiers:
            for dy in range(-radius, radius + 1, 2):
                for dx in range(-radius, radius + 1, 2):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    c = (fx + dx, fy + dy)
                    if c not in reachable or not (3 <= reachable[c] <= max_travel):
                        continue
                    if self.excluded(self.world_of(c, msg)):
                        continue
                    candidates.add(c)
        if not candidates:
            return None

        # The direct goal direction gets priority; modest lateral travel is
        # permitted to see around obstacles. Score in meters, not grid indices.
        current_goal_dist = math.dist(self.world_of(start, msg), self.goal_xy)
        def score(c):
            xy = self.world_of(c, msg)
            goal_dist = math.dist(xy, self.goal_xy)
            travel = reachable[c] * res
            # Discourage lateral/backtracking, but never forbid it: going
            # around a wall may initially increase distance to the goal.
            detour = max(0.0, goal_dist - current_goal_dist)
            return goal_dist + 0.55 * travel + 0.4 * detour
        return min(candidates, key=score)

    def abandon_target(self, reason, pose, failed=True):
        """Skip a blocked staging point and immediately try another one."""
        self.stop()
        self.notice(reason)
        if self.target_xy is not None and self.mode == 'FRONTIER':
            if failed:
                self.failed_xy.append(self.target_xy)
            else:
                self.reached_xy.append(self.target_xy)
        self.reset_path()
        self.recovery_count += 1
        # Pause and reassess using a fresh map rather than requiring the user
        # to resend the same goal. Never drive while paused.
        if self.recovery_count >= self.MAX_RECOVERY_ATTEMPTS:
            self.pause_for_retry('Frontier attempts exhausted; pausing before a fresh-map retry.')

    def pause_for_retry(self, reason):
        self.stop()
        self.reset_path()
        self.waiting_for_map_change = True
        self.waiting_since = time.monotonic()
        self.pause_until = self.waiting_since + self.RETRY_SECONDS
        self.notice(reason)

    def resume_after_pause(self):
        # Retain the original goal. Clear temporary exclusions only after a
        # cooldown so previously rejected frontiers can be reconsidered.
        self.waiting_for_map_change = False
        self.failed_xy.clear()
        self.reached_xy.clear()
        self.recovery_count = 0
        self.retry_rounds += 1
        self.reset_path()
        self.notice('Retrying original goal with the latest Cartographer map.')

    def plan(self, pose):
        msg = self.map_msg
        start = self.cell_of(pose[:2], msg)
        goal = self.cell_of(self.goal_xy, msg)
        w, h = msg.info.width, msg.info.height
        raw, safe = self.make_grids(msg)
        if not self.inside(start, w, h):
            self.notice('STOP: robot is outside the current Cartographer map.')
            return False
        if safe[start[1]][start[0]]:
            # This may be temporary map inflation near a SLAM frontier, but
            # it may also indicate a real wall. Never clear the start cell
            # artificially or drive through it. Wait for a refreshed map.
            reason = ('occupied' if raw[start[1]][start[0]] == 1 else
                      'unknown' if raw[start[1]][start[0]] == -1 else
                      'too close to occupied/unknown cells')
            self.notice('STOP: robot center lacks mapped clearance (' + reason +
                        '). Waiting for map update; no unsafe path attempted.')
            self.plan_at = time.monotonic()
            return False
        reachable = self.distances(safe, start)
        if not reachable:
            self.notice('STOP: no known free cells reachable from robot.')
            return False
        if goal in reachable:
            target = goal
            mode = 'GOAL'
        else:
            mode = 'FRONTIER'
            target = None
            if self.mode == 'FRONTIER' and self.target_xy is not None:
                old = self.cell_of(self.target_xy, msg)
                if old in reachable and not self.excluded(self.target_xy):
                    target = old
            if target is None:
                target = self.choose_frontier(raw, safe, start, goal, msg, reachable)
            if target is None:
                self.pause_for_retry('No safe untried frontier; waiting for a fresh map before retrying.')
                return False
        planner = DStarLite(safe, start, target)
        planner.compute_shortest_path()
        cells = planner.get_path()
        if not cells:
            if mode == 'FRONTIER':
                self.failed_xy.append(self.world_of(target, msg))
            self.notice('STOP: D* Lite could not find a safe route to selected target.')
            return False
        target_xy = self.world_of(target, msg)
        changed = mode != self.mode or self.target_xy is None or math.dist(target_xy, self.target_xy) > 0.10
        if changed:
            self.best_distance = math.inf
            self.progress_at = time.monotonic()
            print('Mode:', mode, '| Target:', tuple(round(v, 2) for v in target_xy),
                  '| D* path:', len(cells)-1, 'cells', flush=True)
        self.mode = mode
        self.target_xy = target_xy
        simplified = self.simplify(cells, safe, msg.info.resolution)
        # Execute only a short prefix of the new route. The next planning
        # cycle uses the current pose and latest occupancy grid.
        limited = [simplified[0]]
        traveled = 0.0
        for a, b in zip(simplified, simplified[1:]):
            segment = math.dist(self.world_of(a, msg), self.world_of(b, msg))
            if traveled + segment > self.SEGMENT_METERS and len(limited) > 1:
                break
            limited.append(b)
            traveled += segment
            if traveled >= self.SEGMENT_METERS:
                break
        self.waypoints = [self.world_of(c, msg) for c in limited[1:]]
        if mode == 'GOAL' and limited[-1] == target and self.line_clear(target, goal, safe):
            # Exact RViz goal, not just the center of its occupancy-grid cell.
            if not self.waypoints or math.dist(self.waypoints[-1], self.goal_xy) > 0.02:
                self.waypoints.append(self.goal_xy)
        self.path_index = 0
        self.plan_at = time.monotonic()
        self.last_notice = None
        self.publish_path(pose)
        if changed or self.plan_at-self.last_display > 10:
            points = [pose[:2]] + self.waypoints
            render_active_path(msg, points, robot_xy=pose[:2], goal_xy=self.goal_xy,
                               title='ACTIVE D* LITE NAVIGATION PATH')
            self.last_display = self.plan_at
        return True

    def control_step(self):
        if self.goal_xy is None:
            self.stop()
            return
        now = time.monotonic()
        if self.map_msg is None or now-self.map_at > 3.0:
            self.stop()
            self.notice('STOP: waiting for a fresh /map.')
            return
        if self.scan_at == 0 or now-self.scan_at > 3.0:
            self.stop()
            self.notice('STOP: waiting for fresh /scan.')
            return
        pose = self.pose()
        if pose is None:
            self.stop()
            self.notice('STOP: waiting for map -> base_link TF.')
            return
        x, y, yaw = pose
        if math.dist((x,y), self.goal_xy) < 0.12:
            self.stop()
            print('Goal reached. Robot stopped.', flush=True)
            self.goal_xy = None
            self.reset_path()
            return
        if self.waiting_for_map_change:
            self.stop()
            if now < self.pause_until:
                return
            if self.retry_rounds >= self.max_retry_rounds:
                self.notice('STOP: repeated retries did not find a safe route. Select a new goal or inspect the map.')
                return
            self.resume_after_pause()
            return
        if (now-self.plan_at > self.REPLAN_SECONDS or
                (self.waypoints and self.path_index >= len(self.waypoints))):
            self.stop()
            if not self.plan(pose):
                self.plan_at = now
                self.waypoints = []
                self.publish_path()
                return
        # Measure actual translation, not straight-line distance to the
        # frontier: moving around a wall may temporarily increase that distance.
        if self.last_pose_xy is None:
            self.last_pose_xy = (x, y)
            self.last_motion_at = now
        elif math.dist((x, y), self.last_pose_xy) >= self.PROGRESS_METERS:
            self.last_pose_xy = (x, y)
            self.last_motion_at = now
            self.retry_rounds = 0
        # Turning in place toward a legitimate waypoint is not a navigation
        # stall. Only declare a stall if neither position nor heading changes.
        if self.last_heading is None:
            self.last_heading = yaw
            self.last_heading_at = now
        elif abs(self.wrap(yaw - self.last_heading)) > 0.12:
            self.last_heading = yaw
            self.last_heading_at = now
        if self.mode == 'FRONTIER' and self.target_xy is not None:
            distance_to_target = math.dist((x,y), self.target_xy)
            if distance_to_target < self.FRONTIER_TOLERANCE:
                self.abandon_target('Frontier observation point reached; seeking next frontier.', pose, failed=False)
                return
            if now-max(self.last_motion_at, self.last_heading_at) > self.PROGRESS_TIMEOUT:
                self.last_motion_at = now
                self.abandon_target('Frontier stalled; retrying a different goal-directed observation point.', pose)
                return
        while self.path_index < len(self.waypoints):
            wx, wy = self.waypoints[self.path_index]
            if math.dist((wx,wy), (x,y)) >= self.WAYPOINT_TOLERANCE:
                break
            self.path_index += 1
        if self.path_index >= len(self.waypoints):
            # A completed short segment is not a completed frontier. Keep the
            # current frontier and replan from the robot's updated position.
            self.waypoints = []
            self.path_index = 0
            self.plan_at = 0.0
            self.publish_path()
            self.stop()
            return
        wx, wy = self.waypoints[self.path_index]
        distance = math.hypot(wx-x, wy-y)
        error = self.wrap(math.atan2(wy-y, wx-x)-yaw)
        cmd = Twist()
        cmd.angular.z = max(-self.MAX_ANGULAR, min(self.MAX_ANGULAR, 1.3*error))
        # Turn toward the waypoint before driving. Never rotate 360 degrees.
        if abs(error) < 0.35:
            if self.front_distance is None:
                self.stop()
                self.notice('STOP: no valid forward LiDAR readings.')
                return
            if self.front_distance < self.FRONT_STOP_M:
                # Stop immediately even on one scan. Do not mark a frontier
                # failed until the obstruction persists across several scans.
                self.stop()
                if self.blocked_scan_count >= 4:
                    if self.mode == 'FRONTIER':
                        self.abandon_target('Persistent obstacle ahead; selecting another safe frontier.', pose)
                    else:
                        self.reset_path()
                        self.notice('Obstacle on goal route; replanning from updated map.')
                return
            cmd.linear.x = min(self.MAX_LINEAR, 0.5*distance)
            cmd.linear.x *= max(0.3, 1.0-abs(error)/0.6)
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = DStarNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
