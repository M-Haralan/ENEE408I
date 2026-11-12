import math
import heapq
from collections import deque

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener


class DStarLite:
    """Four-connected D* Lite planner for a fixed occupancy-grid snapshot."""

    def __init__(self, grid, start, goal):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])
        self.start = start
        self.goal = goal
        self.km = 0
        self.g = {}
        self.rhs = {goal: 0}
        self.queue = []
        self.queued_keys = {}
        self.add_to_queue(goal)

    def neighbors(self, cell):
        x, y = cell
        for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield (nx, ny)

    def is_free(self, cell):
        x, y = cell
        return self.grid[y][x] == 0

    def cost(self, a, b):
        return 1 if self.is_free(a) and self.is_free(b) else math.inf

    def get_g(self, cell):
        return self.g.get(cell, math.inf)

    def get_rhs(self, cell):
        return self.rhs.get(cell, math.inf)

    @staticmethod
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def calculate_key(self, cell):
        value = min(self.get_g(cell), self.get_rhs(cell))
        return (value + self.heuristic(self.start, cell) + self.km, value)

    def add_to_queue(self, cell):
        key = self.calculate_key(cell)
        self.queued_keys[cell] = key
        heapq.heappush(self.queue, (key, cell))

    def clean_queue(self):
        while self.queue:
            key, cell = self.queue[0]
            if self.queued_keys.get(cell) == key:
                return
            heapq.heappop(self.queue)

    def update_vertex(self, cell):
        if cell != self.goal:
            self.rhs[cell] = min(
                (self.cost(cell, neighbor) + self.get_g(neighbor)
                 for neighbor in self.neighbors(cell)),
                default=math.inf
            )
        self.queued_keys.pop(cell, None)
        if self.get_g(cell) != self.get_rhs(cell):
            self.add_to_queue(cell)

    def compute_shortest_path(self):
        while True:
            self.clean_queue()
            top_key = self.queue[0][0] if self.queue else (math.inf, math.inf)
            if not (top_key < self.calculate_key(self.start) or
                    self.get_rhs(self.start) != self.get_g(self.start)):
                break
            if not self.queue:
                break

            old_key, cell = heapq.heappop(self.queue)
            self.queued_keys.pop(cell, None)
            new_key = self.calculate_key(cell)

            if old_key < new_key:
                self.add_to_queue(cell)
            elif self.get_g(cell) > self.get_rhs(cell):
                self.g[cell] = self.get_rhs(cell)
                for neighbor in self.neighbors(cell):
                    self.update_vertex(neighbor)
            else:
                self.g[cell] = math.inf
                self.update_vertex(cell)
                for neighbor in self.neighbors(cell):
                    self.update_vertex(neighbor)

    def get_path(self):
        if not self.is_free(self.start) or not self.is_free(self.goal):
            return []
        if self.get_g(self.start) == math.inf:
            return []

        path = [self.start]
        visited = {self.start}
        current = self.start

        while current != self.goal:
            options = [
                (self.cost(current, neighbor) + self.get_g(neighbor), neighbor)
                for neighbor in self.neighbors(current)
            ]
            options = [(cost, cell) for cost, cell in options
                       if cost < math.inf and cell not in visited]
            if not options:
                return []
            _, current = min(options)
            path.append(current)
            visited.add(current)
        return path


def neighbors(cell, width, height):
    x, y = cell
    for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            yield (nx, ny)


def find_frontiers(grid):
    """Free cells adjacent to at least one unknown cell."""
    height = len(grid)
    width = len(grid[0])
    frontiers = set()
    for y in range(height):
        for x in range(width):
            if grid[y][x] != 0:
                continue
            cell = (x, y)
            if any(grid[ny][nx] == -1
                   for nx, ny in neighbors(cell, width, height)):
                frontiers.add(cell)
    return frontiers


def reachable_distances(grid, start):
    """BFS distances through known free cells only."""
    height = len(grid)
    width = len(grid[0])
    distances = {start: 0}
    pending = deque([start])
    while pending:
        cell = pending.popleft()
        for neighbor in neighbors(cell, width, height):
            nx, ny = neighbor
            if grid[ny][nx] == 0 and neighbor not in distances:
                distances[neighbor] = distances[cell] + 1
                pending.append(neighbor)
    return distances


def render_active_path(msg, points, robot_xy=None, goal_xy=None, title='ACTIVE NAVIGATION PATH'):
    """Display the controller's actual waypoint path on the latest occupancy grid.

    This does NOT compute a second path. '*' interpolates between published
    waypoints for a readable terminal preview; RViz shows the exact Path.
    """
    width, height = msg.info.width, msg.info.height
    if not width or not height or not points:
        return
    resolution = msg.info.resolution
    ox, oy = msg.info.origin.position.x, msg.info.origin.position.y

    def cell(xy):
        return (math.floor((xy[0] - ox) / resolution),
                math.floor((xy[1] - oy) / resolution))

    def inside(c):
        return 0 <= c[0] < width and 0 <= c[1] < height

    # Show the *published* route, including the segments between waypoints.
    path_cells = set()
    for a, b in zip(points, points[1:]):
        steps = max(1, math.ceil(math.hypot(b[0]-a[0], b[1]-a[1]) / (resolution * 0.5)))
        for i in range(steps + 1):
            fraction = i / steps
            c = cell((a[0] + fraction*(b[0]-a[0]),
                      a[1] + fraction*(b[1]-a[1])))
            if inside(c):
                path_cells.add(c)
    if len(points) == 1 and inside(cell(points[0])):
        path_cells.add(cell(points[0]))

    robot = cell(robot_xy) if robot_xy is not None else cell(points[0])
    goal = cell(goal_xy) if goal_xy is not None else None
    target = cell(points[-1])
    scale = 3
    print('\n--- ' + title + ' ---')
    print('Route: /dstar_active_path | Waypoints:', len(points),
          '| Robot:', robot, '| Active target:', target)
    print('Legend: R robot, T route endpoint, G original goal, * active route, # obstacle, ? unknown')
    for top in range(height-1, -1, -scale):
        line = ''
        for left in range(0, width, scale):
            block = [(xx, yy)
                     for yy in range(max(0, top-scale+1), top+1)
                     for xx in range(left, min(left+scale, width))]
            if robot in block:
                symbol = 'R'
            elif target in block:
                symbol = 'T'
            elif goal is not None and goal in block:
                symbol = 'G'
            elif any(c in path_cells for c in block):
                symbol = '*'
            elif any(msg.data[y*width+x] >= 50 for x, y in block):
                symbol = '#'
            elif any(msg.data[y*width+x] == -1 for x, y in block):
                symbol = '?'
            else:
                symbol = '.'
            line += symbol
        print(line)
    if goal is not None and not inside(goal):
        print('Original goal is outside the current map:', goal)


class DStarCartographer(Node):
    """Read-only monitor of the route chosen by dstar_navigation.py."""

    def __init__(self):
        super().__init__('dstar_cartographer')
        from nav_msgs.msg import Path
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, QoSProfile(
            depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Path, '/dstar_active_path', self.path_callback, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.map_msg = None
        self.path_points = []
        self.goal_xy = None
        self.last_display_at = 0.0
        self.last_signature = None
        self.create_timer(1.0, self.display)
        print('\n--- D* LITE PATH MONITOR (READ ONLY) ---')
        print('Waiting for /map and /dstar_active_path from dstar_navigation.py ...')
        print('This node does not plan independently or publish /cmd_vel.')

    def map_callback(self, msg):
        self.map_msg = msg

    def path_callback(self, msg):
        self.path_points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if not self.path_points:
            self.last_signature = None

    def goal_callback(self, msg):
        if msg.header.frame_id.lstrip('/') == 'map':
            self.goal_xy = (msg.pose.position.x, msg.pose.position.y)

    def display(self):
        import time
        if self.map_msg is None or not self.path_points:
            return
        signature = (tuple((round(x, 2), round(y, 2)) for x, y in self.path_points),
                     self.map_msg.info.width, self.map_msg.info.height)
        now = time.monotonic()
        if signature == self.last_signature and now - self.last_display_at < 8.0:
            return
        # Limit terminal redraws while the robot is moving.
        if now - self.last_display_at < 5.0:
            return
        self.last_signature = signature
        self.last_display_at = now
        render_active_path(self.map_msg, self.path_points,
                           robot_xy=self.path_points[0], goal_xy=self.goal_xy,
                           title='ACTIVE PATH FROM NAVIGATION')


def main():
    rclpy.init()
    node = DStarCartographer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
