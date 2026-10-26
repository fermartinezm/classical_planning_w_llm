from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortile22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles with their goal colors. It considers the movement required for robots to reach adjacent tiles of unpainted tiles, necessary color changes, and the paint actions.

    # Assumptions:
    - Robots can move freely between clear tiles.
    - Color changes are only needed if the robot's current color doesn't match the target tile's required color.
    - Each unpainted tile requires at least one paint action and possibly movement and color changes.

    # Heuristic Initialization
    - Extract adjacency relations between tiles from static facts.
    - Determine goal colors for each tile from the problem's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile, check if it's already painted the correct color. If not, proceed.
    2. For each unpainted tile, find its adjacent tiles (from static data).
    3. For each robot, compute the minimal path to each adjacent tile using BFS, considering current clear tiles.
    4. If a path exists, calculate the total cost (movement + color change + paint).
    5. If no path is found, use Manhattan distance as a fallback.
    6. Sum the minimal costs for all unpainted tiles.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static data and goals."""
        self.goal_paint = {}  # Maps each tile to its goal color.
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_paint[tile] = color

        # Build adjacency list from static facts.
        self.adjacent_tiles = defaultdict(list)
        for fact in task.static:
            if not (fact.startswith('(up ') or fact.startswith('(down ') or
                    fact.startswith('(left ') or fact.startswith('(right ')):
                continue
            parts = fact[1:-1].split()
            direction, tile_a, tile_b = parts
            self.adjacent_tiles[tile_b].append(tile_a)  # tile_b can move to tile_a via direction

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state
        clear_tiles = set()
        robot_positions = {}
        robot_colors = {}
        current_paint = {}

        # Extract current state information.
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'clear':
                clear_tiles.add(parts[1])
            elif parts[0] == 'robot-at':
                robot = parts[1]
                tile = parts[2]
                robot_positions[robot] = tile
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color
            elif parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                current_paint[tile] = color

        # Precompute BFS distances for each robot.
        robot_distances = {}
        for robot, start_tile in robot_positions.items():
            visited = {}
            queue = deque([(start_tile, 0)])
            visited[start_tile] = 0
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.adjacent_tiles.get(current, []):
                    if neighbor in clear_tiles and neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            robot_distances[robot] = visited

        total_cost = 0

        # Process each tile in the goal.
        for tile, target_color in self.goal_paint.items():
            current_color = current_paint.get(tile)
            if current_color == target_color:
                continue  # Already painted correctly.

            adjacent_tiles = self.adjacent_tiles.get(tile, [])
            min_cost = float('inf')

            for adj_tile in adjacent_tiles:
                if adj_tile not in clear_tiles:
                    continue  # Adjacent tile is not clear, can't move there.

                for robot in robot_positions:
                    start_tile = robot_positions[robot]
                    distances = robot_distances[robot]
                    if adj_tile in distances:
                        movement_cost = distances[adj_tile]
                    else:
                        # Fallback to Manhattan distance if no path found.
                        rx, ry = self.parse_tile_coords(start_tile)
                        ax, ay = self.parse_tile_coords(adj_tile)
                        movement_cost = abs(rx - ax) + abs(ry - ay)

                    current_robot_color = robot_colors.get(robot)
                    color_change = 0 if current_robot_color == target_color else 1
                    total_tile_cost = movement_cost + color_change + 1  # +1 for paint

                    if total_tile_cost < min_cost:
                        min_cost = total_tile_cost

            if min_cost != float('inf'):
                total_cost += min_cost
            else:
                # Assign a high cost if no path found (unlikely in solvable states)
                total_cost += 1000

        return total_cost

    @staticmethod
    def parse_tile_coords(tile_name):
        """Extract x and y coordinates from a tile name like 'tile_3_2'."""
        parts = tile_name.split('_')
        x = int(parts[1])
        y = int(parts[2])
        return x, y
