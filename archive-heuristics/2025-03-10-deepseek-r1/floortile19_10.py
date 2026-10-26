from collections import deque
from collections import defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortile19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering the minimal distance robots need to move, color changes required, and the paint actions.

    # Assumptions:
    - Robots can move freely between adjacent tiles as per the grid structure.
    - Changing color takes one action and can be done at any time.
    - Each paint action requires the robot to be adjacent to the target tile and have the correct color.

    # Heuristic Initialization
    - Extract adjacency relationships between tiles from static facts.
    - Identify goal tiles and their required colors.
    - Store available colors for color change validity.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet painted:
        a. Find all adjacent tiles (possible positions to paint from).
        b. For each robot, compute the minimal move actions to reach any adjacent tile.
        c. Determine if a color change is needed for the robot.
        d. Calculate the total cost (moves + color change + paint) for each robot.
        e. Take the minimal cost across all robots for this tile.
    2. Sum the minimal costs for all goal tiles to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static information and goals."""
        self.adjacency = defaultdict(list)
        self.available_colors = set()
        self.goal_painted = []

        # Parse static facts to build adjacency list and available colors
        for fact in task.static:
            parts = fact[1:-1].split()
            if not parts:
                continue
            if parts[0] in ['up', 'down', 'left', 'right']:
                tile_from = parts[2]
                tile_to = parts[1]
                self.adjacency[tile_from].append(tile_to)
            elif parts[0] == 'available-color':
                self.available_colors.add(parts[1])

        # Parse goal conditions to extract required painted tiles
        for goal in task.goals:
            if goal.startswith('(painted'):
                parts = goal[1:-1].split()
                tile = parts[1]
                color = parts[2]
                self.goal_painted.append((tile, color))

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state
        robots = {}
        # Extract robot positions and colors from the current state
        for fact in state:
            if fact.startswith('(robot-at'):
                parts = fact[1:-1].split()
                robot = parts[1]
                tile = parts[2]
                robots[robot] = {'pos': tile, 'color': None}
            elif fact.startswith('(robot-has'):
                parts = fact[1:-1].split()
                robot = parts[1]
                color = parts[2]
                if robot in robots:
                    robots[robot]['color'] = color

        # Precompute BFS distances for each robot's current position
        robot_distances = {}
        for robot in robots:
            start = robots[robot]['pos']
            distances = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.adjacency.get(current, []):
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            robot_distances[robot] = distances

        total_cost = 0
        # Check each goal tile
        for tile, color in self.goal_painted:
            # Skip if already painted correctly
            if f'(painted {tile} {color})' in state:
                continue

            min_cost = float('inf')
            # Check each robot's cost to paint this tile
            for robot in robots:
                r_pos = robots[robot]['pos']
                r_color = robots[robot]['color']
                adj_tiles = self.adjacency.get(tile, [])
                if not adj_tiles:
                    continue  # Tile has no adjacent tiles (invalid state)
                # Find minimal distance to any adjacent tile
                distances = robot_distances[robot]
                min_dist = min((distances.get(adj, float('inf')) for adj in adj_tiles), default=float('inf'))
                if min_dist == float('inf'):
                    continue  # Robot can't reach any adjacent tile
                # Check color change cost
                color_cost = 0 if r_color == color else 1
                cost = min_dist + color_cost + 1  # +1 for paint action
                if cost < min_cost:
                    min_cost = cost
            if min_cost == float('inf'):
                # Assume solvable, assign a high cost to avoid zero division
                min_cost = 0 if total_cost == 0 else total_cost * 2
            total_cost += min_cost

        return total_cost
