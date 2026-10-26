from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

class floortile5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles with their required colors. For each unpainted goal tile, it calculates the minimal cost considering the distance to move a robot to an adjacent tile, any required color change, and the paint action. The total heuristic is the sum of these minimal costs for all tiles.

    # Assumptions
    - The grid is structured such that Manhattan distance between tiles is a valid movement cost estimate.
    - Robots can change colors only if the required color is available.
    - Each paint action requires the robot to be on a tile adjacent to the target tile.
    - Robots can work in parallel, but the heuristic sums individual minimal costs.

    # Heuristic Initialization
    - Extracts adjacency relations between tiles from static facts.
    - Extracts available colors from static facts.
    - Extracts goal painted conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet painted correctly:
        a. Check if the required color is available. If not, add a high penalty.
        b. For each robot:
            i. Calculate the minimal movement steps to reach any adjacent tile of the goal.
            ii. Add 1 action for painting.
            iii. Add 1 action if the robot's current color doesn't match and the color is available.
        c. Take the minimal cost across all robots for this tile.
    2. Sum all minimal tile costs to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static information from the task."""
        self.static = task.static
        self.adjacent_tiles = defaultdict(list)
        self.available_colors = set()
        self.goal_painted = {}

        # Extract adjacency relations from static facts
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] in ['up', 'down', 'left', 'right']:
                target_tile = parts[1]
                adjacent_tile = parts[2]
                self.adjacent_tiles[target_tile].append(adjacent_tile)

        # Extract available colors from static facts
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'available-color':
                self.available_colors.add(parts[1])

        # Extract goal painted conditions from task goals
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_painted[tile] = color

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_painted = {}
        robots = {}

        # Parse current state to get painted tiles and robot statuses
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                current_painted[tile] = color
            elif parts[0] == 'robot-at':
                robot_name = parts[1]
                tile = parts[2]
                robots[robot_name] = {'tile': tile, 'color': None}
            elif parts[0] == 'robot-has':
                robot_name = parts[1]
                color = parts[2]
                if robot_name in robots:
                    robots[robot_name]['color'] = color

        total_heuristic = 0

        # Evaluate each goal tile
        for tile, req_color in self.goal_painted.items():
            current_color = current_painted.get(tile, None)
            if current_color == req_color:
                continue  # Already correctly painted

            # Check if required color is available
            if req_color not in self.available_colors:
                total_heuristic += 1000  # High penalty for unavailable color
                continue

            min_cost = float('inf')
            # Check each robot's cost to paint this tile
            for robot in robots.values():
                current_robot_tile = robot['tile']
                current_robot_color = robot['color']

                # Get adjacent tiles to the goal tile
                adjacent_tiles = self.adjacent_tiles.get(tile, [])
                if not adjacent_tiles:
                    continue  # No adjacent tiles (invalid state for solvable problem)

                # Calculate minimal Manhattan distance to any adjacent tile
                min_distance = min(
                    self.manhattan_distance(current_robot_tile, adj_tile)
                    for adj_tile in adjacent_tiles
                )

                # Determine color change cost
                color_change_cost = 0 if current_robot_color == req_color else 1

                # Total cost for this robot: move + paint + color change
                total_cost = min_distance + 1 + color_change_cost
                if total_cost < min_cost:
                    min_cost = total_cost

            # Add the minimal cost for this tile, or a penalty if no robot can reach
            if min_cost != float('inf'):
                total_heuristic += min_cost
            else:
                total_heuristic += 1000  # Penalty for unreachable tile

        return total_heuristic

    def manhattan_distance(self, tile1, tile2):
        """Compute Manhattan distance between two tiles based on their coordinates."""
        x1, y1 = self.parse_tile_coords(tile1)
        x2, y2 = self.parse_tile_coords(tile2)
        return abs(x1 - x2) + abs(y1 - y2)

    def parse_tile_coords(self, tile_name):
        """Extract (x, y) coordinates from a tile name (e.g., 'tile_3_2' -> (3, 2))."""
        parts = tile_name.split('_')
        return (int(parts[1]), int(parts[2]))
