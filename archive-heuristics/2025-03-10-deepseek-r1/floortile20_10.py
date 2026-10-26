from heuristics.heuristic_base import Heuristic
from collections import defaultdict

class floortile20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with their respective colors. It considers the minimal steps each robot needs to move to an adjacent tile of the target, change color if necessary, and paint.

    # Assumptions
    - Robots can move freely between tiles (ignoring current 'clear' status for efficiency).
    - Each tile requires a separate paint action, even if multiple tiles can be painted in sequence.
    - Color changes are counted per tile, even if a robot can paint multiple tiles of the same color after one change.

    # Heuristic Initialization
    - Extract the goal conditions to determine which tiles need to be painted and their colors.
    - Precompute adjacency relations between tiles from static facts (up, down, left, right).

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet painted correctly:
        a. Find its required color.
        b. For each robot:
            i. Calculate the minimal movement steps to reach any adjacent tile of the goal.
            ii. Add 1 action for painting.
            iii. Add 1 action if the robot's color doesn't match the required color.
        c. Take the minimal cost across all robots for this tile.
    2. Sum the minimal costs for all goal tiles.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal and adjacency information."""
        self.goal_painted = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_painted[tile] = color

        # Build adjacency map: tile -> list of adjacent tiles
        self.adjacent = defaultdict(list)
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] in ['up', 'down', 'left', 'right']:
                tile1, tile2 = parts[1], parts[2]
                self.adjacent[tile2].append(tile1)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the given state."""
        state = node.state
        sum_h = 0

        # Extract robot positions and colors
        robot_pos = {}
        robot_color = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'robot-at':
                robot = parts[1]
                tile = parts[2]
                robot_pos[robot] = tile
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                robot_color[robot] = color

        robots = robot_pos.keys()

        for tile, required_color in self.goal_painted.items():
            # Check if already painted correctly
            if f'(painted {tile} {required_color})' in state:
                continue

            min_cost = float('inf')
            adjacent_tiles = self.adjacent.get(tile, [])

            if not adjacent_tiles:
                continue  # No adjacent tiles (unlikely)

            for robot in robots:
                current_pos = robot_pos.get(robot)
                if not current_pos:
                    continue
                current_clr = robot_color.get(robot)
                if not current_clr:
                    continue

                # Find minimal distance to any adjacent tile
                min_distance = float('inf')
                for adj_tile in adjacent_tiles:
                    # Parse coordinates
                    def get_coords(t):
                        parts = t.split('_')
                        return (int(parts[1]), int(parts[2]))
                    current_x, current_y = get_coords(current_pos)
                    adj_x, adj_y = get_coords(adj_tile)
                    distance = abs(current_x - adj_x) + abs(current_y - adj_y)
                    if distance < min_distance:
                        min_distance = distance

                if min_distance == float('inf'):
                    continue

                # Calculate cost
                color_change = 0 if current_clr == required_color else 1
                total_cost = min_distance + color_change + 1  # move + change + paint

                if total_cost < min_cost:
                    min_cost = total_cost

            if min_cost == float('inf'):
                return float('inf')  # Unsolvable state

            sum_h += min_cost

        return sum_h
