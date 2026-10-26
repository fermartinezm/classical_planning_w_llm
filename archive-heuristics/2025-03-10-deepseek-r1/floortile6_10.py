from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def get_tile_coords(tile_name):
    """Extract coordinates from a tile name (assumes format tile_X_Y)."""
    parts = tile_name.split('_')
    if len(parts) < 3:
        return None
    try:
        x = int(parts[1])
        y = int(parts[2])
        return (x, y)
    except:
        return None

class floortile6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles by calculating the minimal cost for each tile, considering robot movement, color changes, and painting actions.

    # Assumptions
    - Tiles are named in the format tile_X_Y, where X and Y are integers representing coordinates.
    - Robots can move freely between tiles without considering other robots (optimistic assumption).
    - Color changes are possible if the required color is available.

    # Heuristic Initialization
    - Extract adjacency relationships between tiles from static facts.
    - Identify available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet painted correctly:
        a. Identify required color and adjacent tiles.
        b. For each robot, compute the minimal movement cost to reach any adjacent tile.
        c. Add color change cost if the robot's current color doesn't match.
        d. Include the paint action cost.
    2. Sum the minimal costs for all tiles to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize with adjacency map and available colors from static facts."""
        self.goals = task.goals
        static = task.static

        self.adjacent_map = defaultdict(list)
        self.available_colors = set()

        for fact in static:
            parts = get_parts(fact)
            if parts[0] == 'available-color':
                self.available_colors.add(parts[1])
            elif parts[0] in ['up', 'down', 'left', 'right']:
                x, y = parts[1], parts[2]
                self.adjacent_map[y].append(x)  # y's adjacent tile x in direction

        # Extract goal painted conditions
        self.goal_painted = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        if self.goal_painted.items() <= {fact[1:-1].split()[1:3] for fact in state if fact.startswith('(painted')}:
            return 0

        # Extract current painted tiles and robot states
        current_painted = {}
        robots = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted':
                current_painted[parts[1]] = parts[2]
            elif parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robots[robot] = {'tile': tile, 'coords': get_tile_coords(tile)}
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                if robot not in robots:
                    robots[robot] = {'color': color}
                else:
                    robots[robot]['color'] = color

        # Identify unpainted goal tiles
        unpainted = []
        for tile, req_color in self.goal_painted.items():
            if current_painted.get(tile) != req_color:
                unpainted.append((tile, req_color))

        total_cost = 0
        for tile, req_color in unpainted:
            adjacent = self.adjacent_map.get(tile, [])
            if not adjacent:
                continue

            min_tile_cost = float('inf')
            for robot in robots.values():
                if 'coords' not in robot or not robot['coords']:
                    continue

                rx, ry = robot['coords']
                min_dist = min(
                    (abs(rx - x) + abs(ry - y) for x, y in (get_tile_coords(a) for a in adjacent) if x is not None and y is not None),
                    default=float('inf')
                )

                if min_dist == float('inf'):
                    continue

                color_cost = 0 if robot.get('color') == req_color else 1 if req_color in self.available_colors else 0
                total_robot_cost = min_dist + color_cost + 1  # move + change + paint

                if total_robot_cost < min_tile_cost:
                    min_tile_cost = total_robot_cost

            if min_tile_cost != float('inf'):
                total_cost += min_tile_cost

        return total_cost
