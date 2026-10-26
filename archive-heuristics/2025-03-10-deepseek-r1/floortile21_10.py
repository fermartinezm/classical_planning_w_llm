from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortile21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with their respective colors. It considers robot movements, color changes, and painting actions. The heuristic sums the minimal cost for each unpainted goal tile, considering the closest robot and optimal path.

    # Assumptions
    - Robots can move freely between adjacent tiles (ignoring dynamic obstacles).
    - Each tile can be painted from any adjacent tile using the corresponding paint action.
    - Color changes are required if the robot's current color does not match the goal color.

    # Heuristic Initialization
    - Extracts goal conditions to determine required colors for each tile.
    - Parses static facts to build adjacency lists for each tile (tiles from which it can be painted).
    - Computes coordinates for each tile based on their names for Manhattan distance calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Current State Information**:
       - Identify painted tiles and their colors.
       - Track each robot's position and current color.

    2. **Evaluate Each Goal Tile**:
       - Skip tiles already painted correctly.
       - For each unpainted tile, determine adjacent tiles from which it can be painted.

    3. **Calculate Minimal Cost per Tile**:
       - For each adjacent tile, compute the Manhattan distance from each robot's current position.
       - Include color change cost if the robot's color does not match the goal.
       - Add 1 action for painting.
       - Track the minimal cost across all robots and adjacent tiles.

    4. **Sum Costs**:
       - Aggregate minimal costs for all unpainted tiles to form the heuristic value.
    """

    def __init__(self, task):
        """Initialize heuristic with goal colors, adjacency data, and tile coordinates."""
        self.goal_colors = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                self.goal_colors[parts[1]] = parts[2]

        # Build adjacency list and tile coordinates
        tiles = set()
        self.adjacents = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                t1, t2 = parts[1], parts[2]
                tiles.update({t1, t2})
                self.adjacents.setdefault(t1, []).append(t2)

        self.tile_coords = {}
        for tile in tiles:
            _, x, y = tile.split('_')
            self.tile_coords[tile] = (int(x), int(y))

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_painted = {}
        robot_positions = {}
        robot_colors = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted':
                current_painted[parts[1]] = parts[2]
            elif parts[0] == 'robot-at':
                robot_positions[parts[1]] = parts[2]
            elif parts[0] == 'robot-has':
                robot_colors[parts[1]] = parts[2]

        heuristic_value = 0
        for tile, req_color in self.goal_colors.items():
            if current_painted.get(tile) == req_color:
                continue

            min_cost = float('inf')
            for adj_tile in self.adjacents.get(tile, []):
                adj_coords = self.tile_coords.get(adj_tile)
                if not adj_coords:
                    continue

                for robot, pos in robot_positions.items():
                    robot_coords = self.tile_coords.get(pos)
                    if not robot_coords:
                        continue

                    dx = abs(robot_coords[0] - adj_coords[0])
                    dy = abs(robot_coords[1] - adj_coords[1])
                    distance = dx + dy

                    current_color = robot_colors.get(robot, '')
                    color_cost = 0 if current_color == req_color else 1
                    total_cost = distance + color_cost + 1  # +1 for paint action

                    if total_cost < min_cost:
                        min_cost = total_cost

            if min_cost != float('inf'):
                heuristic_value += min_cost

        return heuristic_value
