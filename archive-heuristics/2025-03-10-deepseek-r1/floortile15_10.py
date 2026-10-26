from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def parse_tile(tile_name):
    """Parse tile coordinates from a tile name (e.g., 'tile_1_2' into (1, 2))."""
    parts = tile_name.split('_')
    return (int(parts[1]), int(parts[2]))

class floortile15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with their respective colors. It considers the minimal robot movement to adjacent tiles, necessary color changes, and the paint action for each tile.

    # Assumptions
    - The grid is regular, and tile names follow the format 'tile_X_Y' where X and Y are integers.
    - Robots can move freely between adjacent tiles as defined by the static 'up', 'down', 'left', 'right' predicates.
    - Each robot can carry one color at a time, and color changes are instantaneous actions.

    # Heuristic Initialization
    - Extract the goal conditions to determine which tiles need to be painted and their required colors.
    - Static facts are not directly used since Manhattan distance is assumed for movement cost.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Goal Extraction**: Identify all tiles that need to be painted and their target colors from the problem's goal conditions.
    2. **Current State Analysis**: Check which tiles are already correctly painted and gather robots' current positions and colors.
    3. **Cost Calculation for Each Tile**:
        a. For each unpainted goal tile, calculate the minimal distance any robot needs to move to an adjacent tile.
        b. Determine if the closest robot needs to change color to match the target tile's color.
        c. Sum the movement, color change (if needed), and painting actions for each tile.
    4. **Summing Costs**: Aggregate the costs for all tiles to get the total estimated actions required.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_tiles = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the given state."""
        state = node.state
        current_painted = {}
        robots = {}

        # Extract current painted tiles
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                current_painted[tile] = color
            elif parts[0] == 'robot-at':
                robot = parts[1]
                tile = parts[2]
                x, y = parse_tile(tile)
                if robot not in robots:
                    robots[robot] = {'pos': None, 'color': None}
                robots[robot]['pos'] = (x, y)
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                if robot not in robots:
                    robots[robot] = {'pos': None, 'color': None}
                robots[robot]['color'] = color

        total_cost = 0

        for tile, req_color in self.goal_tiles.items():
            if current_painted.get(tile) == req_color:
                continue  # Already painted correctly

            tx, ty = parse_tile(tile)
            min_distance = float('inf')
            candidates = []

            for robot in robots.values():
                pos = robot['pos']
                if pos is None:
                    continue  # Skip if robot's position is unknown (invalid state)
                rx, ry = pos
                distance_to_t = abs(rx - tx) + abs(ry - ty)
                adj_distance = distance_to_t - 1  # Distance to adjacent tile

                if adj_distance < min_distance:
                    min_distance = adj_distance
                    candidates = [robot]
                elif adj_distance == min_distance:
                    candidates.append(robot)

            if not candidates:
                continue  # No robots available; assume this is handled elsewhere

            # Check if any closest robot has the required color
            color_cost = 0 if any(r['color'] == req_color for r in candidates) else 1
            cost = min_distance + color_cost + 1  # +1 for paint action
            total_cost += cost

        return total_cost
