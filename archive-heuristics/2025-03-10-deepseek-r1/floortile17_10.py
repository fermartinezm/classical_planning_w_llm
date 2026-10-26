from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class FloorTile17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles by considering:
    - The minimal movement required for robots to reach each tile.
    - The color changes needed if the robot's current color doesn't match the goal.
    - The paint action itself.

    # Assumptions
    - Tiles are arranged in a grid, and their coordinates can be derived from their names (e.g., tile_x_y).
    - Robots can move freely between adjacent tiles (Manhattan distance is used as a movement estimate).
    - Color changes are possible if the target color is available.

    # Heuristic Initialization
    - Extract goal tiles and their required colors.
    - Identify available colors from static facts.
    - Build a coordinate map for all tiles using adjacency relations from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet correctly painted:
        a. Check if the required color is available; if not, return a high cost.
        b. For each robot, calculate the Manhattan distance from its current position to the tile.
        c. Add 1 if the robot needs to change color and the target color is available.
        d. Add 1 for the paint action.
        e. Take the minimal cost across all robots for the tile.
    2. Sum the minimal costs for all tiles to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goals, static info, and tile coordinates."""
        self.goal_painted = {}
        for fact in task.goals:
            parts = get_parts(fact)
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_painted[tile] = color

        self.available_colors = set()
        self.tile_coords = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'available-color':
                self.available_colors.add(parts[1])
            elif parts[0] in ['up', 'down', 'left', 'right']:
                for tile in [parts[1], parts[2]]:
                    if tile.startswith('tile_'):
                        coords = list(map(int, tile.split('_')[1:3]))
                        self.tile_coords[tile] = (coords[0], coords[1])

        # Ensure all goal tiles have coordinates (in case not in static)
        for tile in self.goal_painted:
            if tile not in self.tile_coords and tile.startswith('tile_'):
                coords = list(map(int, tile.split('_')[1:3]))
                self.tile_coords[tile] = (coords[0], coords[1])

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        total_cost = 0

        # Check each goal tile
        for tile, required_color in self.goal_painted.items():
            if f'(painted {tile} {required_color})' in state:
                continue  # Already painted correctly

            if required_color not in self.available_colors:
                return float('inf')  # Unsolvable state

            # Collect robot positions and colors
            robots = {}
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'robot-at':
                    robot = parts[1]
                    pos = parts[2]
                    robots[robot] = {'pos': pos, 'color': None}
                elif parts[0] == 'robot-has':
                    robot = parts[1]
                    color = parts[2]
                    if robot in robots:
                        robots[robot]['color'] = color
                    else:
                        robots[robot] = {'pos': None, 'color': color}

            min_cost = float('inf')
            tile_x, tile_y = self.tile_coords[tile]

            for robot in robots.values():
                if robot['pos'] is None:
                    continue  # Invalid state, skip robot
                if robot['pos'] not in self.tile_coords:
                    continue  # Unknown position, skip

                # Calculate Manhattan distance
                rx, ry = self.tile_coords[robot['pos']]
                distance = abs(rx - tile_x) + abs(ry - tile_y)

                # Check color change needed
                color_change = 0 if robot['color'] == required_color else 1

                # Total cost for this robot
                cost = distance + color_change + 1  # +1 for paint
                if cost < min_cost:
                    min_cost = cost

            if min_cost == float('inf'):
                return float('inf')  # No robot can reach this tile

            total_cost += min_cost

        return total_cost
