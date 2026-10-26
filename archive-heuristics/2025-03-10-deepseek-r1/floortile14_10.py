from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortile14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles by calculating the minimal steps each robot would need to move to each tile, considering color changes and painting actions.

    # Assumptions
    - Robots can move freely between tiles (ignoring 'clear' preconditions for simplicity).
    - Each tile is painted by the closest robot, and color changes are done once per tile if needed.
    - The grid structure is inferred from tile names (tile_x_y) to compute Manhattan distances.

    # Heuristic Initialization
    - Extract goal tiles and their required colors.
    - Parse static adjacency facts to build a coordinate system for tiles.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet painted correctly:
        a. Determine the required color.
        b. For each robot, calculate the Manhattan distance from its current position to the tile.
        c. Check if the robot's current color matches the required color.
        d. Compute the cost as distance + (1 if color change needed else 0).
        e. Select the minimal cost across all robots.
    2. Sum all minimal costs for all tiles to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and tile coordinates."""
        self.goal_tiles = {}
        self.tile_coords = {}

        # Extract goal tiles and their colors
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'painted' and len(parts) == 3:
                tile = parts[1]
                color = parts[2]
                self.goal_tiles[tile] = color

        # Extract tile coordinates from static adjacency facts
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] in ('up', 'down', 'left', 'right') and len(parts) == 3:
                for tile in (parts[1], parts[2]):
                    if tile not in self.tile_coords:
                        coords = tile.split('_')[1:]
                        if len(coords) >= 2:
                            try:
                                x = int(coords[0])
                                y = int(coords[1])
                                self.tile_coords[tile] = (x, y)
                            except ValueError:
                                pass

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        total_cost = 0

        # Extract current robot positions and colors
        robots = {}
        for fact in state:
            parts = fact[1:-1].split()
            if not parts:
                continue
            if parts[0] == 'robot-at' and len(parts) == 3:
                robot = parts[1]
                tile = parts[2]
                robots[robot] = {'pos': tile}
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot = parts[1]
                color = parts[2]
                if robot in robots:
                    robots[robot]['color'] = color
                else:
                    robots[robot] = {'color': color}

        # Check each goal tile
        for tile, req_color in self.goal_tiles.items():
            # Check if already painted correctly
            if f'(painted {tile} {req_color})' in state:
                continue

            tile_coords = self.tile_coords.get(tile)
            if not tile_coords:
                continue

            min_cost = float('inf')
            for robot, info in robots.items():
                robot_pos = info.get('pos')
                robot_color = info.get('color')
                if not robot_pos or not robot_color:
                    continue

                robot_coords = self.tile_coords.get(robot_pos)
                if not robot_coords:
                    continue

                # Calculate Manhattan distance
                distance = abs(robot_coords[0] - tile_coords[0]) + abs(robot_coords[1] - tile_coords[1])
                # Check color match
                color_cost = 0 if robot_color == req_color else 1
                # Total cost for this robot
                cost = distance + color_cost
                if cost < min_cost:
                    min_cost = cost

            if min_cost != float('inf'):
                total_cost += min_cost

        return total_cost
