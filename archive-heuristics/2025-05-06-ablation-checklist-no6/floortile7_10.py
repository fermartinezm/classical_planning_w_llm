import re
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses 'tile_R_C' into (R, C) tuple of integers."""
    match = re.match(r"tile_(\d+)_(\d+)", tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    # Return None if the tile name format is unexpected.
    # Assuming all tile names follow tile_R_C format based on examples.
    return None

def manhattan_distance(coord1, coord2):
    """Calculates Manhattan distance between two (row, col) coordinates."""
    if coord1 is None or coord2 is None:
        # Cannot calculate distance if coordinates are invalid
        return float('inf')
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost based on:
    1. Detecting unsolvable states (goal tile painted with wrong color).
    2. The number of goal tiles not yet painted correctly.
    3. The number of colors needed for unpainted goal tiles that no robot currently possesses.
    4. An approximation of movement cost: sum of minimum distances from any robot to any tile adjacent to an unpainted goal tile from which it can be painted.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and grid structure."""
        self.goals = task.goals

        # Extract goal tiles and their required colors
        self.goal_tiles_and_colors = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_tiles_and_colors[tile] = color

        # Extract grid adjacency information to find adjacent tiles from which a tile can be painted
        # Map tile_Y -> set of tiles tile_X such that robot at tile_X can paint tile_Y
        # based on (up tile_Y tile_X), (down tile_Y tile_X), etc.
        self.paintable_from = {}
        for fact in task.static:
             parts = get_parts(fact)
             if parts[0] in ["up", "down", "left", "right"]:
                 # (dir tile_Y tile_X) means tile_Y is in direction 'dir' from tile_X
                 # Robot at tile_X can paint tile_Y using paint_dir action
                 tile_y, tile_x = parts[1], parts[2]
                 self.paintable_from.setdefault(tile_y, set()).add(tile_x)

    def __call__(self, node):
        """
        Compute the heuristic value.

        Returns float('inf') if a goal tile is painted with the wrong color.
        Otherwise, returns (Num unpainted goal tiles) + (Num needed colors not held) + (Sum of min movement costs).
        """
        state = node.state

        # Build necessary state information maps/sets for efficient lookup
        current_painted_colors = {}
        robot_locations = {}
        robot_colors = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                current_painted_colors[tile] = color
            elif parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has":
                color = parts[2]
                robot_colors.add(color)

        unpainted_tile_count = 0
        colors_needed_for_unpainted_tiles = set()
        movement_cost_sum = 0

        # Process goal tiles
        for tile, required_color in self.goal_tiles_and_colors.items():
            current_color = current_painted_colors.get(tile) # None if not painted

            if current_color is None:
                # Tile is not painted (must be clear if solvable)
                unpainted_tile_count += 1
                colors_needed_for_unpainted_tiles.add(required_color)

                # Calculate minimum movement cost to a paintable adjacent tile
                min_dist_to_paintable_adj = float('inf')

                # Get tiles from which this goal tile can be painted
                paintable_adj_tiles = self.paintable_from.get(tile, set())

                # If a goal tile cannot be painted from anywhere, it's unsolvable.
                # This check is important for robustness, although unlikely in valid instances.
                if not paintable_adj_tiles:
                     return float('inf')

                # Calculate min distance from any robot to any paintable adjacent tile
                for robot_tile in robot_locations.values():
                    robot_coord = parse_tile_name(robot_tile)
                    for adj_tile in paintable_adj_tiles:
                        adj_coord = parse_tile_name(adj_tile)
                        dist = manhattan_distance(robot_coord, adj_coord)
                        min_dist_to_paintable_adj = min(min_dist_to_paintable_adj, dist)

                # Add the minimum distance to the sum.
                # If min_dist_to_paintable_adj is still inf, it means no robot can reach any paintable tile.
                # This could happen if the grid is disconnected or there are no robots.
                # In such cases, the state is likely unsolvable.
                if min_dist_to_paintable_adj != float('inf'):
                     movement_cost_sum += min_dist_to_paintable_adj
                else:
                     # No robot can reach a paintable position for this tile. Unsolvable.
                     return float('inf')


            elif current_color != required_color:
                # Tile is painted with the wrong color
                return float('inf') # Unsolvable state

            # If current_color == required_color, the goal is met for this tile, do nothing.

        # If all goal tiles are painted correctly, unpainted_tile_count will be 0.
        if unpainted_tile_count == 0:
            return 0

        # Count colors needed for unpainted tiles that no robot has
        color_change_count = 0
        for color in colors_needed_for_unpainted_tiles:
            if color not in robot_colors:
                color_change_count += 1

        # Total heuristic
        # Each unpainted tile needs a paint action (cost 1).
        # Colors might need changing (cost 1 per color type not held).
        # Robots need to move to paintable adjacent tiles (sum of min distances).
        # Total = unpainted_tile_count + color_change_count + movement_cost_sum.

        return unpainted_tile_count + color_change_count + movement_cost_sum
