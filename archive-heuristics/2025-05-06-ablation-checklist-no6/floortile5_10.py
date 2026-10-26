from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    # Use fnmatch for pattern matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-specific helper functions for floortile
def get_tile_coords(tile_name):
    """Parses a tile name like 'tile_r_c' into (row, column) integers."""
    # Assumes tile names are always in the format 'tile_row_col'
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # If row/col parts are not integers, this tile name is invalid for coordinate calculation.
            # Raising an error is appropriate as we cannot proceed with distance calculation.
            raise ValueError(f"Invalid integer in tile name: {tile_name}")
    else:
        # Handle unexpected format
        raise ValueError(f"Unexpected tile name format: {tile_name}")


def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two (row, col) tuples."""
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to paint all goal tiles that are not yet painted
    correctly. The estimate for each unpainted goal tile includes:
    1. The paint action itself (cost 1).
    2. The minimum movement cost for the closest robot to reach an adjacent tile.
    3. A color change cost (cost 1) if the closest robot does not have the required color.

    This heuristic is non-admissible but aims to guide a greedy search by
    prioritizing states where robots are closer to unpainted goal tiles
    and already possess the necessary color.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing goal conditions."""
        self.goals = task.goals
        # Static facts are not explicitly needed for this heuristic
        # as tile structure is inferred from naming convention.

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Extract relevant information from the current state
        robot_locations = {} # Map robot name to tile name
        robot_colors = {}    # Map robot name to color name
        painted_tiles = {}   # Map tile name to color name for painted tiles

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif predicate == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif predicate == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color

        total_cost = 0

        # Iterate through goal conditions to find unpainted goal tiles
        for goal_fact in self.goals:
            goal_parts = get_parts(goal_fact)
            if not goal_parts: continue # Skip malformed goals

            goal_predicate = goal_parts[0]
            # We only care about 'painted' goal conditions
            if goal_predicate == "painted" and len(goal_parts) == 3:
                target_tile, required_color = goal_parts[1], goal_parts[2]

                # Check if the tile is already painted correctly in the current state
                if painted_tiles.get(target_tile) != required_color:
                    # This tile needs to be painted with the required_color

                    # Cost Component 1: The paint action itself
                    total_cost += 1

                    # Cost Component 2 & 3: Movement and Color Change
                    # Estimate the cost for the closest robot to handle this tile
                    try:
                        target_coords = get_tile_coords(target_tile)
                    except ValueError:
                        # If target tile name is unparseable, we cannot estimate cost for it.
                        # This indicates an issue with the problem definition or state.
                        # For this heuristic, we'll skip this goal tile.
                        continue # Skip this goal tile

                    min_dist_to_adjacent = float('inf')
                    closest_robot = None

                    # Find the closest robot to any tile adjacent to the target tile
                    for robot, robot_tile in robot_locations.items():
                        try:
                            robot_coords = get_tile_coords(robot_tile)
                        except ValueError:
                             # Skip robot if its current tile name is unparseable
                             continue

                        # Calculate Manhattan distance from robot's current tile to the target tile
                        dist_to_target = manhattan_distance(robot_coords, target_coords)

                        # Calculate minimum moves to reach a tile adjacent to the target tile
                        if dist_to_target == 0:
                            # Robot is at the target tile, needs 1 move to an adjacent tile
                            dist_to_adjacent = 1
                        else:
                            # Robot is not at the target tile, needs dist_to_target - 1 moves
                            # to reach a tile adjacent to the target.
                            dist_to_adjacent = dist_to_target - 1


                        if dist_to_adjacent < min_dist_to_adjacent:
                            min_dist_to_adjacent = dist_to_adjacent
                            closest_robot = robot

                    # Add movement cost if at least one robot was found
                    if min_dist_to_adjacent != float('inf'):
                        total_cost += min_dist_to_adjacent # Cost for movement

                        # Cost Component 3: Color change if the closest robot doesn't have the color
                        # We assume the closest robot will paint this tile.
                        robot_current_color = robot_colors.get(closest_robot)
                        if robot_current_color != required_color:
                            total_cost += 1 # Cost for change_color action
                    # else: No robots found. The paint cost (1) is already added.
                    #       Movement and color change costs cannot be estimated.

        return total_cost
