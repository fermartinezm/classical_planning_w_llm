from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and ensure consistent splitting
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_tile_coords(tile_name):
    """
    Extract row and column coordinates from a tile name like 'tile_R_C'.
    Assumes tile names follow the format 'tile_row_col'.
    Returns a tuple (row, col) of integers.
    """
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col might not be integers, though unlikely in benchmarks
            pass
    # Return None or raise error for unexpected formats
    return None

def manhattan_distance(coords1, coords2):
    """
    Calculate the Manhattan distance between two tile coordinates (row, col).
    """
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance for invalid coordinates
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles with the correct color. It considers the cost of painting,
    changing colors, and moving robots.

    # Assumptions
    - Tiles are named in a grid format 'tile_R_C' allowing Manhattan distance calculation.
    - Multiple robots can exist and work in parallel.
    - The cost of moving is estimated by Manhattan distance, ignoring the 'clear'
      precondition for movement paths (only considering 'clear' for the paint target).
    - Color change cost is estimated based on the number of distinct colors needed.

    # Heuristic Initialization
    - Extract the goal conditions (which tiles need which color).
    - Static facts like grid connectivity are not explicitly used for movement,
      relying on the tile naming convention for distance.

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unpainted Goal Tiles:
       - Iterate through the goal conditions `(painted ?tile ?color)`.
       - For each such goal, check if the state contains the exact same fact.
       - If the goal fact is not in the state, the tile needs to be painted.
       - Store these unpainted goal tiles along with their required color.

    2. Identify Needed Colors:
       - Collect the set of distinct colors required by the unpainted goal tiles.

    3. Identify Robot States:
       - Find the current location `(robot-at ?r ?x)` and held color `(robot-has ?r ?c)`
         for each robot in the state.

    4. Calculate Base Cost (Painting):
       - Each unpainted goal tile requires one `paint` action. Add the count of
         unpainted goal tiles to the heuristic.

    5. Estimate Color Change Cost:
       - Each distinct color needed for the unpainted tiles must be acquired by
         at least one robot at some point. A simple estimate is to add the number
         of distinct colors needed. This is an overestimate if robots can share
         colors efficiently or if a robot already has a needed color, but it
         encourages working towards states where the right colors are available.

    6. Estimate Movement Cost:
       - For each unpainted goal tile, a robot needs to move to a tile adjacent
         to it.
       - A simplified movement cost estimate: For each unpainted goal tile, find
         the minimum Manhattan distance from *any* robot's current location to
         that tile's location. Sum these minimum distances. This captures the
         spatial distribution of the work and accounts for multiple robots
         sharing the load by considering the closest robot for each tile. It
         ignores the adjacency requirement for painting and the 'clear' constraint
         for movement paths, providing a non-admissible estimate.

    7. Sum Costs:
       - The total heuristic value is the sum of the painting cost, the estimated
         color change cost, and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are not strictly needed for this heuristic's calculation,
        # as grid structure is inferred from tile names and movement is estimated
        # via Manhattan distance.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Unpainted Goal Tiles and Needed Colors
        unpainted_goal_tiles_info = [] # List of (tile_name, color_name)
        needed_colors = set()
        goal_painted_facts = {fact for fact in self.goals if match(fact, "painted", "*", "*")}

        for goal_fact in goal_painted_facts:
            if goal_fact not in state:
                # This tile needs to be painted
                _, tile_name, color_name = get_parts(goal_fact)
                unpainted_goal_tiles_info.append((tile_name, color_name))
                needed_colors.add(color_name)

        # If all goal tiles are painted correctly, the heuristic is 0.
        if not unpainted_goal_tiles_info:
            return 0

        # 3. Identify Robot States (Locations and Colors)
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}

        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot_name, tile_name = get_parts(fact)
                robot_locations[robot_name] = tile_name
            elif match(fact, "robot-has", "*", "*"):
                _, robot_name, color_name = get_parts(fact)
                robot_colors[robot_name] = color_name

        # Convert robot locations to coordinates
        robot_coords = {
            robot_name: get_tile_coords(tile_name)
            for robot_name, tile_name in robot_locations.items()
        }

        # 4. Calculate Base Cost (Painting)
        # Each unpainted tile needs one paint action.
        h = len(unpainted_goal_tiles_info)

        # 5. Estimate Color Change Cost
        # Add the number of distinct colors needed.
        h += len(needed_colors)

        # 6. Estimate Movement Cost
        # For each unpainted tile, find the minimum distance from any robot.
        # Sum these minimum distances.
        movement_cost = 0
        for tile_name, _ in unpainted_goal_tiles_info:
            tile_coords = get_tile_coords(tile_name)
            if tile_coords is None:
                 # Cannot calculate distance for this tile, skip or add penalty
                 continue # Skipping is safer

            min_dist_to_tile = float('inf')
            for robot_name, current_robot_coords in robot_coords.items():
                 dist = manhattan_distance(current_robot_coords, tile_coords)
                 min_dist_to_tile = min(min_dist_to_tile, dist)

            if min_dist_to_tile != float('inf'):
                 movement_cost += min_dist_to_tile

        h += movement_cost

        return h

