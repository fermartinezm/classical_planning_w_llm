# The code should be placed in a file like heuristics/floortileHeuristic.py
# It needs access to the Heuristic base class.
# Assuming the directory structure allows importing heuristics.heuristic_base

from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for float('inf')

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    # Handle potential empty strings from multiple spaces or malformed facts
    return [part for part in fact[1:-1].split() if part]

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_row_col' into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            # Convert parts[1] and parts[2] to integers
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # If conversion fails, it's not in the expected format
            # print(f"Warning: Could not parse tile name '{tile_name}' as tile_row_col") # Avoid printing in heuristic
            return None
    else:
        # If parts structure is wrong, it's not in the expected format
        # print(f"Warning: Tile name '{tile_name}' does not match tile_row_col format") # Avoid printing in heuristic
        return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coord1 = parse_tile_name(tile1_name)
    coord2 = parse_tile_name(tile2_name)
    if coord1 is None or coord2 is None:
        # If parsing failed for either tile, distance is effectively infinite
        # This indicates a problem with the input state/goals/objects
        return float('inf')
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated minimum cost for each goal tile that is not yet painted
    with the correct color. For each such tile, it calculates the minimum
    cost across all robots to move adjacent to the tile, change color if
    necessary, and paint it. The 'clear' constraint for movement is ignored
    (Manhattan distance is used), and resource conflicts (multiple robots
    needing the same tile or color) are not explicitly modeled, making it
    non-admissible but potentially effective for greedy search.

    Assumptions:
    - Tile names follow the format 'tile_row_col' where row and col are integers.
    - Robots always hold a color (no 'free-color' state relevant for change_color).
    - Available colors are static and globally accessible for change_color action.
    - The grid is connected as implied by 'up', 'down', 'left', 'right' predicates,
      and Manhattan distance is a reasonable estimate of movement cost ignoring
      dynamic 'clear' constraints.
    - Goal predicates are only of the form (painted ?tile ?color).

    Heuristic Initialization:
    The constructor extracts the goal conditions, specifically identifying
    which tiles need to be painted and with which color. This information
    is stored in a dictionary mapping tile names to required colors.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value to 0.
    2. Identify the current state of all robots (their location and color).
       This is done by iterating through the current state facts and finding
       '(robot-at ?r ?x)' and '(robot-has ?r ?c)' predicates. Store this in
       a dictionary mapping robot names to (location, color).
    3. Iterate through each goal condition '(painted ?tile ?color)' stored
       during initialization.
    4. For each goal condition, check if the tile is already painted with
       the correct color in the current state. This is done by checking
       if '(painted ?tile ?color)' is present in the state.
    5. If the tile is already painted correctly, this goal is satisfied,
       and we continue to the next goal tile.
    6. If the tile is NOT painted correctly (it must be 'clear' based on
       domain rules, as painting a non-clear tile is not possible),
       we need to estimate the cost to paint it.
    7. For this unpainted goal tile needing color C at location T_loc:
        a. Initialize a variable `min_cost_for_tile` to infinity.
        b. Iterate through each robot R with current location R_loc and color R_color.
        c. Calculate the Manhattan distance between R_loc and T_loc.
        d. Calculate the estimated move cost for robot R to get adjacent to T_loc:
           `moves_cost = max(0, dist - 1)`.
           (We need to reach a tile 1 step away from the target tile).
        e. Calculate the color change cost for robot R:
           `color_cost = 1` if `R_color != C` else `0`.
           (Assumes changing color costs 1 action if the robot has any color).
        f. The estimated cost for robot R to paint this tile is `moves_cost + color_cost + 1` (the +1 is for the paint action itself).
        g. Update `min_cost_for_tile = min(min_cost_for_tile, estimated_cost_for_robot_R)`.
    8. After checking all robots, add `min_cost_for_tile` to the total heuristic value.
    9. After iterating through all unpainted goal tiles, the total heuristic value
       is the sum of the minimum estimated costs for each. Return this total.
    10. If the total heuristic is calculated to be 0, it means all goal painted
        predicates were satisfied, so the state is a goal state.
    """
    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static

        # Extract goal painted tiles and their required colors
        self.goal_painted_tiles = {}
        # The task.goals attribute is a frozenset of goal facts.
        for goal_fact in self.goals:
            # Goal facts are expected to be strings like '(painted tile_x_y color)'
            parts = get_parts(goal_fact)
            # Ensure the goal is a painted predicate with correct number of arguments
            if parts and parts[0] == "painted" and len(parts) == 3:
                 tile, color = parts[1], parts[2]
                 self.goal_painted_tiles[tile] = color
            # Note: More complex goal structures (like 'and', 'or', 'not') are not
            # handled by this simple extraction, assuming goals are flat conjunctions
            # of positive painted literals as seen in the examples.

        # Note: We don't strictly need the grid structure or adjacency
        # from static facts if we rely purely on Manhattan distance
        # derived from tile names. If tile names weren't 'tile_r_c',
        # we would need to build the graph here.

    def __call__(self, node):
        state = node.state

        # Check if goal is reached (heuristic is 0)
        # This is a quick check. The main loop below should also result in 0
        # if the goal is reached, but this is more explicit and potentially faster.
        if self.goals <= state:
             return 0

        # Extract current robot locations and colors
        robot_info = {} # {robot_name: [location, color]}
        for fact in state:
            parts = get_parts(fact)
            # Check for robot-at and robot-has predicates
            if len(parts) >= 3: # Ensure enough parts before accessing indices
                if parts[0] == "robot-at":
                    robot, location = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = [None, None]
                    robot_info[robot][0] = location
                elif parts[0] == "robot-has":
                    robot, color = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = [None, None]
                    robot_info[robot][1] = color

        total_heuristic = 0

        # Iterate through goal painted tiles
        for goal_tile, goal_color in self.goal_painted_tiles.items():
            # Check if this tile is already painted correctly in the current state
            is_painted_correctly = f"(painted {goal_tile} {goal_color})" in state

            if not is_painted_correctly:
                # This tile needs painting. It must be clear according to domain rules
                # for the paint action to be applicable.
                # Find the minimum cost for any robot to paint this tile.
                min_cost_for_tile = float('inf')

                # Check if there are any robots available
                if not robot_info:
                    # If no robots and tiles need painting, it's unsolvable
                    return float('inf') # Return infinity to indicate unsolvability

                for robot_name, (robot_location, robot_color) in robot_info.items():
                    # Ensure robot location and color are known. In a valid state,
                    # all robots should have both robot-at and robot-has facts.
                    if robot_location is None or robot_color is None:
                         # This robot's state is incomplete, skip it.
                         continue

                    # Calculate Manhattan distance from robot's current location to the goal tile
                    dist = manhattan_distance(robot_location, goal_tile)

                    # If distance is inf (e.g., tile name parsing failed), this robot cannot reach
                    if dist == float('inf'):
                        continue

                    # Estimated move cost: number of moves to get adjacent to the goal tile.
                    # This is max(0, distance - 1).
                    moves_cost = max(0, dist - 1)

                    # Estimated color change cost: 1 if the robot doesn't have the required color, 0 otherwise.
                    # Assumes a robot can change color in 1 step if it has any color.
                    color_cost = 0
                    if robot_color != goal_color:
                        color_cost = 1

                    # Total estimated cost for this robot to paint this specific tile:
                    # moves_cost + color_cost + 1 (for the paint action itself).
                    estimated_cost_for_robot = moves_cost + color_cost + 1

                    # Update the minimum cost found so far for this goal tile
                    min_cost_for_tile = min(min_cost_for_tile, estimated_cost_for_robot)

                # Add the minimum cost found for this tile to the total heuristic.
                # If min_cost_for_tile is still inf, it means no robot could reach this tile.
                # This state is likely unsolvable.
                if min_cost_for_tile == float('inf'):
                    return float('inf') # Return infinity to indicate unsolvability
                else:
                     total_heuristic += min_cost_for_tile

        return total_heuristic
