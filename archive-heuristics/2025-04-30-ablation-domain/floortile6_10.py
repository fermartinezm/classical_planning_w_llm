from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

# Helper functions

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_tile_coords(tile_name):
    """
    Parses a tile name string like 'tile_r_c' into integer coordinates (r, c).
    Returns None if parsing fails.
    """
    if not isinstance(tile_name, str):
        return None
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
        else:
            return None
    except (ValueError, IndexError):
        return None

# The Heuristic class implementation
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles.
    It calculates the minimum Manhattan distance for a robot with the correct color
    to reach each unpainted goal tile, plus one action for painting. The total
    heuristic is the sum of these costs for all unpainted goal tiles.

    # Assumptions
    - Robots have a fixed color throughout the task (as defined by robot-has in the state).
    - Tiles that need to be painted are initially clear and remain clear until painted.
    - Any robot with the correct color can paint a tile once it reaches it.
    - The cost of movement between adjacent tiles is 1.
    - The cost of painting a tile is 1.
    - Manhattan distance is a reasonable estimate for movement cost on the grid.
    - There is at least one robot with the required color for each goal tile in a solvable problem.

    # Heuristic Initialization
    - Extracts the goal conditions from the task to identify which tiles need to be painted and what color.
    - Note: Robot colors and locations are dynamic and extracted from the state in __call__.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted tile color)` from the task's goals. Store these as a mapping from tile to required color.
    2. For a given state:
        a. Identify the current location of each robot from the state facts `(robot-at robot location)`.
        b. Identify the color of each robot from the state facts `(robot-has robot color)`. Create a mapping from color to a list of robots that have that color.
        c. Initialize the total heuristic cost to 0.
        d. Iterate through each goal tile and its required color (from step 1).
        e. For the current goal tile `T` and required color `C`:
            i. Check if the fact `(painted T C)` is already true in the current state. If yes, this goal is satisfied for this tile; continue to the next goal tile.
            ii. If the goal `(painted T C)` is not satisfied:
                - Find all robots `R` that have color `C` (using the mapping from step 2b).
                - If no such robot exists in the current state, this goal is unreachable from here; return infinity.
                - For each candidate robot `R`, find its current location `Loc_R` from the mapping in step 2a.
                - Calculate the Manhattan distance between `Loc_R` and `T`. This involves parsing the tile names (e.g., 'tile_r_c') into row and column coordinates using `get_tile_coords` and computing `abs(r_R - r_T) + abs(c_R - c_T)`.
                - Find the minimum Manhattan distance among all candidate robots with color `C` to tile `T`.
                - Add this minimum distance plus 1 (for the paint action) to the total heuristic cost.
        f. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        Robot properties (location, color) are dynamic and read from the state.
        """
        self.goals = task.goals

        # Extract goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to paint all goal tiles.
        """
        state = node.state

        # Identify current robot locations and colors from the state
        robot_locations = {} # robot -> location
        robot_colors = {} # robot -> color
        color_robots = {} # color -> list of robots

        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "robot-at":
                robot, location = parts[1], parts[2]
                robot_locations[robot] = location
            elif len(parts) == 3 and parts[0] == "robot-has":
                 robot, color = parts[1], parts[2]
                 robot_colors[robot] = color
                 color_robots.setdefault(color, []).append(robot)

        total_cost = 0

        # Iterate through each goal tile
        for goal_tile, required_color in self.goal_tiles.items():
            # Check if the tile is already painted with the correct color
            if f"(painted {goal_tile} {required_color})" in state:
                continue # This goal is satisfied

            # This tile needs to be painted. Find the closest robot with the right color.
            candidate_robots = color_robots.get(required_color, [])

            if not candidate_robots:
                # If no robot has the required color in the current state,
                # this goal is unreachable from here.
                return math.inf

            min_dist = math.inf

            goal_coords = get_tile_coords(goal_tile)
            if goal_coords is None:
                 # Handle parsing error for goal tile name
                 return math.inf # Indicate unreachable or invalid state

            goal_r, goal_c = goal_coords

            found_located_robot = False
            for robot in candidate_robots:
                robot_loc = robot_locations.get(robot)
                if robot_loc:
                    found_located_robot = True
                    robot_coords = get_tile_coords(robot_loc)
                    if robot_coords is not None:
                        robot_r, robot_c = robot_coords
                        dist = abs(robot_r - goal_r) + abs(robot_c - goal_c)
                        min_dist = min(min_dist, dist)

            if not found_located_robot or min_dist == math.inf:
                 # No located robot found with the required color
                 # This might happen if a robot exists with the color but its location is not in the state
                 # (e.g., problem definition error or unexpected state).
                 # Treat as unreachable.
                 return math.inf

            # Cost for this tile: min moves to get a robot there + 1 paint action
            total_cost += min_dist + 1

        return total_cost
