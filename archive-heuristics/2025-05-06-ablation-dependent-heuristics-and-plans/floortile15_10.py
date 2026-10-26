# Ensure necessary imports are at the top
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re
import math

# Helper functions (kept outside the class as they are general utilities)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_coords(tile_name):
    """Parses tile name 'tile_R_C' into (R, C) tuple."""
    # Use regex to extract row and column numbers
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        # Convert captured groups to integers
        return (int(match.group(1)), int(match.group(2)))
    # Return None if the name doesn't match the expected format
    return None

def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two (R, C) coordinate tuples."""
    # Return infinity if either coordinate is invalid
    if coords1 is None or coords2 is None:
        return math.inf
    # Calculate Manhattan distance: |r1 - r2| + |c1 - c2|
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

# Heuristic class definition
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the estimated minimum cost for each unpainted
    goal tile independently. The cost for a single tile includes the paint action,
    the cost for a robot to acquire the correct color, and the minimum movement
    cost for any robot to reach a tile adjacent to the target tile.

    # Assumptions
    - Tiles are arranged in a grid structure, allowing Manhattan distance to approximate movement cost. Tile names are assumed to be in the format 'tile_R_C'.
    - Goal tiles are initially clear or already correctly painted. Tiles painted with the wrong color are not handled (assumed unsolvable or not occurring in valid instances).
    - Robots always hold a color initially.
    - The 'change_color' action is the only way to change the robot's held color.
    - All colors required by the goal are available.
    - The heuristic ignores potential path blocking by other robots or painted tiles.
    - The heuristic ignores the fact that a robot might need to move off a goal tile if it's currently occupying it (assuming goal tiles needing painting are clear).

    # Heuristic Initialization
    - Extract the goal conditions to identify target tiles and their required colors.
    - Store goal locations and colors in a dictionary `self.goal_painted_tiles`.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal facts of the form `(painted tile_X color_Y)` from `self.goals`. Store these target tiles and their required colors in `self.goal_painted_tiles`.
    2. From the current state, identify which of these goal facts are *not* true. These are the "unsatisfied goal tiles".
    3. From the current state, identify the current location (`robot-at`) and held color (`robot-has`) for each robot. Store these in dictionaries `robot_locations` and `robot_colors`.
    4. Initialize the total heuristic value `h = 0`.
    5. For each `tile_X` and its required `color_Y` in the set of unsatisfied goal tiles:
        a. Add 1 to `h` for the paint action itself, which is always required for this tile.
        b. Parse the coordinates of `tile_X` as `(tx, ty)` using `parse_tile_coords`. If parsing fails, this tile is problematic; handle by returning infinity.
        c. Determine the coordinates of the four potential adjacent tiles: `(tx+1, ty)`, `(tx-1, ty)`, `tx, ty+1)`, `(tx, ty-1)`.
        d. Calculate the minimum cost for *any* robot to reach *any* of these adjacent tile coordinates and have the correct color `color_Y`.
            - Initialize `min_robot_cost_for_tile = math.inf`.
            - For each robot `robot_R` identified in step 3:
                - Get its current tile `tile_T` from `robot_locations` and color `color_C` from `robot_colors`. If robot state is incomplete, skip this robot.
                - Parse `tile_T` coordinates as `(rx, ry)` using `parse_tile_coords`. If parsing fails, skip this robot.
                - Calculate the minimum Manhattan distance from `(rx, ry)` to any of the adjacent coordinates `(ax, ay)` determined in step 5c. Let this be `min_dist_to_adj`.
                - Calculate the color change cost for this robot: `cost_color_change = 0` if `robot_color == color_y` else `1`. This assumes the robot needs to change color if it doesn't currently hold the target color.
                - The estimated cost for this robot `R` to get into a position to paint `tile_X` is `min_dist_to_adj + cost_color_change`.
                - Update `min_robot_cost_for_tile = min(min_robot_cost_for_tile, cost_for_robot_R)`.
        e. If `min_robot_cost_for_tile == math.inf` (meaning no robot could reach an adjacent tile), the problem is likely unsolvable from this state; return infinity.
        f. Otherwise, add `min_robot_cost_for_tile` to the total heuristic value `h`. This represents the estimated minimum cost (moves + color change) to get *a* robot ready to paint this specific tile.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal tiles and their required colors.
        # We only care about (painted tile color) goals.
        self.goal_painted_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                # Ensure goal fact has correct number of arguments
                if len(args) == 2:
                    tile, color = args
                    self.goal_painted_tiles[tile] = color
                # else: Ignore malformed painted goal facts


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        # The goal is reached if all facts in self.goals are in the state.
        if self.goals <= state:
             return 0

        # Identify unpainted goal tiles
        unsatisfied_goals = {} # {tile_name: color_name}
        for tile, goal_color in self.goal_painted_tiles.items():
            # Check if the tile is painted with the correct color in the current state
            is_painted_correctly = False
            # Check for the exact goal fact in the state string representation
            if f"(painted {tile} {goal_color})" in state:
                 is_painted_correctly = True

            if not is_painted_correctly:
                unsatisfied_goals[tile] = goal_color

        # If there are no unsatisfied painted goals, but the overall goal wasn't met,
        # this heuristic cannot estimate the remaining cost.
        # Assume painted goals are the primary concern for this heuristic.
        # If unsatisfied_goals is empty, it means all painted goals are met,
        # so the heuristic contribution from painting is 0.
        # The initial `self.goals <= state` check handles the true goal state.
        # If that check fails, and unsatisfied_goals is empty, it implies a problem structure
        # this heuristic doesn't handle (non-painted goals).
        # Returning 0 in this case for the painting part is consistent with the heuristic's scope.


        # Identify robot locations and colors
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}
        robots = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                robots.add(robot)
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                robots.add(robot)
            # Ignore other facts

        total_cost = 0

        # Calculate cost for each unsatisfied goal tile independently
        for tile_x, color_y in unsatisfied_goals.items():
            # Cost for the paint action
            cost_paint = 1

            # Cost to get a robot with the right color adjacent to tile_x
            min_robot_cost_for_tile = math.inf

            tile_x_coords = parse_tile_coords(tile_x)
            if tile_x_coords is None:
                 # If tile name is malformed, this goal is unreachable by coordinate logic
                 return math.inf

            # Determine coordinates of potential adjacent tiles
            tx, ty = tile_x_coords
            potential_adj_coords = [(tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)]

            if not robots:
                 # If no robots exist, no tile can be painted. Problem is unsolvable.
                 return math.inf

            for robot_r in robots:
                robot_tile = robot_locations.get(robot_r)
                robot_color = robot_colors.get(robot_r)

                if robot_tile is None or robot_color is None:
                    # Robot state is incomplete, skip this robot for this calculation
                    continue

                robot_coords = parse_tile_coords(robot_tile)
                if robot_coords is None:
                    # Robot's tile name is malformed
                    continue

                # Calculate minimum distance from this robot to any adjacent coordinate
                min_dist_to_adj = math.inf
                for adj_coords in potential_adj_coords:
                    dist = manhattan_distance(robot_coords, adj_coords)
                    min_dist_to_adj = min(min_dist_to_adj, dist)

                # Calculate color change cost for this robot
                cost_color_change = 0
                if robot_color != color_y:
                    cost_color_change = 1

                # Estimated cost for this robot to get ready to paint tile_x
                # This is the moves needed + color change needed
                cost_for_robot_r = min_dist_to_adj + cost_color_change

                min_robot_cost_for_tile = min(min_robot_cost_for_tile, cost_for_robot_r)

            # Add the minimum cost (move + color) for any robot plus the paint action cost
            if min_robot_cost_for_tile == math.inf:
                 # If no robot could reach an adjacent tile for this specific goal tile,
                 # this goal is likely unreachable. Return a large value.
                 return math.inf
            else:
                 total_cost += min_robot_cost_for_tile + cost_paint

        return total_cost
