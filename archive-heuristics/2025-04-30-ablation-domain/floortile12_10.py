# from heuristics.heuristic_base import Heuristic # Assuming this is provided by the environment
import fnmatch
import re

# Assume Heuristic base class is available
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern and the number of parts match, else `False`.
    """
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))


def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_r_c' into (row, col) integers."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen if names are consistent

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        # This indicates an issue with tile naming convention or missing tile.
        # Returning infinity makes states with unparseable/unknown tiles highly undesirable.
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It sums the estimated cost for each unpainted tile
    independently, considering the closest robot and the color needed.

    # Assumptions
    - Tile names follow the format 'tile_r_c' where r and c are integers representing
      row and column, allowing Manhattan distance calculation.
    - Robots can only hold one color at a time.
    - Picking up a color replaces the currently held color (or is only possible if
      no color is held). The cost of getting the correct color is estimated as 1 action (pickup)
      if the robot doesn't already have it.
    - Paint/Repaint action costs 1.
    - Movement between adjacent tiles costs 1, and the minimum movement cost
      between two tiles is their Manhattan distance.
    - The heuristic sums the minimum estimated cost for each unpainted tile,
      considering the closest robot and the color needed for that specific tile.
      This is a relaxation and might overestimate the total cost as it doesn't
      account for a single robot painting multiple tiles efficiently or shared
      pickup actions.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which color.
    - Pre-calculates tile coordinates based on tile names found in goals and
      static adjacency facts for efficient distance calculation. (Assumes tile names are 'tile_r_c').

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted tile color)`. Store these
       as a mapping from tile to required color (`goal_tiles`).
    2. Pre-calculate coordinates for all tiles mentioned in the problem based on
       their names (`tile_r_c`). This is done once during heuristic initialization.
    3. In the `__call__` method, parse the current state to find:
       - The current location of each robot (`robot_locations`).
       - The current color held by each robot (`robot_colors`).
       - The current painted status of tiles (`painted_tiles`).
    4. Identify unsatisfied goals: Create a list of `(tile, color)` pairs from
       `goal_tiles` where the tile is not currently painted with the correct color
       in the state.
    5. Initialize the total heuristic cost `h = 0`.
    6. For each unsatisfied goal `(tile_T, color_C)`:
       - Add 1 to `h` for the paint/repaint action required for `tile_T`.
       - Find the minimum Manhattan distance from any robot's current location
         to `tile_T`. Let this be `min_dist`.
       - Add `min_dist` to `h` for the movement cost.
       - Identify the robot `R_closest` that achieves this `min_dist`.
       - Check if `R_closest` currently holds `color_C`.
       - If `R_closest` does *not* hold `color_C`, add 1 to `h` for the pickup action.
    7. Return the total calculated cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and tile coordinates."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals

        # Store goal tiles and their required colors.
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

        # Extract all tile names from static facts (like up, down, left, right)
        # and goals, and parse their coordinates.
        all_tile_names = set()
        for fact in task.static:
             parts = get_parts(fact)
             # Adjacency predicates involve tiles
             if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                 all_tile_names.add(parts[1])
                 all_tile_names.add(parts[2])

        # Add tiles mentioned in goals
        all_tile_names.update(self.goal_tiles.keys())

        # Map tile names to their (row, col) coordinates
        self.tile_coords = {}
        for tile_name in all_tile_names:
            coords = parse_tile_name(tile_name)
            if coords:
                self.tile_coords[tile_name] = coords
            # Note: If a tile name doesn't match the expected pattern,
            # it won't be added to tile_coords. manhattan_distance will
            # return inf for such tiles, effectively penalizing goals
            # involving them.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Parse current state to find robot locations, colors, and painted tiles.
        robot_locations = {}
        robot_colors = {} # Store color held, or None if no color

        # Collect all robot names first
        robots = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robots.add(parts[1])
            elif parts[0] == "robot-has":
                 robots.add(parts[1])

        # Now parse state facts
        painted_tiles = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == "painted":
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color
            # 'clear' facts indicate the tile is not painted (with goal color or any color)
            # We don't need to store 'clear' explicitly if we rely on 'painted_tiles'
            # to check goal satisfaction.

        # Ensure all robots are in robot_colors dict, defaulting to None if no color fact exists
        for robot in robots:
             if robot not in robot_colors:
                 robot_colors[robot] = None

        # 2. Identify unsatisfied goals
        unsatisfied_goals = []
        for tile_T, color_C in self.goal_tiles.items():
            # Goal is satisfied if tile_T is painted with color_C
            if tile_T in painted_tiles and painted_tiles[tile_T] == color_C:
                continue
            # Otherwise, goal is unsatisfied
            unsatisfied_goals.append((tile_T, color_C))

        # If all goals are satisfied, heuristic is 0
        if not unsatisfied_goals:
            return 0

        # 3. Calculate heuristic cost
        total_cost = 0

        # If there are no robots but unsatisfied goals, the problem is likely unsolvable
        # from this state, return infinity.
        if not robot_locations:
             return float('inf')

        for tile_T, color_C in unsatisfied_goals:
            # Cost for paint/repaint action
            total_cost += 1

            # Find minimum movement cost to get any robot to tile_T
            min_dist = float('inf')
            best_robot = None

            for robot_R, robot_loc in robot_locations.items():
                dist = manhattan_distance(robot_loc, tile_T)
                if dist < min_dist:
                    min_dist = dist
                    best_robot = robot_R

            # If min_dist is still inf, it means tile_T was not found in tile_coords,
            # likely due to a parsing issue or inconsistent naming.
            # This goal is effectively unreachable by movement.
            if min_dist == float('inf'):
                 return float('inf') # Problem likely unsolvable from here

            total_cost += min_dist

            # Cost for pickup action if the best robot doesn't have the color
            # We need to make sure best_robot was actually found (min_dist was not inf)
            if best_robot is not None and robot_colors.get(best_robot) != color_C:
                 total_cost += 1 # Assume 1 action to pick up the required color

        return total_cost
