from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check is just element-wise matching with fnmatch
    if len(parts) != len(args):
         return False # Cannot match if different number of elements

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """
    Parses a tile name like 'tile_R_C' into a (row, column) tuple of integers.
    Assumes tile names follow the format 'tile_row_col'.
    """
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected format, maybe return None or raise error
            print(f"Warning: Unexpected tile name format: {tile_name}")
            return None
    except ValueError:
        print(f"Warning: Could not parse row/col from tile name: {tile_name}")
        return None

def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles given their names.
    Returns infinity if parsing fails for either tile.
    """
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)

    if coords1 is None or coords2 is None:
        return math.inf # Cannot calculate distance if parsing fails

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles with the correct color. It sums the estimated minimum cost
    for each unpainted goal tile independently.

    # Heuristic Initialization
    - Extracts the goal conditions (which tiles need which color).
    - Parses static facts to build a mapping from each tile to the set
      of adjacent tiles from which a robot can paint it.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For each tile T that needs to be painted with color C according to the goal,
    and is *not* currently painted with color C:

    1.  Identify the target tile T and the required color C from the goal fact.
    2.  Check if the state already satisfies `(painted T C)`. If yes, cost for this tile is 0.
    3.  If not, find the minimum cost to paint tile T with color C:
        a.  Iterate through all robots R.
        b.  For each robot R, find its current location R_loc and current color R_color.
        c.  Calculate the cost for R to change color to C: 1 if R_color is not C, else 0.
        d.  Identify the set of tiles Paint_Locs from which T can be painted (these are tiles X such that (up T X), (down T X), (left T X), or (right T X) is true). This information is precomputed during initialization.
        e.  Calculate the minimum Manhattan distance from R_loc to any tile in Paint_Locs. This is the minimum number of move actions required for R to get into a position to paint T.
        f.  The total estimated cost for robot R to paint tile T is:
            (Minimum distance to a paint location) + (Color change cost) + 1 (Paint action).
        g.  The minimum cost for tile T is the minimum of the costs calculated for each robot R.
    4.  The total heuristic value is the sum of the minimum costs calculated for each unpainted goal tile.

    This heuristic is non-admissible as it ignores potential conflicts (multiple robots needing the same tile or color)
    and synergies (one robot painting multiple nearby tiles) and assumes free movement (Manhattan distance).
    However, it captures the essential components of the task: robot location, required color, and the need for a paint action.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Builds a map from a tile to the tiles adjacent to it from which it can be painted.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build a map from a tile Y to the set of tiles X such that a robot at X
        # can paint Y using one of the paint actions.
        # paint_up ?r ?y ?x ?c requires robot-at ?r ?x and up ?y ?x
        # paint_down ?r ?y ?x ?c requires robot-at ?r ?x and down ?y ?x
        # etc.
        # So, if (up Y X) is true, a robot at X can paint Y.
        self.paintable_from = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                painted_tile = parts[1] # This is ?y in the action
                robot_tile = parts[2]   # This is ?x in the action
                if painted_tile not in self.paintable_from:
                    self.paintable_from[painted_tile] = set()
                self.paintable_from[painted_tile].add(robot_tile)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        total_heuristic = 0  # Initialize action cost counter.

        # Iterate through goal facts to find unpainted goal tiles
        for goal_fact in self.goals:
            # We only care about (painted ?tile ?color) goals
            if not match(goal_fact, "painted", "*", "*"):
                continue # Ignore other types of goals if any

            # Check if the goal fact is already satisfied in the current state
            if goal_fact in state:
                continue # This tile is already painted correctly

            # Goal fact is not satisfied: (painted T C) is required, but not in state
            # Find the tile T and color C from the goal fact
            parts = get_parts(goal_fact)
            tile_to_paint = parts[1]
            required_color = parts[2]

            min_cost_for_tile = math.inf # Minimum cost to paint this specific tile

            # Consider each robot
            for robot, robot_loc in robot_locations.items():
                robot_color = robot_colors.get(robot) # Get robot's current color

                # Cost to change color if needed
                color_change_cost = 0
                # Check if robot has a color and if it's the wrong color
                if robot_color is None or robot_color != required_color:
                     # Assumes a robot without a color needs 1 action to get the right one
                     # and a robot with the wrong color needs 1 action to change
                     color_change_cost = 1

                # Find minimum distance from robot's current location to any valid paint location for this tile
                min_dist_to_paint_loc = math.inf
                possible_paint_locations = self.paintable_from.get(tile_to_paint, set())

                if not possible_paint_locations:
                    # This tile cannot be painted from any adjacent tile based on static facts
                    # This might indicate an unreachable goal or a problem definition issue
                    # We treat this as infinite cost for this tile, but don't add infinity
                    # to the total unless a path is found.
                    continue # Skip this tile for this robot, min_cost_for_tile remains inf if no robot can paint it

                for paint_loc in possible_paint_locations:
                    dist = manhattan_distance(robot_loc, paint_loc)
                    min_dist_to_paint_loc = min(min_dist_to_paint_loc, dist)

                # If a paint location was found (min_dist_to_paint_loc is not inf)
                if min_dist_to_paint_loc != math.inf:
                    # Cost for this robot to paint this tile:
                    # moves + color change + paint action
                    cost_for_robot = min_dist_to_paint_loc + color_change_cost + 1
                    min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)

            # Add the minimum cost found for this tile (across all robots) to the total heuristic
            if min_cost_for_tile != math.inf:
                total_heuristic += min_cost_for_tile
            # Note: If min_cost_for_tile remains infinity, it means no robot can reach
            # a valid paint location for this tile. This tile won't contribute to the
            # heuristic, which is acceptable for a non-admissible heuristic, although
            # in a real problem, this might imply the goal is unreachable.

        return total_heuristic

