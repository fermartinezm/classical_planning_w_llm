from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for infinity

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(painted tile_1_1 white)" -> ["painted", "tile_1_1", "white"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(robot-at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper functions for grid geometry based on tile names
def parse_tile_name(tile_str):
    """
    Parses a tile name string like 'tile_row_col' into a (row, col) tuple.
    Assumes tile names follow the format 'tile_<row>_<col>'.
    """
    try:
        parts = tile_str.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return row, col
        else:
            # Handle unexpected format, though problem instances should be consistent
            raise ValueError(f"Unexpected tile name format: {tile_str}")
    except (ValueError, IndexError):
        # Handle parsing errors
        raise ValueError(f"Could not parse tile name: {tile_str}")

def manhattan_distance(r1, c1, r2, c2):
    """Calculates the Manhattan distance between two grid coordinates."""
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal
    tiles with the correct color. It sums the estimated cost for each unpainted
    goal tile independently. The cost for a single tile is estimated as:
    1 (paint action) + minimum cost for any robot to get ready to paint it.
    The cost for a robot to get ready includes:
    - 1 action if the robot needs to change color to the target color.
    - Manhattan distance from the robot's current location to any tile
      from which the target tile can be painted (based on adjacency predicates).

    # Assumptions
    - Tile names follow the format 'tile_row_col'.
    - The grid is connected as defined by the adjacency predicates.
    - All goal tiles are initially 'clear' or painted with the wrong color
      (if painted with the wrong color, the problem is likely unsolvable,
       we assume valid solvable instances where unpainted goal tiles are clear).
    - The 'clear' constraint for movement paths is ignored for distance calculation
      (Manhattan distance on the full grid).
    - Multiple robots work independently on different tiles (no coordination cost modeled).

    # Heuristic Initialization
    - Extracts goal conditions.
    - Builds a map of which tiles can be painted from which adjacent tiles,
      based on the static adjacency predicates (up, down, left, right).

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all goal predicates of the form `(painted ?tile ?color)`.
    2. Filter these goals to find the set of tiles that are *not* currently
       painted with the required color in the given state. These are the
       `unpainted_goals`.
    3. If there are no `unpainted_goals`, the heuristic value is 0.
    4. For each robot, find its current location and the color it is holding.
    5. Initialize the total heuristic value to 0.
    6. For each `(target_tile, target_color)` in `unpainted_goals`:
       a. Initialize `min_cost_for_tile` to infinity.
       b. For each robot:
          i. Get the robot's current location (`robot_loc`) and color (`robot_color`).
          ii. Calculate the cost to change color: 1 if `robot_color` is not `target_color`, else 0.
          iii. Calculate the minimum movement cost for this robot to reach *any* tile
               from which `target_tile` can be painted. This is the minimum Manhattan
               distance from `robot_loc` to any tile `x` such that `(relation target_tile x)`
               exists in the static facts (precomputed in `paintable_from_map`).
          iv. The cost for this robot to paint `target_tile` is (movement cost + color change cost).
          v. Update `min_cost_for_tile` with the minimum cost found so far across all robots.
       c. Add the cost for painting this tile to the total heuristic:
          `total_heuristic += 1  # Cost of the paint action itself`
          `total_heuristic += min_cost_for_tile`
    7. Return `total_heuristic`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the map of paintable tiles from static facts.
        """
        self.goals = task.goals  # Goal conditions (frozenset of strings)

        # Build a map: target_tile -> list of tiles a robot can be at to paint target_tile
        self.paintable_from_map = self._build_paintable_map(task.static)

    def _build_paintable_map(self, static_facts):
        """
        Parses static facts to build a map indicating which tiles can be
        painted from which adjacent tiles.
        Based on actions like `paint_up ?r ?y ?x ?c` requiring `(up ?y ?x)`
        and `(robot-at ?r ?x)`. This means tile `y` can be painted if robot is at `x`.
        So, for predicate `(relation y x)`, tile `y` is paintable from tile `x`.
        """
        paintable_map = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Check for adjacency predicates with 3 parts: (relation tile1 tile2)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                relation, tile1, tile2 = parts
                # In (relation tile1 tile2), if robot is at tile2, it can paint tile1.
                # So, tile1 is paintable from tile2.
                target_tile = tile1
                robot_location_to_paint_from = tile2

                if target_tile not in paintable_map:
                    paintable_map[target_tile] = []
                # Add the tile from which painting is possible
                paintable_map[target_tile].append(robot_location_to_paint_from)

        # Remove duplicates from lists (though unlikely with standard grids)
        for tile in paintable_map:
            paintable_map[tile] = list(set(paintable_map[tile]))

        return paintable_map

    def _min_manhattan_dist_to_paint_location(self, from_tile_str, target_tile_str):
        """
        Calculates the minimum Manhattan distance from `from_tile_str` to any
        tile from which `target_tile_str` can be painted.
        """
        # Get the list of possible robot locations to paint the target tile
        possible_robot_locations = self.paintable_from_map.get(target_tile_str, [])

        if not possible_robot_locations:
            # This tile cannot be painted according to the static map.
            # This indicates an issue with the problem definition or parsing,
            # but we return infinity to make this state seem undesirable.
            return float('inf')

        # Parse the starting tile coordinates
        try:
            fr, fc = parse_tile_name(from_tile_str)
        except ValueError:
            return float('inf') # Handle parsing error

        min_dist = float('inf')
        # Calculate distance to each possible painting location and find the minimum
        for robot_loc_str in possible_robot_locations:
            try:
                rr, rc = parse_tile_name(robot_loc_str)
                dist = manhattan_distance(fr, fc, rr, rc)
                min_dist = min(min_dist, dist)
            except ValueError:
                continue # Skip if a paintable location name is unparseable

        return min_dist

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state  # Current world state (frozenset of strings)

        # 1. Identify unpainted goal tiles
        unpainted_goals = [] # List of (tile_name, color_name) tuples
        for goal in self.goals:
            # Goal predicates are expected to be like '(painted tile_x_y color)'
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == 'painted':
                target_tile, target_color = parts[1], parts[2]
                # Check if the tile is NOT painted with the target color in the current state
                # It could be clear or painted with a different color.
                # We assume valid problems don't require repainting a tile painted with the wrong color.
                # So, we just check if the exact goal fact is missing.
                if goal not in state:
                     unpainted_goals.append((target_tile, target_color))

        # If all goal tiles are painted correctly, the heuristic is 0
        if not unpainted_goals:
            return 0

        # 2. Find robot locations and colors in the current state
        robot_info = {} # {robot_name: {'loc': tile_str, 'color': color_str}}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3:
                if parts[0] == 'robot-at':
                    robot_name, loc = parts[1], parts[2]
                    if robot_name not in robot_info:
                        robot_info[robot_name] = {}
                    robot_info[robot_name]['loc'] = loc
                elif parts[0] == 'robot-has':
                    robot_name, color = parts[1], parts[2]
                    if robot_name not in robot_info:
                        robot_info[robot_name] = {}
                    robot_info[robot_name]['color'] = color

        # If there are unpainted goals but no robots, the problem is unsolvable
        # Return infinity or a very large number.
        if not robot_info:
             return float('inf')

        # 3. Calculate total heuristic cost
        total_heuristic = 0

        for target_tile, target_color in unpainted_goals:
            min_cost_for_tile = float('inf') # Minimum cost for *any* robot to paint this tile

            for robot_name, info in robot_info.items():
                robot_loc = info.get('loc')
                robot_color = info.get('color')

                # A robot must have a location and a color to be useful
                if robot_loc is None or robot_color is None:
                     continue # Skip this robot if info is incomplete

                # Calculate movement cost to a tile adjacent to the target tile
                movement_cost = self._min_manhattan_dist_to_paint_location(robot_loc, target_tile)

                # Calculate color change cost
                color_change_cost = 1 if robot_color != target_color else 0

                # Total cost for this specific robot to paint this specific tile
                cost_for_this_robot = movement_cost + color_change_cost

                # Update the minimum cost for this tile across all robots
                min_cost_for_tile = min(min_cost_for_tile, cost_for_this_robot)

            # If min_cost_for_tile is still infinity, it means no robot can reach a paintable location
            # for this tile, which shouldn't happen in a connected grid with robots.
            # Add the cost for this tile: 1 (the paint action itself) + the minimum robot preparation cost
            if min_cost_for_tile != float('inf'):
                 total_heuristic += 1 + min_cost_for_tile
            else:
                 # If a tile is unreachable, the problem is likely unsolvable from this state
                 return float('inf')


        return total_heuristic

