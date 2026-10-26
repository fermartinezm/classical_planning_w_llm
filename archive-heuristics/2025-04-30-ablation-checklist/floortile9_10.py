from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check is just to zip and compare, fnmatch handles the wildcard logic
    if len(parts) != len(args):
         return False # Mismatch in structure
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_tile_coords(tile_name):
    """
    Parses a tile name like 'tile_row_col' into integer coordinates (row, col).
    Assumes the format is strictly 'tile_R_C' where R and C are integers.
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
        print(f"Warning: Could not parse coordinates from tile name: {tile_name}")
        return None

def manhattan_distance(coords1, coords2):
    """Calculates the Manhattan distance between two coordinate tuples (r1, c1) and (r2, c2)."""
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance if coordinates are invalid
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles that are not yet painted correctly. It considers the cost
    of moving a robot to a position adjacent to the target tile, changing
    the robot's color if necessary, and performing the paint action.

    # Assumptions:
    - Tiles are named in the format 'tile_row_col' allowing coordinate extraction.
    - The grid is traversable (movement actions are possible between adjacent clear tiles).
    - Goal tiles are initially clear or already painted correctly.
    - Robots always hold a color initially.
    - The cost of each action (move, change_color, paint) is 1.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which color.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify all goal tiles that are *not* currently painted with the
       correct color according to the goal state.
    2. For each such unpainted goal tile `T` that needs color `C`:
       a. Determine the coordinates of tile `T`.
       b. For each robot `R`:
          i. Find the current location `LocR` of robot `R`.
          ii. Find the current color `ColorR` held by robot `R`.
          iii. Calculate the minimum number of move actions required for robot `R`
               to reach *any* tile adjacent to `T`. This is `max(0, ManhattanDistance(LocR, T) - 1)`.
               (A robot needs to be adjacent to paint, distance 1 away. If distance is 0, it's on the tile, needs 0 moves to get adjacent. If distance is 1, it's already adjacent, needs 0 moves. If distance > 1, needs dist-1 moves).
          iv. Calculate the cost for robot `R` to have the correct color `C`.
              This is 1 if `ColorR` is not `C`, and 0 if `ColorR` is `C`.
              (Assumes change_color costs 1 and is always possible if robot has *a* color).
          v. The cost for robot `R` to *enable* painting tile `T` is
             (Movement cost) + (Color change cost).
       c. Find the minimum enabling cost among all robots for tile `T`.
       d. Add this minimum enabling cost + 1 (for the paint action) to the total heuristic value.
    3. The total heuristic value is the sum of these costs for all unpainted goal tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are not strictly needed for this heuristic's calculation logic,
        # but we keep the task reference in case needed later.
        self.static_facts = task.static

        # Store goal locations and required colors for each tile.
        # We only care about (painted tile color) goals.
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paintings[tile] = color

        # Pre-calculate tile coordinates for faster lookup
        self.tile_coords = {}
        # We need to find all tiles first. They are listed in the :objects section
        # but we don't have direct access to that here.
        # We can infer tiles from predicates like 'up', 'down', 'left', 'right', 'clear', 'painted', 'robot-at'
        # A robust way is to parse all facts in initial state and static facts.
        all_facts = task.initial_state | task.static
        all_tiles = set()
        for fact in all_facts:
             parts = get_parts(fact)
             if parts[0] in ['up', 'down', 'left', 'right']:
                 # (direction tile1 tile2)
                 all_tiles.add(parts[1])
                 all_tiles.add(parts[2])
             elif parts[0] in ['robot-at', 'clear', 'painted']:
                 # (predicate tile ...)
                 all_tiles.add(parts[1])

        for tile_name in all_tiles:
             coords = get_tile_coords(tile_name)
             if coords is not None:
                 self.tile_coords[tile_name] = coords


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify robot locations and colors
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, location = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['location'] = location
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['color'] = color

        # 2. Identify currently painted tiles and their colors
        current_paintings = {} # {tile_name: color_name}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                current_paintings[tile] = color

        total_cost = 0  # Initialize action cost counter.

        # 3. Iterate through goal tiles and calculate cost for unpainted ones
        for goal_tile, required_color in self.goal_paintings.items():
            # Check if the tile is already painted correctly
            if current_paintings.get(goal_tile) == required_color:
                continue # This goal is already satisfied

            # Tile needs painting. Calculate minimum cost to paint it.
            goal_tile_coords = self.tile_coords.get(goal_tile)
            if goal_tile_coords is None:
                 # Should not happen if init parsed correctly, but handle defensively
                 print(f"Warning: Coordinates not found for goal tile {goal_tile}. Skipping.")
                 continue

            min_robot_cost_for_this_tile = float('inf')

            for robot_name, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                if robot_location is None or robot_color is None:
                    # Robot state is incomplete, cannot use this robot
                    continue

                robot_location_coords = self.tile_coords.get(robot_location)
                if robot_location_coords is None:
                    # Robot is at an unknown location, cannot use this robot
                    continue

                # Calculate movement cost for this robot to get adjacent to the goal tile
                dist_to_tile = manhattan_distance(robot_location_coords, goal_tile_coords)
                # Need to reach a tile adjacent to the goal tile.
                # If dist_to_tile is 0 (on the tile), need 0 moves to get adjacent (e.g., move up/down/left/right).
                # If dist_to_tile is 1 (already adjacent), need 0 moves.
                # If dist_to_tile > 1, need dist_to_tile - 1 moves to reach an adjacent tile.
                movement_cost = max(0, dist_to_tile - 1)

                # Calculate color change cost for this robot
                color_cost = 0
                if robot_color != required_color:
                    color_cost = 1 # Cost to change color

                # Total cost for this robot to enable painting this tile
                robot_cost = movement_cost + color_cost

                # Update minimum cost for this tile
                min_robot_cost_for_this_tile = min(min_robot_cost_for_this_tile, robot_cost)

            # Add the minimum enabling cost + paint action cost (1) to the total
            if min_robot_cost_for_this_tile != float('inf'):
                 total_cost += min_robot_cost_for_this_tile + 1
            # else: No robot can reach this tile? This implies an unsolvable problem or a grid parsing issue.
            # For heuristic purposes, we might return inf or a very large number,
            # but assuming solvable problems, at least one robot should be reachable.


        return total_cost

