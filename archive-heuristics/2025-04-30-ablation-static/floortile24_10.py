from fnmatch import fnmatch
# Assuming heuristics.heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic
# Define a mock Heuristic base class if running standalone for testing
# In the actual environment, this import will work.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a mock class for testing purposes
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError

import re
import math # For float('inf')

# Helper function to extract parts from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise error
        # For robustness, return empty list or handle specific cases
        # Assuming valid PDDL fact strings like "(predicate arg1 arg2)"
        return [] # Or raise ValueError("Invalid fact format")

    # Remove outer parentheses and split by whitespace
    # Use shlex.split for more robust parsing if arguments can contain spaces or quotes,
    # but standard PDDL facts use simple space separation.
    content = fact[1:-1].strip()
    if not content:
        return []
    return content.split()

# Helper function to match a PDDL fact against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)

    # The number of parts in the fact must match the number of arguments in the pattern
    if len(parts) != len(args):
        return False

    # Check if each part matches the corresponding pattern argument (with wildcard support)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to parse tile name into coordinates
def parse_tile_name(tile_name):
    """Parses 'tile_r_c' into (r, c) tuple of integers."""
    # Use regex to extract row and column numbers
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        # Convert extracted strings to integers
        return int(match.group(1)), int(match.group(2))
    else:
        # If the tile name format is unexpected, raise an error.
        # This indicates a potential issue with the problem definition or assumption.
        # For a heuristic, we might return None or a default, but raising helps debug problem files.
        raise ValueError(f"Could not parse tile name into row/column: {tile_name}")

# Helper function for Manhattan distance
def manhattan_distance(tile1_name, tile2_name, coords):
    """Calculates Manhattan distance between two tiles using their coordinates."""
    # Get coordinates for both tiles
    coord1 = coords.get(tile1_name)
    coord2 = coords.get(tile2_name)

    # Ensure both tiles exist in the coordinate map
    if coord1 is None or coord2 is None:
        # This indicates an issue with the grid parsing or an unexpected tile name.
        # Returning infinity signifies that these tiles are unreachable or invalid for distance calculation.
        return float('inf')

    r1, c1 = coord1
    r2, c2 = coord2

    # Calculate Manhattan distance
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the estimated cost for each goal tile
    that is not yet painted correctly. The cost for a single tile includes:
    1. Cost to move a robot off the tile if it's currently occupied.
    2. Cost of the paint action itself.
    3. Minimum cost for any robot to reach a tile adjacent to the target tile
       and have the correct color.

    # Assumptions
    - The grid structure is defined by `up`, `down`, `left`, `right` predicates
      connecting tiles named `tile_r_c`, where `r` and `c` are integers representing
      row and column.
    - Every tile that is a goal can be painted from at least one adjacent tile
      defined by the static adjacency facts.
    - States where a goal tile is painted with the wrong color are considered
      unsolvable (heuristic returns infinity).
    - Robots can change color anywhere (the `change_color` action has no location precondition).
    - Manhattan distance is a reasonable estimate for movement cost on the grid,
      ignoring obstacles (other robots or un-clear tiles on the path).

    # Heuristic Initialization
    - Parses the goal conditions to identify which tiles need to be painted
      and with which colors.
    - Parses static facts (`up`, `down`, `left`, `right`) to build:
        - A mapping from tile name to (row, column) coordinates for distance calculation.
        - A mapping from tile name to the set of tiles from which it can be painted.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location and color of each robot by scanning the state facts.
    2. Initialize the total heuristic cost to 0.
    3. Check if there are any robots. If there are goal tiles but no robots, the problem
       is unsolvable unless all goals are already met. Return infinity if unsolvable.
    4. For each tile specified in the goal conditions (`goal_tile`) and its target color (`target_color`):
        a. Check the current state of `goal_tile` (painted correctly, painted wrong, clear, or occupied).
        b. If `goal_tile` is already painted with `target_color`, this goal is met; continue to the next goal tile.
        c. If `goal_tile` is painted with a *different* color, the state is likely unsolvable; return infinity.
        d. If `goal_tile` is not painted correctly (i.e., it's clear or occupied by a robot):
            i. Initialize the cost for this specific tile (`tile_cost`) to 0.
            ii. If `goal_tile` is *not* currently `clear` (and not painted correctly), it must be occupied by a robot. Add 1 to `tile_cost` (representing the cost for the occupying robot to move off).
            iii. Add 1 to `tile_cost` for the `paint` action itself.
            iv. Retrieve the set of tiles (`PaintFromTiles`) from which this `goal_tile` can be painted (pre-calculated in `__init__`). If this set is empty, the tile is unpaintable; return infinity.
            v. Find the minimum "preparation cost" among all robots to paint this tile. For each robot:
                - Calculate the minimum Manhattan distance from the robot's current location to any tile in `PaintFromTiles`. If the robot's location or any `PaintFromTile` is not in the coordinate map, this path is invalid; consider its cost infinity for this robot.
                - Add 1 to this distance if the robot does not currently have the `target_color` (for a `change_color` action).
                - The preparation cost for this robot is the sum of the minimum distance and the color change cost.
            vi. The overall minimum robot preparation cost for this `goal_tile` is the minimum of the costs calculated in step v across all robots. If no robot can reach a `PaintFromTile` (e.g., due to grid issues), return infinity.
            vii. Add this minimum robot preparation cost to `tile_cost`.
            viii. Add `tile_cost` to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and grid structure.
        """
        # Store goal locations and target colors for each tile.
        self.goal_tiles = {} # Map from tile name to target color
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                # Ensure the goal fact has the expected structure (painted tile color)
                if len(args) == 2:
                    tile, color = args
                    self.goal_tiles[tile] = color
                # else: Ignore malformed goal facts? Or raise error? Assuming valid goals.


        # Build grid structure and paint_from mapping from static facts.
        self.coords = {} # Map from tile name to (row, col)
        self.tile_names = {} # Map from (row, col) to tile name
        self.paint_from = {} # Map from tile name to set of tiles it can be painted from

        # First pass to get all tile names and their coordinates from adjacency facts
        all_tiles = set()
        for fact in task.static:
            parts = get_parts(fact)
            # Check for adjacency facts like (direction tile1 tile2)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                 # Fact is (direction tile_y tile_x) meaning tile_y is direction of tile_x
                 # A robot at tile_x can paint tile_y
                 tile_y, tile_x = parts[1], parts[2]
                 all_tiles.add(tile_x)
                 all_tiles.add(tile_y)

        # Populate coords and tile_names maps
        for tile_name in all_tiles:
             try:
                 r, c = parse_tile_name(tile_name)
                 self.coords[tile_name] = (r, c)
                 self.tile_names[(r, c)] = tile_name
                 # Initialize paint_from set for all tiles found in adjacency facts
                 self.paint_from[tile_name] = set()
             except ValueError:
                 # If a tile name doesn't match the expected format, it won't be in coords.
                 # This might cause issues later if such a tile is a robot location or goal.
                 # Assuming valid tile names in problem instances.
                 pass


        # Second pass to build paint_from mapping
        for fact in task.static:
            parts = get_parts(fact)
            # Check for adjacency facts like (direction tile_y tile_x)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                 # Fact is (direction tile_y tile_x)
                 # Robot at tile_x can paint tile_y using paint_direction action
                 tile_y, tile_x = parts[1], parts[2]
                 # Ensure tile_y is a tile we care about (e.g., potentially a goal tile)
                 # and tile_x is a valid tile location in our grid map.
                 if tile_y in self.paint_from and tile_x in self.coords:
                     self.paint_from[tile_y].add(tile_x)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Get robot locations and colors from the current state
        robot_locations = {} # Map robot name to tile name
        robot_colors = {} # Map robot name to color name
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif len(parts) == 2 and parts[0] == "robot-has":
                 robot, color = parts[1], parts[2]
                 robot_colors[robot] = color

        total_cost = 0  # Initialize action cost counter.

        # If there are goal tiles but no robots, the problem is unsolvable unless all goals are already met.
        if not robot_locations and self.goal_tiles:
             # Check if all goals are already met
             all_goals_met = True
             for goal_tile, target_color in self.goal_tiles.items():
                 if f"(painted {goal_tile} {target_color})" not in state:
                     all_goals_met = False
                     break
             if not all_goals_met:
                 return float('inf')


        # Iterate through each goal tile and its target color
        for goal_tile, target_color in self.goal_tiles.items():
            is_painted_correctly = False
            is_painted_wrong = False
            is_clear = False

            # Check the current state of the goal tile
            # Iterate through state facts only once for this tile
            for fact in state:
                parts = get_parts(fact)
                if len(parts) >= 2 and parts[1] == goal_tile: # Check facts involving this tile
                    if parts[0] == "painted":
                        if len(parts) == 3 and parts[2] == target_color:
                            is_painted_correctly = True
                            break # Found correct painting, no need to check further for this tile state
                        elif len(parts) == 3: # Painted with some color, but not the target
                            is_painted_wrong = True
                            break # Found wrong painting, no need to check further
                    elif parts[0] == "clear":
                         if len(parts) == 2:
                             is_clear = True
                             # Don't break, still need to check for painted status

            # If the goal is already met, continue to the next goal tile
            if is_painted_correctly:
                continue

            # If the tile is painted with the wrong color, the state is unsolvable
            if is_painted_wrong:
                return float('inf')

            # The tile is not painted correctly (it's either clear or occupied by a robot)
            tile_cost = 0

            # Cost to make the tile clear if it's currently occupied by a robot
            # If it's not clear and not painted (correctly or wrongly), it must be occupied.
            # Add 1 action cost for the robot to move off this tile.
            if not is_clear:
                 tile_cost += 1

            # Cost for the paint action itself
            tile_cost += 1

            # Cost for a robot to get ready (move to a paint_from tile + change color)
            min_robot_prep_cost = float('inf')
            paint_from_tiles = self.paint_from.get(goal_tile, set())

            # If there are no tiles from which this goal tile can be painted, it's unsolvable.
            if not paint_from_tiles:
                 return float('inf')

            # Calculate the minimum preparation cost over all robots
            for robot_name, robot_tile in robot_locations.items():
                # Ensure the robot's current tile is in our coordinate map
                if robot_tile not in self.coords:
                    # This indicates an issue with the problem definition or grid parsing.
                    # A robot at an unmapped tile cannot move correctly. Unsolvable.
                    # Return infinity for this robot's prep cost, it won't be chosen as min unless no valid robots exist.
                    robot_prep_cost = float('inf')
                else:
                    min_move_dist = float('inf')
                    # Find the minimum distance from the robot's current tile to any paint_from tile
                    for paint_tile in paint_from_tiles:
                        # Ensure the paint_from tile is in our coordinate map
                        if paint_tile in self.coords:
                            dist = manhattan_distance(robot_tile, paint_tile, self.coords)
                            min_move_dist = min(min_move_dist, dist)
                        # else: Ignore paint_from tiles that are not in our coordinate map.
                        # This shouldn't happen if paint_from is built correctly from coords.

                    # If min_move_dist is still inf, it means none of the paint_from tiles
                    # for this goal tile were found in the coordinate map, or the robot's tile wasn't.
                    # This robot cannot reach a paintable location for this tile.
                    if min_move_dist == float('inf'):
                         robot_prep_cost = float('inf')
                    else:
                        # Calculate the cost to change color if needed
                        robot_current_color = robot_colors.get(robot_name)
                        color_change_cost = 1 if robot_current_color != target_color else 0

                        # Total preparation cost for this robot for this goal tile
                        robot_prep_cost = min_move_dist + color_change_cost

                # Update the minimum preparation cost over all robots
                min_robot_prep_cost = min(min_robot_prep_cost, robot_prep_cost)

            # If min_robot_prep_cost is still inf, it means no robot could reach
            # any paint_from tile (e.g., no robots exist, or grid is disconnected).
            # If no robots exist, we already returned inf earlier if goals exist.
            # If grid is disconnected such that no robot can reach any paint_from tile, it's unsolvable.
            if min_robot_prep_cost == float('inf'):
                 return float('inf')


            # Add the minimum robot preparation cost to the tile's total cost
            tile_cost += min_robot_prep_cost

            # Add the cost for this tile to the overall total cost
            total_cost += tile_cost

        # Return the total estimated cost
        return total_cost
