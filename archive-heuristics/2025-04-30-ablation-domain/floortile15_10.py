from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For infinity

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint
    all tiles that are not currently painted with their goal color. It sums
    the minimum estimated cost for each individual unpainted goal tile,
    considering the cost for the closest robot to move to the tile,
    acquire the correct color, and paint it.

    # Assumptions
    - The environment is a grid where tiles are connected by up/down/left/right relations.
    - Tile names follow the format 'tile_row_col', allowing coordinate extraction.
    - Movement cost between adjacent tiles is 1 (Manhattan distance is used as a lower bound).
    - Getting a color costs 1 action (if the robot has no color).
    - Switching color costs 2 actions (drop current color + get new color).
    - Painting a tile costs 1 action.
    - Robots can acquire any available color at their current location.
    - The heuristic ignores potential conflicts (e.g., multiple robots wanting the same tile, robots blocking paths, clear predicate constraints). It is non-admissible.

    # Heuristic Initialization
    - The goal conditions are stored to identify target colors for tiles.
    - Tile coordinates (row, col) are extracted from tile names found in static facts
      defining the grid structure (up, down, left, right relations) and initial robot locations.
    - All robot names are identified from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unpainted Goal Tiles:
       Iterate through the goal conditions. For each goal fact `(painted tile color)`,
       check if this exact fact is present in the current state. If not, this tile
       needs to be painted with the target color. Collect all such `(tile, color)` pairs.

    2. Initialize Total Heuristic Cost:
       Set `total_heuristic = 0`.

    3. Calculate Cost for Each Unpainted Goal Tile:
       For each `(tile, target_color)` pair identified in step 1:
       a. Initialize `min_cost_for_tile = infinity`. This will track the minimum cost
          for *any* robot to paint this specific tile.
       b. Get the coordinates `(target_row, target_col)` for the target `tile` using the precomputed map.
       c. For each robot `R`:
          i. Find the robot's current tile `robot_current_tile` from the current state.
          ii. Get the coordinates `(robot_row, robot_col)` for the robot's current tile.
          iii. Calculate the Manhattan distance (move cost) between the robot's current
              tile and the target tile: `dist = abs(robot_row - target_row) + abs(robot_col - target_col)`.
          iv. Find the color `robot_current_color` the robot is currently holding (if any) from the current state.
          v. Calculate the color preparation cost:
              - If `robot_current_color` is the same as `target_color`: `color_prep_cost = 0`.
              - If `robot_current_color` is different from `target_color`:
                  - If the robot has *any* color (`robot_current_color is not None`): `color_prep_cost = 2` (1 for drop, 1 for get).
                  - If the robot has *no* color (`robot_current_color is None`): `color_prep_cost = 1` (1 for get).
          vi. Calculate the total cost for this specific robot to paint this specific tile:
             `cost_for_this_robot = dist + color_prep_cost + 1` (move + color prep + paint).
          vii. Update `min_cost_for_tile = min(min_cost_for_tile, cost_for_this_robot)`.
       d. Add `min_cost_for_tile` to `total_heuristic`.

    4. Return Total Heuristic Cost:
       The final `total_heuristic` is the estimated number of actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, tile coordinates,
        and robot names.
        """
        self.goals = task.goals  # Goal conditions (frozenset of facts)

        # Extract tile names from static facts defining the grid and initial robot locations
        all_tile_names = set()
        for fact in task.static:
            parts = get_parts(fact)
            # Facts like (up t1 t2), (down t1 t2), (left t1 t2), (right t1 t2) define grid
            if len(parts) == 3 and parts[0] in {'up', 'down', 'left', 'right'}:
                all_tile_names.add(parts[1])
                all_tile_names.add(parts[2])
        # Add tiles mentioned in initial state (e.g., robot-at)
        for fact in task.initial_state:
             if match(fact, "robot-at", "*", "*"):
                 all_tile_names.add(get_parts(fact)[2])
        # Add tiles mentioned in goals (e.g., painted)
        for fact in task.goals:
             if match(fact, "painted", "*", "*"):
                 all_tile_names.add(get_parts(fact)[1])


        # Parse tile coordinates (row, col) from tile names
        self.tile_coords = {}
        for tile_name in all_tile_names:
            parts = tile_name.split('_')
            if len(parts) == 3 and parts[0] == 'tile':
                try:
                    row = int(parts[1])
                    col = int(parts[2])
                    self.tile_coords[tile_name] = (row, col)
                except ValueError:
                    # Ignore tile names that don't fit the tile_row_col format
                    pass

        # Identify all robots from the initial state
        self.robots = set()
        for fact in task.initial_state:
            if match(fact, "robot-at", "*", "*"):
                self.robots.add(get_parts(fact)[1])

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify Unpainted Goal Tiles
        needed_paintings = set() # Stores (tile, target_color) pairs
        for goal_fact in self.goals:
            if match(goal_fact, "painted", "*", "*"):
                # Check if the specific goal fact is NOT in the current state
                if goal_fact not in state:
                    tile, target_color = get_parts(goal_fact)[1:]
                    needed_paintings.add((tile, target_color))

        # If all goal tiles are painted correctly, the heuristic is 0
        if not needed_paintings:
            return 0

        # Extract current robot locations and colors from the state
        robot_locations = {}
        robot_colors = {} # None if robot has no color
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, tile = get_parts(fact)[1:]
                robot_locations[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                robot, color = get_parts(fact)[1:]
                robot_colors[robot] = color

        # 2. Initialize Total Heuristic Cost
        total_heuristic = 0

        # 3. Calculate Cost for Each Unpainted Goal Tile
        for tile, target_color in needed_paintings:
            min_cost_for_tile = float('inf')

            # Get target tile coordinates
            target_coords = self.tile_coords.get(tile)
            if target_coords is None:
                 # This tile exists in goals but not in the grid structure parsed.
                 # Problem might be unsolvable or malformed. Treat as infinite cost.
                 total_heuristic += float('inf')
                 continue

            # Consider each robot as a potential candidate for painting this tile
            for robot in self.robots:
                robot_current_tile = robot_locations.get(robot)
                if robot_current_tile is None:
                    # Robot is not on a tile? Should not happen in this domain.
                    # Treat as unable to paint this tile.
                    continue

                robot_coords = self.tile_coords.get(robot_current_tile)
                if robot_coords is None:
                    # Robot is on a tile not in the parsed grid.
                    continue # Treat as unable to paint this tile.

                # Calculate move cost (Manhattan distance)
                dist = abs(robot_coords[0] - target_coords[0]) + abs(robot_coords[1] - target_coords[1])

                # Get robot's current color
                current_color = robot_colors.get(robot) # Returns None if robot_has fact is missing

                # Calculate color preparation cost at the target tile
                color_prep_cost = 0
                if current_color != target_color:
                    if current_color is not None:
                        color_prep_cost += 1 # drop current color
                    color_prep_cost += 1 # get target color

                # Total cost for this robot to paint this tile
                # move_cost + color_prep_cost + paint_cost (1)
                cost_for_this_robot = dist + color_prep_cost + 1

                # Update minimum cost for this tile
                min_cost_for_tile = min(min_cost_for_tile, cost_for_this_robot)

            # Add the minimum cost required to paint this tile to the total heuristic
            total_heuristic += min_cost_for_tile

        # 4. Return Total Heuristic Cost
        return total_heuristic
