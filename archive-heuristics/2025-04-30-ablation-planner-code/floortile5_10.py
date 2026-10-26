from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re # Needed for parsing tile names

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to handle cases where parts might be longer than args (e.g., extra parameters in future domains)
    # or where args might contain wildcards matching multiple parts (though not typical for simple predicates)
    # A simpler check for exact predicate matching is often sufficient:
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all tiles
    that are not currently painted with their goal color. It sums the minimum
    estimated cost for each unpainted goal tile, considering the closest robot
    and the need to change color.

    # Assumptions
    - Tiles are arranged in a grid structure, and movement cost between tiles
      can be estimated using Manhattan distance.
    - Each unpainted goal tile must be painted.
    - Painting a tile requires a robot to be at an adjacent tile, holding the
      correct color, and the tile must be clear (which is assumed for unpainted
      goal tiles in valid problem instances).
    - The cost for each unpainted goal tile is independent and can be assigned
      to the "best" robot for that tile without considering resource conflicts
      (like multiple robots needing the same clear tile to move through or paint from).
    - Robots can change color instantly if the target color is available (always true
      based on the domain definition).

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which color.
    - Parses tile names (e.g., 'tile_row_col') to build a mapping from tile
      name strings to (row, col) integer coordinates, enabling Manhattan distance
      calculations. Tile names are collected from initial state and goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts of the form `(painted tile_X color_Y)`.
    2. Filter these goal facts to find the set of tiles that are *not* currently
       painted with their required goal color in the current state. These are
       the "unpainted goal tiles".
    3. If there are no unpainted goal tiles, the heuristic is 0 (goal state).
    4. For each unpainted goal tile `tile_X` requiring color `color_Y`:
       a. Determine the grid coordinates `(r_X, c_X)` for `tile_X` using the precomputed map.
       b. Find the current location `tile_R` and held color `color_R` for each robot `R` from the current state.
       c. For each robot `R`:
          i. Determine the grid coordinates `(r_R, c_R)` for `tile_R` using the precomputed map.
          ii. Calculate the Manhattan distance `d = abs(r_R - r_X) + abs(c_R - c_X)`
              between the robot's tile and the target tile.
          iii. Estimate the minimum number of move actions for the robot to reach
               *any* tile adjacent to `tile_X`. If the robot is currently on `tile_X`
               (Manhattan distance `d == 0`), it needs 1 move to reach an adjacent tile.
               If the robot is already adjacent to `tile_X` (distance `d == 1`), it needs
               0 moves. If the robot is further away (distance `d > 1`), it needs `d - 1` moves.
               This can be calculated as `(d == 0) + max(0, d - 1)`. Let this be `move_cost`.
          iv. Estimate the cost to get the correct color. If `color_R` is not `color_Y`,
              a `change_color` action is needed (cost 1). Otherwise, cost is 0.
              Let this be `color_cost`.
          v. The estimated cost for robot `R` to paint `tile_X` is `move_cost + color_cost + 1` (for the paint action itself).
       d. The minimum cost to paint `tile_X` is the minimum of `move_cost + color_cost + 1`
          over all robots.
    5. The total heuristic value is the sum of the minimum costs calculated for
       each unpainted goal tile.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and tile coordinates.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations and colors for each tile that needs painting.
        # Mapping: tile_name -> required_color
        self.goal_paintings = {}
        # Assuming goals are a conjunction of simple predicates or a single predicate
        if isinstance(self.goals, frozenset): # Simple set of goals
             goal_list = list(self.goals)
        elif isinstance(self.goals, str) and self.goals.startswith('(and'): # Conjunction
             # Extract individual goals from (and (...)(...)...)
             # This is a simplified parser; a full PDDL parser would be more robust
             goal_list = [g.strip() for g in self.goals[4:-1].strip().split(') (') if g.strip()]
             goal_list = ['(' + g + ')' for g in goal_list] # Add back parentheses
        else: # Single goal predicate
             goal_list = [self.goals]

        for goal in goal_list:
            parts = get_parts(goal)
            if parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Build a mapping from tile name string to (row, col) integer coordinates.
        # Assumes tile names are in the format 'tile_row_col'.
        self.tile_coords = {}

        # Collect all potential tile names from initial state and goal facts
        all_tile_names = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                 all_tile_names.add(parts[1])
                 all_tile_names.add(parts[2])
             elif parts[0] in ['clear', 'painted'] and len(parts) == 2:
                 all_tile_names.add(parts[1])
             elif parts[0] == 'robot-at' and len(parts) == 3:
                 all_tile_names.add(parts[2])

        # Add tiles from goals that might not be in init (e.g., painted in init)
        all_tile_names.update(self.goal_paintings.keys())

        # Regex to extract row and column from tile names like 'tile_3_2'
        tile_name_pattern = re.compile(r'tile_(\d+)_(\d+)')

        for tile_name in all_tile_names:
            match = tile_name_pattern.match(tile_name)
            if match:
                row = int(match.group(1))
                col = int(match.group(2))
                self.tile_coords[tile_name] = (row, col)
            # If a tile name doesn't match the pattern, it won't be added to tile_coords.
            # This is acceptable if only 'tile_row_col' names represent the grid.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Find current robot locations and colors.
        robot_locations = {} # robot_name -> tile_name
        robot_colors = {}    # robot_name -> color_name
        robots = set() # Collect robot names

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                robots.add(robot)
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                robots.add(robot)

        total_cost = 0  # Initialize heuristic cost.

        # Identify unpainted goal tiles.
        unpainted_goal_tiles = {} # tile_name -> required_color
        for goal_tile, required_color in self.goal_paintings.items():
            # Check if the tile is already painted correctly
            is_painted_correctly = False
            # Construct the target painted fact string
            target_painted_fact = f'(painted {goal_tile} {required_color})'
            if target_painted_fact in state:
                 is_painted_correctly = True

            # If not painted correctly, add to the list of tiles to paint
            if not is_painted_correctly:
                 # We assume that if it's not painted correctly, it must be clear
                 # and needs painting. The domain doesn't allow painting over.
                 # So, if (painted tile_X color_Y) is a goal and not in state,
                 # we assume tile_X is clear and needs painting with color_Y.
                 unpainted_goal_tiles[goal_tile] = required_color


        # If all goal tiles are painted correctly, the heuristic is 0.
        if not unpainted_goal_tiles:
            return 0

        # Calculate cost for each unpainted goal tile.
        for target_tile, required_color in unpainted_goal_tiles.items():
            # Skip if target_tile coordinates weren't parsed (e.g., unexpected name format)
            if target_tile not in self.tile_coords:
                 # This tile cannot be handled by the heuristic, potentially return infinity
                 # or a large value, or just skip and rely on other parts of the heuristic.
                 # Skipping assumes it's a rare case or handled by other means.
                 # For simplicity and assuming standard naming, we proceed.
                 # A robust heuristic might return float('inf') here.
                 continue # Skip this tile

            target_r, target_c = self.tile_coords[target_tile]

            min_tile_cost = float('inf') # Minimum cost to paint this specific tile

            for robot in robots:
                robot_tile = robot_locations.get(robot) # Get robot's current tile
                robot_color = robot_colors.get(robot)   # Get robot's current color

                if robot_tile is None or robot_color is None:
                    # This robot's state is incomplete (missing location or color), skip
                    continue

                # Skip if robot_tile coordinates weren't parsed
                if robot_tile not in self.tile_coords:
                     continue # Skip this robot for this tile

                robot_r, robot_c = self.tile_coords[robot_tile]

                # Calculate Manhattan distance
                d = abs(robot_r - target_r) + abs(robot_c - target_c)

                # Calculate movement cost to reach a tile adjacent to the target tile
                # If robot is at target (d=0), needs 1 move to adjacent.
                # If robot is adjacent (d=1), needs 0 moves to adjacent.
                # If robot is further (d>1), needs d-1 moves to adjacent.
                move_cost = (d == 0) + max(0, d - 1)

                # Calculate color change cost
                color_cost = 1 if robot_color != required_color else 0

                # Total cost for this robot to paint this tile
                robot_tile_cost = move_cost + color_cost + 1 # +1 for the paint action

                # Update minimum cost for this tile
                min_tile_cost = min(min_tile_cost, robot_tile_cost)

            # Add the minimum cost for this tile to the total heuristic cost
            # If min_tile_cost is still infinity, it means there are no robots
            # or no robots could reach the tile (e.g., due to unparsed coords).
            # In a valid problem, there will be robots and reachable tiles.
            if min_tile_cost != float('inf'):
                 total_cost += min_tile_cost
            # else: This tile cannot be painted by any known robot.
            # Adding infinity would make the total heuristic infinity, which is
            # appropriate for an unsolvable state. However, for greedy search,
            # any large finite number works. Summing float('inf') results in float('inf').
            # Let's allow it to become infinity if necessary.

        return total_cost

