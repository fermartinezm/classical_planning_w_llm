from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

# Define helper functions outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 city1-1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_tile_coords(tile_name):
    """Parses tile name 'tile_r_c' into (row, col) tuple."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen for valid tile names

def manhattan_distance(coord1, coord2):
    """Calculates Manhattan distance between two (row, col) coordinates."""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

# Define the heuristic class
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total cost to paint all required tiles
    by summing the minimum estimated cost for each individual tile that
    needs to be painted according to the goal. The minimum cost for a
    single tile is estimated by considering the closest robot, the cost
    for that robot to acquire the correct color, the Manhattan distance
    for that robot to reach an adjacent tile, and the paint action itself.
    It also includes a cost if the tile to be painted is currently occupied
    by a robot.

    # Assumptions
    - Tiles are arranged in a grid, and tile names follow the pattern 'tile_row_col'.
    - Movement cost between adjacent tiles is 1.
    - Color change cost is 1.
    - Painting a tile costs 1.
    - The 'clear' precondition for movement and painting adjacent tiles is ignored
      when calculating movement distance (Manhattan distance assumes free movement).
    - The 'clear' precondition for the target tile itself is considered: if a robot
      is currently on the tile that needs painting, an extra move action is assumed
      to clear it before it can be painted.
    - The problem is solvable, meaning any tile needing painting is either clear
      or occupied by a robot, and not painted the wrong color.

    # Heuristic Initialization
    - Extracts all goal predicates of the form `(painted tile_name color)`.
    - Identifies all tile objects mentioned in the problem and maps their names
      to (row, col) coordinates based on the 'tile_r_c' naming convention.
    - Stores the set of all valid tile coordinates.
    - Identifies all robot names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal predicates of the form `(painted T C)`.
    2. From the current state, identify which of these goal predicates are not satisfied.
       A goal `(painted T C)` is unsatisfied if the state does NOT contain `(painted T C)`.
       Let this set of unsatisfied goals be `UnsatisfiedGoals = {(T, C) | (painted T C) is goal AND (painted T C) NOT in state}`.
    3. If `UnsatisfiedGoals` is empty, the heuristic value is 0.
    4. Initialize the total heuristic value `h = 0`.
    5. For each unsatisfied goal `(T, C)` where tile `T` needs color `C`:
        a. Calculate the minimum estimated cost to paint tile `T` with color `C`.
        b. This minimum cost includes:
           - `1` for the paint action itself.
           - `(1 if any robot is currently at tile T else 0)`: This accounts for the cost to move a robot off the tile so it becomes clear and can be painted.
           - The minimum cost over all robots `R` to get `R` into a state where it can paint `T`. This involves getting the correct color and moving adjacent to `T`.
        c. The cost for a robot `R` to get color `C` and move adjacent to `T` is:
           - `cost_color = 1` if robot `R` does not have color `C`, `0` otherwise.
           - `cost_move = min_distance(R.location, adjacent_to(T))` using Manhattan distance.
           - `min_distance(R.location, adjacent_to(T))` is the minimum Manhattan distance from `R`'s current location to any valid tile coordinate adjacent to `T`.
        d. The minimum estimated cost for tile `(T, C)` is `1 (paint) + (1 if robot on T else 0) + min_{R} (cost_color + cost_move)`.
        e. Add this minimum estimated cost for tile `(T, C)` to the total heuristic value `h`.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and tile information.
        """
        self.goal_painted_tiles = {}
        # Extract goal painted tiles
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_painted_tiles[tile] = color

        # Collect all tile names and map to coordinates
        self.tile_coords = {}
        self.valid_coords = set()

        # Helper to find all object names of a specific type by checking fact arguments
        def find_objects_by_prefix(fact_list, prefix):
            objects = set()
            for fact in fact_list:
                 parts = get_parts(fact)
                 for part in parts:
                     if part.startswith(prefix):
                         objects.add(part)
            return objects

        # Collect tile names from initial state, goals, and static facts
        all_tile_names = find_objects_by_prefix(task.initial_state, 'tile_') | \
                         find_objects_by_prefix(task.goals, 'tile_') | \
                         find_objects_by_prefix(task.static, 'tile_')

        for tile_name in all_tile_names:
            coords = get_tile_coords(tile_name)
            if coords:
                self.tile_coords[tile_name] = coords
                self.valid_coords.add(coords)

        # Collect robot names from initial state
        self.robot_names = find_objects_by_prefix(task.initial_state, 'robot')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Identify unsatisfied painted goals
        unsatisfied_goals = {}
        for tile, goal_color in self.goal_painted_tiles.items():
            # Check if the tile is painted with the correct color in the current state
            is_painted_correctly = False
            for fact in state:
                if match(fact, "painted", tile, goal_color):
                    is_painted_correctly = True
                    break

            if not is_painted_correctly:
                 # If not painted correctly, it needs attention.
                 # Assuming solvable problems, it's either clear or has a robot on it.
                 unsatisfied_goals[tile] = goal_color


        if not unsatisfied_goals:
            return 0  # Goal reached

        # Get current robot states (location and color)
        robot_state = {}
        for robot_name in self.robot_names:
            location = None
            color = None
            # Find robot's location and color in the current state
            for fact in state:
                if match(fact, "robot-at", robot_name, "*"):
                    location = get_parts(fact)[2]
                if match(fact, "robot-has", robot_name, "*"):
                    color = get_parts(fact)[2]
            # A robot must have a location and color to be useful
            if location and color:
                 robot_state[robot_name] = {'location': location, 'color': color}
            # else: This robot is not currently in a state we can use (e.g., not at a location?)
            # For this heuristic, we only consider robots with known location and color.


        total_h = 0

        # Calculate cost for each unsatisfied tile independently
        for tile_to_paint, needed_color in unsatisfied_goals.items():
            min_cost_for_tile_by_any_robot = float('inf')

            # Cost to clear the tile if a robot is on it
            # Check if any robot is currently at tile_to_paint
            is_robot_on_tile = False
            for robot_name in self.robot_names:
                 if f"(robot-at {robot_name} {tile_to_paint})" in state:
                      is_robot_on_tile = True
                      break
            cost_to_clear_tile = 1 if is_robot_on_tile else 0 # Assumes 1 move action clears it

            # Get coordinates of the tile to paint
            tile_coords = self.tile_coords.get(tile_to_paint)
            if tile_coords is None:
                 # Should not happen in valid problems, but handle defensively
                 continue

            (tr, tc) = tile_coords

            # Get coordinates of potential adjacent tiles
            potential_adj_coords = [(tr-1, tc), (tr+1, tc), (tr, tc-1), (tr, tc+1)]
            # Filter to include only valid tile coordinates that exist in the problem
            valid_adj_coords = [coord for coord in potential_adj_coords if coord in self.valid_coords]

            if not valid_adj_coords:
                 # Tile has no valid adjacent tiles? Should not happen in valid grid problems.
                 # This tile is unreachable for painting. Problem likely unsolvable.
                 # Assign a very high cost? Or skip? Skipping might underestimate if it's the only goal.
                 # Let's assume valid problems have reachable goal tiles.
                 continue

            # Find the minimum cost among all available robots to get ready and adjacent
            for robot_name, r_info in robot_state.items():
                r_location = r_info['location']
                r_color = r_info['color']

                # Cost to change color if needed
                cost_color_change = 1 if r_color != needed_color else 0

                # Get robot's current coordinates
                r_coords = self.tile_coords.get(r_location)
                if r_coords is None:
                     # Robot location not found in tile_coords? Problematic state.
                     continue # Skip this robot

                # Minimum movement cost to reach any valid adjacent tile
                min_move_cost = float('inf')
                for (ar, ac) in valid_adj_coords:
                    move_cost = manhattan_distance(r_coords, (ar, ac))
                    min_move_cost = min(min_move_cost, move_cost)

                # Cost for this robot to get ready and adjacent to paint this tile
                cost_for_robot = cost_color_change + min_move_cost

                min_cost_for_tile_by_any_robot = min(min_cost_for_tile_by_any_robot, cost_for_robot)

            # Add the minimum cost found for this tile (including paint and clearing)
            if min_cost_for_tile_by_any_robot != float('inf'):
                 total_h += cost_to_clear_tile + min_cost_for_tile_by_any_robot + 1 # +1 for the paint action
            # else: Tile is unreachable by any available robot? Should not happen in valid problems.


        return total_h
