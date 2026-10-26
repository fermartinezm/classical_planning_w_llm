from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Use math.inf for infinity

def get_parts(fact):
    """
    Helper function to parse a PDDL fact string into its components.
    E.g., '(predicate arg1 arg2)' -> ['predicate', 'arg1', 'arg2']
    """
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    the estimated minimum cost for each unpainted goal tile. For each
    unpainted goal tile, it calculates the minimum cost for any robot
    to paint that tile, considering the robot's current color, the required
    color, the robot's location, and the distance to a clear adjacent tile
    from which the painting can occur.

    Assumptions:
    - Tile names follow the format 'tile_row_col', allowing extraction of
      grid coordinates for Manhattan distance calculation.
    - The problem instances are solvable, meaning no goal tile is painted
      with the wrong color in the initial state or any reachable state.
    - Robots can change color if the target color is available.
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates
      corresponds to a regular grid where tile_r_c is adjacent to
      tile_{r+/-1}_c and tile_r_{c+/-1}.

    Heuristic Initialization:
    During initialization, the heuristic processes the static information
    from the task definition:
    - It identifies all goal facts of the form (painted tile color) and stores
      them in a dictionary mapping goal tiles to their required colors.
    - It parses the 'up', 'down', 'left', 'right' static facts to build a map
      from each tile that needs to be painted to the set of tiles from which
      a robot can paint it (i.e., its adjacent tiles in the grid).
    - It extracts the row and column coordinates for each tile name based on
      the 'tile_row_col' format, storing them in a dictionary. This is used
      later for Manhattan distance calculations.
    - It identifies the set of available colors.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify all goal tiles that are not currently painted with the correct
       color according to the goal state. These are the 'unpainted goal tiles'.
    2. If there are no unpainted goal tiles, the state is a goal state, and the
       heuristic value is 0.
    3. Parse the current state to determine:
       - The current location of each robot.
       - The current color held by each robot.
       - The set of tiles that are currently clear.
    4. Initialize the total heuristic value `h` to 0.
    5. For each unpainted goal tile `T` that needs to be painted with color `C`:
       a. Initialize a variable `min_cost_for_tile_T` to infinity. This will store
          the minimum estimated cost for *any* robot to paint tile `T`.
       b. Identify the set of tiles `AdjT` from which tile `T` can be painted
          (using the precomputed adjacency map).
       c. For each robot `R` currently at location `R_loc` with color `R_color`:
          i. Calculate the estimated cost for robot `R` to paint tile `T`.
             Start with `current_robot_cost = 0`.
          ii. If `R_color` is not the required color `C`:
              Add 1 to `current_robot_cost` (representing the `change_color` action).
          iii. Find the minimum movement cost for robot `R` to reach *any* tile
               in `AdjT` that is currently `clear`.
               Initialize `min_move_cost_to_adj = infinity`.
               For each tile `adj_tile` in `AdjT`:
                 If `adj_tile` is in the set of `clear_tiles` in the current state:
                   Calculate the Manhattan distance between `R_loc` and `adj_tile`.
                   Update `min_move_move_cost_to_adj = min(min_move_cost_to_adj, distance)`.
          iv. If `min_move_cost_to_adj` is still infinity (meaning robot `R` cannot
              reach any clear adjacent tile):
              Robot `R` cannot paint tile `T` in this state. Continue to the next robot.
          v. Otherwise (robot `R` can reach a clear adjacent tile):
             Add `min_move_cost_to_adj` to `current_robot_cost`.
             Add 1 to `current_robot_cost` (representing the `paint_...` action).
             Update `min_cost_for_tile_T = min(min_cost_for_tile_T, current_robot_cost)`.
       d. After checking all robots, if `min_cost_for_tile_T` is still infinity,
          it means no robot can paint tile `T` in this state. This suggests the
          state is likely unsolvable. Return infinity as the heuristic value.
       e. Otherwise, add `min_cost_for_tile_T` to the total heuristic value `h`.
    6. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        """
        Initializes the heuristic with static task information.

        Args:
            task: The planning task object containing goals and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Preprocess goal facts
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                self.goal_painted_tiles[parts[1]] = parts[2]

        # Preprocess static facts
        self.tile_coords = {}
        self.adjacency_map_for_painting = {} # Maps tile_to_be_painted -> list of tiles_robot_can_stand_on_to_paint_it
        self.all_tiles = set()
        self.available_colors = set()

        for s in static_facts:
            parts = get_parts(s)
            if parts[0] in ['up', 'down', 'left', 'right']:
                # (direction tile_y tile_x) means tile_y is in that direction from tile_x
                # A robot at tile_x can paint tile_y
                tile_y = parts[1] # The tile being painted
                tile_x = parts[2] # The tile the robot stands on

                self.all_tiles.add(tile_y)
                self.all_tiles.add(tile_x)

                self.adjacency_map_for_painting.setdefault(tile_y, []).append(tile_x)
            elif parts[0] == 'available-color':
                self.available_colors.add(parts[1])

        # Infer tile coordinates from names assuming tile_r_c format
        for tile_name in self.all_tiles:
            try:
                # Example: 'tile_0_1' -> ['tile', '0', '1']
                parts = tile_name.split('_')
                if len(parts) == 3 and parts[0] == 'tile':
                    row = int(parts[1])
                    col = int(parts[2])
                    self.tile_coords[tile_name] = (row, col)
                # Add other potential tile name formats if necessary, or handle errors
            except (ValueError, IndexError):
                # Handle tiles that don't match the expected format if any exist
                # For this problem, assuming tile_r_c is sufficient based on examples
                print(f"Warning: Could not parse coordinates for tile '{tile_name}'. Manhattan distance might be inaccurate.")
                pass # Tile won't have coordinates, distance calculation will return inf

    def manhattan_distance(self, tile1_name, tile2_name):
        """
        Calculates the Manhattan distance between two tiles based on their names.
        Assumes tile names are in 'tile_row_col' format.
        """
        coords1 = self.tile_coords.get(tile1_name)
        coords2 = self.tile_coords.get(tile2_name)

        if coords1 is None or coords2 is None:
            # Cannot calculate distance if coordinates are missing
            return float('inf')

        r1, c1 = coords1
        r2, c2 = coords2
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated cost to reach the goal state from the current state.
            Returns math.inf if the state is estimated to be unsolvable.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Parse dynamic state facts
        robot_locations = {}
        robot_colors = {}
        clear_tiles = set()
        painted_tiles_state = {} # Store painted tiles and their colors in the state

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == 'robot-has':
                robot_colors[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                clear_tiles.add(parts[1])
            elif parts[0] == 'painted':
                 painted_tiles_state[parts[1]] = parts[2]

        # Identify unpainted goal tiles
        unpainted_goal_tiles = {}
        for tile, required_color in self.goal_painted_tiles.items():
            state_fact = f'(painted {tile} {required_color})'
            if state_fact not in state:
                 # Check if it's painted with the wrong color (unsolvable assumption)
                 if tile in painted_tiles_state and painted_tiles_state[tile] != required_color:
                     # Goal tile painted wrong color -> unsolvable state
                     return math.inf
                 # It's not painted correctly, and not painted wrong -> it's clear or painted wrong (handled above)
                 # Assuming solvable problems, if not painted correctly, it must be clear.
                 # Let's explicitly check if it's clear. If not clear and not painted correctly, something is wrong.
                 if f'(clear {tile})' not in state:
                      # This case should ideally not happen in solvable problems
                      # A tile is either clear or painted. If not painted correctly, and not clear,
                      # it must be painted with the wrong color (handled) or some other unexpected state.
                      # Return inf for safety.
                      return math.inf

                 unpainted_goal_tiles[tile] = required_color


        # If no unpainted goal tiles, but goal not reached, something is wrong (e.g., other goal conditions)
        # But floortile goal is only painted facts. So this case implies goal reached.
        if not unpainted_goal_tiles:
             # This should only happen if self.goals <= state was false incorrectly, or if goals
             # include more than just painted facts (which they don't in this domain).
             # Given the domain, if unpainted_goal_tiles is empty, the goal *is* reached.
             # The initial check `if self.goals <= state:` handles this.
             # This part of the code should ideally not be reached if the initial check is correct.
             # However, as a fallback, returning 0 here is consistent with no unpainted tiles.
             return 0


        total_heuristic = 0

        # Calculate cost for each unpainted goal tile
        for tile, required_color in unpainted_goal_tiles.items():
            min_cost_for_tile = float('inf')
            adjacent_tiles_T = self.adjacency_map_for_painting.get(tile, [])

            # If a tile has no adjacent tiles from which it can be painted, it's unsolvable
            if not adjacent_tiles_T:
                 return math.inf

            for robot, R_loc in robot_locations.items():
                R_color = robot_colors.get(robot) # Get robot's current color

                current_robot_cost = 0
                # Cost to change color if needed
                if R_color != required_color:
                    # Assuming robot can change color if the required color is available
                    # The domain requires (available-color c2) for change_color
                    # We assume all required colors are available (from problem definition)
                    current_robot_cost += 1 # Cost of change_color action

                # Find minimum movement cost to a clear adjacent tile
                min_move_cost_to_adj = float('inf')
                for adj_tile in adjacent_tiles_T:
                    if f'(clear {adj_tile})' in state:
                        dist = self.manhattan_distance(R_loc, adj_tile)
                        min_move_cost_to_adj = min(min_move_cost_to_adj, dist)

                # If robot can reach a clear adjacent tile
                if min_move_cost_to_adj is not float('inf'):
                    current_robot_cost += min_move_cost_to_adj
                    current_robot_cost += 1 # Cost of paint action
                    min_cost_for_tile = min(min_cost_for_tile, current_robot_cost)

            # If no robot can paint this tile (e.g., no clear adjacent tile reachable by any robot)
            if min_cost_for_tile is float('inf'):
                return math.inf # State is unsolvable

            total_heuristic += min_cost_for_tile

        return total_heuristic

