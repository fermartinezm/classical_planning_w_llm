from heuristics.heuristic_base import Heuristic
# No other external libraries like fnmatch are used in this heuristic.
import math # Used for float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe return empty list or raise error
        # For PDDL facts from the planner, this format is expected.
        return []
    return fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all
    goal tiles with their target colors. It sums up the estimated minimum cost
    for each individual goal tile that is not yet correctly painted. The cost
    for a single tile includes moving a robot to an adjacent position, changing
    the robot's color if necessary, and performing the paint action.

    # Assumptions
    - Tile names follow the format 'tile_R_C' where R and C are integers representing
      row and column, allowing Manhattan distance calculation.
    - Manhattan distance is a reasonable approximation for movement cost on the grid.
    - The 'change_color' action is always possible if the robot has any color
      and the target color is available (heuristic ignores the 'available-color'
      precondition for simplicity).
    - If a goal tile is already painted with a color different from the goal color,
      the problem is considered highly costly (potentially unsolvable within the
      standard actions), and a large penalty is added.
    - The heuristic ignores the 'clear' predicate for intermediate tiles along
      a robot's path and for the adjacent tile needed for painting, assuming
      these can be managed. The goal tile itself must be clear to be painted
      (this is implicitly handled by checking if it's already painted wrongly).

    # Heuristic Initialization
    - Parses the goal conditions to identify which tiles need to be painted
      and with which colors. Stores this in `self.goal_tiles`.
    - Parses the initial state and static facts to identify all tile objects
      and extracts their row and column coordinates based on the 'tile_R_C'
      naming convention. Stores this in `self.tile_coords`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic value `h` to 0.
    2. Iterate through each goal requirement `(painted T C)` stored in `self.goal_tiles`.
    3. For the current goal tile `T` and target color `C`:
       a. Check if the fact `(painted T C)` is already present in the current state. If yes, this goal is satisfied for this tile; continue to the next goal tile.
       b. Check if the tile `T` is currently painted with a *different* color `C'`. Iterate through the state facts looking for `(painted T C')` where `C' != C`.
       c. If `T` is painted with a wrong color, add a large penalty (e.g., 1000) to `h`. This signifies a difficult or impossible situation for this tile with standard actions. Continue to the next goal tile.
       d. If `T` is not correctly painted and not wrongly painted (implying it is clear or has no paint), it needs to be painted `C`. Estimate the minimum cost for this tile:
          i. Find the current location and color of every robot in the state.
          ii. For each robot `R` at `R_loc` with color `R_color`:
             - Get the coordinates of `R_loc` and `T` using `self.tile_coords`.
             - Calculate the Manhattan distance `d` between `R_loc` and `T`.
             - Calculate the estimated move cost for `R` to reach a tile adjacent to `T` and perform the paint action. If the Manhattan distance `d` between `R_loc` and `T` is 0 (robot is at T), it needs 1 move away and 1 paint action (total 2 actions). If `d >= 1`, it needs `d-1` moves to get adjacent and 1 paint action (total `d` actions). The combined move+paint cost is `d` if `d >= 1 else 2`.
             - Calculate the color change cost: 1 if `R_color` is not `C`, otherwise 0.
             - The total cost for robot `R` to paint tile `T` is (move+paint cost) + color change cost.
          iii. Find the minimum total cost among all robots to paint tile `T`.
          iv. Add this minimum cost to `h`.
    4. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and tile coordinates.
        """
        self.goals = task.goals

        # 1. Parse goal facts to identify target tiles and colors.
        self.goal_tiles = {} # {tile_name: target_color}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        # 2. Parse tile names from facts to get coordinates.
        self.tile_coords = {} # {tile_name: (row, col)}
        all_tiles = set()

        # Helper to add tile names from fact parts based on domain predicate structure
        def add_tiles_from_fact(fact_parts):
             if not fact_parts: return
             predicate = fact_parts[0]
             # Check predicates that involve tile objects based on domain definition
             if predicate == "robot-at" and len(fact_parts) == 3:
                  # (robot-at ?r - robot ?x - tile)
                  all_tiles.add(fact_parts[2])
             elif predicate in ["up", "down", "left", "right"] and len(fact_parts) == 3:
                  # (up ?x - tile ?y - tile), etc.
                  all_tiles.add(fact_parts[1])
                  all_tiles.add(fact_parts[2])
             elif predicate == "clear" and len(fact_parts) == 2:
                  # (clear ?x - tile)
                  all_tiles.add(fact_parts[1])
             elif predicate == "painted" and len(fact_parts) == 3:
                  # (painted ?x - tile ?c - color)
                  all_tiles.add(fact_parts[1])
             # Predicates like robot-has, available-color, free-color do not involve tiles

        # Collect all tile names from initial state and static facts
        for fact in task.initial_state | task.static:
             add_tiles_from_fact(get_parts(fact))

        # Now parse coordinates from tile names
        for tile_name in all_tiles:
            try:
                # Assuming tile names are like 'tile_R_C'
                _, r_str, c_str = tile_name.split('_')
                row = int(r_str)
                col = int(c_str)
                self.tile_coords[tile_name] = (row, col)
            except ValueError:
                # Handle unexpected tile name format if necessary
                # print(f"Warning: Could not parse tile name {tile_name}")
                pass # Ignore tiles with unexpected names

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        h = 0  # Initialize heuristic value

        # Find robot locations and colors in the current state
        robot_info = {} # {robot_name: [location, color]}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                robot, location = parts[1], parts[2]
                if robot not in robot_info: robot_info[robot] = [None, None]
                robot_info[robot][0] = location
            elif predicate == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                if robot not in robot_info: robot_info[robot] = [None, None]
                robot_info[robot][1] = color

        # Iterate through each goal tile requirement
        for target_tile, target_color in self.goal_tiles.items():
            # Check if the goal is already satisfied for this tile
            if f"(painted {target_tile} {target_color})" in state:
                continue # This tile is done

            # Check if the tile is painted with the wrong color
            is_wrongly_painted = False
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "painted" and len(parts) == 3 and parts[1] == target_tile:
                    # It's painted, but not with the target color (checked above)
                    is_wrongly_painted = True
                    break # Found wrong paint, no need to check other painted facts for this tile

            if is_wrongly_painted:
                 # Problem likely unsolvable for this tile with standard actions
                 h += 1000 # Large penalty
                 continue # Move to the next goal tile

            # Tile is not correctly painted and not wrongly painted (assumed clear/unpainted)
            # Estimate cost to paint this tile
            min_robot_cost_for_tile = math.inf

            target_coords = self.tile_coords.get(target_tile)
            if target_coords is None:
                 # Should not happen for a valid goal tile name extracted from task
                 # print(f"Warning: Coordinates not found for goal tile {target_tile}")
                 h += 1000 # Add penalty if goal tile coords are unknown
                 continue

            # Calculate cost for each robot to paint this tile
            for robot, (robot_loc, robot_color) in robot_info.items():
                if robot_loc is None or robot_color is None:
                     # Robot info incomplete, skip this robot for this tile calculation
                     continue

                robot_coords = self.tile_coords.get(robot_loc)
                if robot_coords is None:
                     # Robot is at an unknown location? Should not happen based on domain.
                     # print(f"Warning: Coordinates not found for robot location {robot_loc}")
                     continue

                # Calculate Manhattan distance between robot and target tile
                dist = abs(robot_coords[0] - target_coords[0]) + abs(robot_coords[1] - target_coords[1])

                # Estimated actions for move + paint:
                # If dist == 0 (robot at target tile): 1 move away + 1 paint = 2 actions
                # If dist >= 1 (robot >= 1 step away): (dist - 1) moves to adjacent + 1 paint = dist actions
                move_paint_cost = dist if dist >= 1 else 2

                # Estimated actions for color change:
                color_change_cost = 0
                if robot_color != target_color:
                    color_change_cost = 1 # Assume target color is available

                # Total cost for this robot to paint this tile
                robot_total_cost = move_paint_cost + color_change_cost

                min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_total_cost)

            # Add the minimum cost for this tile to the total heuristic
            if min_robot_cost_for_tile != math.inf:
                 h += min_robot_cost_for_tile
            else:
                 # No robots found or issue with robot info for this tile
                 h += 1000 # Add penalty

        return h
