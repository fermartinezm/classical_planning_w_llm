# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# In a real environment, this would be provided by the planner framework.
class Heuristic:
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input format gracefully, though planner states should be consistent
        # print(f"Warning: Unexpected fact format: {fact}") # Too noisy
        return []
    return fact[1:-1].split()

# Helper function to parse tile coordinates
def parse_tile_coords(tile_name):
    """Parses 'tile_r_c' string into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle unexpected tile name format (e.g., non-integer row/col)
            # print(f"Warning: Could not parse integer coordinates from tile name: {tile_name}") # Too noisy
            return None
    # Handle unexpected tile name format (e.g., not starting with 'tile_', wrong number of parts)
    # print(f"Warning: Unexpected tile name format: {tile_name}") # Too noisy
    return None


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the estimated cost for each unsatisfied
    goal tile, considering the painting action, the movement cost for a robot
    to reach an adjacent tile, and the cost of changing the robot's color.

    # Assumptions
    - Tiles are arranged in a grid structure, and tile names like 'tile_r_c'
      can be parsed to extract row and column coordinates (r, c).
    - Movement cost between adjacent tiles is 1. Manhattan distance is used
      as a lower bound for movement cost on the grid.
    - A robot must be at a tile adjacent to the target tile to paint it.
    - The target tile must be clear to be painted.
    - If a goal tile is already painted with the wrong color, the state is
      considered unsolvable from this point (heuristic returns infinity).
    - The cost of changing color is counted once for each unique color needed
      by unsatisfied clear goal tiles, if no robot currently holds that color.
      This assumes one `change_color` action can potentially satisfy the need
      for that color for multiple tiles, or another robot can change color.
      This is a simplification for non-admissibility.
    - Robots start with a color, allowing `change_color` action if `available-color` holds.
      If no robot holds any color in a state where colors are needed, it's unsolvable.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which colors. Stores this in `self.goal_tiles_info`.
    - Extracts available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic value `h` to 0.
    2. Extract relevant information from the current state:
       - Robot locations (`robot_locations`).
       - Colors held by robots (`robot_colors`).
       - Tiles that are clear (`clear_tiles`).
       - Tiles that are painted and their colors (`painted_tiles`).
    3. Initialize a set `needed_colors` to track colors required by unsatisfied
       clear goal tiles.
    4. Iterate through each goal tile and its target color from `self.goal_tiles_info`:
       a. Check if the goal `(painted tile_X color_Y)` is already true in the current state.
          If yes, this goal is satisfied; continue to the next goal.
       b. If the goal is not satisfied, check the state of `tile_X`:
          - If `tile_X` is in `painted_tiles` (meaning it's painted with *some* color),
            and it's not painted with the goal color `color_Y` (checked in step 4a),
            then it must be painted with the wrong color. This state is likely
            unsolvable. Return a very large number (e.g., 1000000).
          - If `tile_X` is in `clear_tiles` (it's clear and needs painting):
            i. Add `color_Y` to the `needed_colors` set.
            ii. Add 1 to `h` for the painting action itself.
            iii. Calculate the minimum movement cost for any robot to reach a tile
                 adjacent to `tile_X`.
                 - Parse coordinates for `tile_X` (`rx`, `cx`). Handle parsing errors.
                 - For each robot at `tile_R` (`rr`, `cr`), parse coordinates. Handle parsing errors.
                 - Calculate Manhattan distance `D = abs(rx - rr) + abs(cx - cr)`.
                 - The number of moves required for the robot to reach *any* tile adjacent to `tile_X` from `tile_R` is estimated: If `D=0`, moves = 1 (move away). If `D>0`, moves = `D-1`.
                 - Find the minimum such `moves` value over all robots. If no robots or parsing fails, return infinity.
                 - Add this minimum movement cost to `h`.
    5. After iterating through all goal tiles, calculate the color change cost:
       - Determine the set of colors currently held by robots (`robot_colors`).
       - Check if any robot exists (`can_change_color`). If not, but colors are needed, return infinity.
       - For each unique color in `needed_colors`: if that color is not in `robot_colors`, add 1 to `h` for the color change action.
    6. Return the total calculated value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        super().__init__(task)

        # Store goal locations and colors for each tile
        self.goal_tiles_info = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                if len(parts) == 3:
                    tile_name = parts[1]
                    color_name = parts[2]
                    self.goal_tiles_info[tile_name] = color_name

        # Extract available colors from static facts (not strictly needed for this heuristic logic,
        # but good practice to extract all relevant static info)
        self.available_colors = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "available-color":
                 if len(parts) == 2:
                     self.available_colors.add(parts[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Extract relevant information from the current state
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = set()    # {color_name} - just need the set of colors held
        clear_tiles = set()  # {tile_name}
        painted_tiles = {}   # {tile_name: color_name}

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue

            predicate = parts[0]
            if predicate == "robot-at":
                if len(parts) == 3:
                    robot_name, tile_name = parts[1], parts[2]
                    robot_locations[robot_name] = tile_name
            elif predicate == "robot-has":
                 if len(parts) == 3:
                    robot_name, color_name = parts[1], parts[2]
                    robot_colors.add(color_name) # Add color to set
            elif predicate == "clear":
                 if len(parts) == 2:
                    tile_name = parts[1]
                    clear_tiles.add(tile_name)
            elif predicate == "painted":
                 if len(parts) == 3:
                    tile_name, color_name = parts[1], parts[2]
                    painted_tiles[tile_name] = color_name

        # If there are no robots, it's impossible to paint unless goal is already reached (checked above)
        if not robot_locations:
             return 1000000 # Represents infinity

        total_cost = 0
        needed_colors = set()
        unreachable_goal = False # Flag for unsolvable states

        # Calculate cost for each unsatisfied goal tile
        for goal_tile, goal_color in self.goal_tiles_info.items():
            is_goal_satisfied = (goal_tile in painted_tiles and painted_tiles[goal_tile] == goal_color)

            if not is_goal_satisfied:
                # Goal is not satisfied
                if goal_tile in painted_tiles:
                    # Tile is painted, but not with the goal color (must be wrong color)
                    # This state is likely unsolvable
                    unreachable_goal = True
                    break # No need to check other goals

                elif goal_tile in clear_tiles:
                    # Tile is clear and needs painting
                    needed_colors.add(goal_color)
                    total_cost += 1 # Cost for the paint action

                    # Calculate minimum movement cost for a robot to get adjacent
                    tile_rx_cx = parse_tile_coords(goal_tile)
                    if tile_rx_cx is None:
                         unreachable_goal = True # Cannot parse tile, cannot solve
                         break

                    min_moves_to_adj = float('inf')
                    found_parseable_robot_location = False

                    for robot_name, robot_tile in robot_locations.items():
                        robot_rr_cr = parse_tile_coords(robot_tile)
                        if robot_rr_cr is None:
                             continue # Skip this robot if location unparseable

                        found_parseable_robot_location = True

                        # Manhattan distance between robot and target tile
                        D = abs(tile_rx_cx[0] - robot_rr_cr[0]) + abs(tile_rx_cx[1] - robot_rr_cr[1])

                        # Moves needed to reach an adjacent tile from robot's current location
                        # If D=0 (robot at target tile), needs 1 move away.
                        # If D=1 (robot adjacent), needs 0 moves.
                        # If D>1, needs D-1 moves.
                        if D == 0:
                            moves = 1 # Must move away first
                        else:
                            moves = D - 1 # Distance to reach an adjacent cell

                        min_moves_to_adj = min(min_moves_to_adj, moves)

                    if not found_parseable_robot_location:
                         # No robots with parseable locations found
                         unreachable_goal = True
                         break

                    total_cost += min_moves_to_adj

                # Else: tile is not clear and not painted (should not happen for goal tiles in valid states)
                pass # No cost added if not clear and not painted

        # If any goal was unreachable (wrongly painted or parsing failed), return infinity
        if unreachable_goal:
            return 1000000

        # Calculate color change cost
        can_change_color = len(robot_locations) > 0 # Can only change color if you have a robot

        color_change_cost = 0
        if needed_colors: # If there are tiles needing painting
            if not can_change_color:
                # Need colors, but no robot exists to paint or change color
                return 1000000 # Impossible
            else:
                for color in needed_colors:
                    if color not in robot_colors:
                        # Need this color, but no robot currently has it.
                        # Assume one change_color action is needed for this color globally.
                        color_change_cost += 1

        total_cost += color_change_cost

        return total_cost
