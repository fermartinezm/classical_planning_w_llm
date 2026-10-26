import re
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure fact has at least as many parts as args for zip to work correctly
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the minimum estimated cost for each unpainted
    goal tile, considering the closest robot with the correct color. The movement
    cost is estimated using Manhattan distance on the grid.

    # Assumptions
    - Tiles are arranged in a grid, and tile names follow the format 'tile_row_col'.
    - The first number in 'tile_row_col' is the row index, and the second is the column index.
    - 'up' means decreasing row index, 'down' means increasing row index,
      'left' means decreasing column index, 'right' means increasing column index.
    - If a goal tile is painted with the wrong color, the state is considered
      unsolvable or very far from the goal, resulting in a high heuristic value.
    - The cost of moving between adjacent tiles is 1. Manhattan distance is used
      as an estimate for movement cost, ignoring obstacles ('clear' status).
    - Changing color costs 1 action if the robot doesn't have the required color.
    - Painting costs 1 action.
    - The heuristic ignores potential conflicts between robots (e.g., path blocking,
      contention for colors or tiles) and the 'clear' precondition for moves and paints,
      except for detecting wrongly painted goal tiles.

    # Heuristic Initialization
    - Parses all tile names from initial state and static facts to create a mapping
      between tile names (strings) and their grid coordinates (tuples `(row, col)`).
    - Identifies all robots.
    - Stores the target color for each goal tile by parsing the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify the current location and held color for each robot in the state.
    3. Identify which tiles are currently painted and their colors.
    4. Iterate through each tile specified in the goal conditions:
        a. If the tile is already painted with the correct goal color, add 0 to the total cost.
        b. If the tile is painted with a different color, return a very large number (indicating a likely unsolvable state).
        c. If the tile is not painted with the correct color:
            i. Determine the required painting positions (adjacent tiles where a robot must be to paint the goal tile). Based on the domain actions, to paint tile (r, c), the robot must be at (r+1, c) for paint_up, (r-1, c) for paint_down, (r, c+1) for paint_left, or (r, c-1) for paint_right.
            ii. Filter these positions to include only valid tiles present in the problem instance grid.
            iii. Initialize the minimum cost for any robot to paint this tile to infinity.
            iv. For each robot:
                - Calculate the Manhattan distance from the robot's current location to the closest required painting position for the goal tile.
                - Add 1 to the cost if the robot does not currently hold the required color.
                - Add 1 for the paint action itself.
                - The total cost for this robot is the sum of movement cost, color change cost, and paint cost.
                - Update the minimum cost for this goal tile with the minimum cost found among all robots.
            v. Add the minimum cost for this goal tile to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and parsing the grid structure from tile names.
        """
        self.goals = task.goals
        self.static = task.static

        self.tile_name_to_coords = {}
        self.coords_to_tile_name = {}
        self.robots = set()

        # Parse tile names and build coordinate maps
        # Assuming tile names are consistently tile_row_col
        tile_pattern = re.compile(r"tile_(\d+)_(\d+)")
        all_objects = set()
        # Collect all objects mentioned in initial state and static facts
        for fact in task.initial_state | task.static:
             parts = get_parts(fact)
             for part in parts:
                 all_objects.add(part)

        # Filter for tile objects and parse coordinates
        for obj in all_objects:
            match = tile_pattern.match(obj)
            if match:
                r, c = int(match.group(1)), int(match.group(2))
                self.tile_name_to_coords[obj] = (r, c)
                self.coords_to_tile_name[(r, c)] = obj

        # Identify robots
        for fact in task.initial_state | task.static:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                self.robots.add(parts[1])
            elif parts[0] == 'robot-has': # Also can find robots here
                 self.robots.add(parts[1])


        # Store goal locations and colors for tiles
        self.goal_painted_color = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_painted_color[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract current state information ---
        robot_locations = {}
        robot_colors = {}
        current_painted_color = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                current_painted_color[tile] = color

        # --- Compute heuristic cost ---
        total_cost = 0
        large_penalty = 10000 # Penalty for likely unsolvable states

        for goal_tile, goal_color in self.goal_painted_color.items():
            # Check if goal for this tile is already satisfied
            if goal_tile in current_painted_color and current_painted_color[goal_tile] == goal_color:
                continue # Goal satisfied for this tile

            # Check if tile is painted with the wrong color (likely unsolvable)
            if goal_tile in current_painted_color and current_painted_color[goal_tile] != goal_color:
                 # If a goal tile is painted with the wrong color, it cannot be repainted
                 # as it's not clear. This state is likely unsolvable.
                 return large_penalty # Return a high cost

            # Tile needs to be painted with goal_color

            # Determine required painting positions (adjacent tiles)
            # A robot at (r+1, c) can paint (r, c) using paint_up
            # A robot at (r-1, c) can paint (r, c) using paint_down
            # A robot at (r, c+1) can paint (r, c) using paint_left
            # A robot at (r, c-1) can paint (r, c) using paint_right
            goal_r, goal_c = self.tile_name_to_coords[goal_tile]
            required_adj_coords = [
                (goal_r + 1, goal_c), # Robot pos for paint_up on (r,c)
                (goal_r - 1, goal_c), # Robot pos for paint_down on (r,c)
                (goal_r, goal_c + 1), # Robot pos for paint_left on (r,c)
                (goal_r, goal_c - 1), # Robot pos for paint_right on (r,c)
            ]

            # Filter for valid adjacent tiles that exist in the problem grid
            required_painting_positions = [
                self.coords_to_tile_name[coords]
                for coords in required_adj_coords
                if coords in self.coords_to_tile_name
            ]

            if not required_painting_positions:
                 # This tile cannot be painted if it has no valid adjacent tiles
                 # (e.g., a 1x1 grid, or edge tiles without corresponding neighbors defined)
                 # This indicates a potentially malformed problem or an unsolvable state.
                 # For robustness, return a large penalty.
                 return large_penalty


            # Calculate minimum cost for any robot to paint this tile
            min_robot_cost = float('inf')

            for robot_name in self.robots:
                # Get robot's current location and coordinates
                robot_loc = robot_locations[robot_name]
                robot_coords = self.tile_name_to_coords[robot_loc]

                # Calculate minimum Manhattan distance from robot's current location
                # to any of the required painting positions for this goal tile.
                min_dist_to_paint_pos = float('inf')
                for paint_pos_tile in required_painting_positions:
                    paint_pos_coords = self.tile_name_to_coords[paint_pos_tile]
                    dist = abs(robot_coords[0] - paint_pos_coords[0]) + abs(robot_coords[1] - paint_pos_coords[1])
                    min_dist_to_paint_pos = min(min_dist_to_paint_pos, dist)

                # Cost to move to the painting position (estimated by Manhattan distance)
                move_cost = min_dist_to_paint_pos

                # Cost to get the correct color (1 if robot doesn't have it, 0 otherwise)
                cost_to_get_color = 1 if robot_colors.get(robot_name) != goal_color else 0 # Use .get for safety

                # Cost to paint the tile
                paint_cost = 1

                # Total estimated cost for this robot to paint this tile
                cost_for_this_robot = move_cost + cost_to_get_color + paint_cost

                # Update the minimum cost for this goal tile
                min_robot_cost = min(min_robot_cost, cost_for_this_robot)

            # Add the minimum cost for this tile (over all robots) to the total cost
            total_cost += min_robot_cost

        # The heuristic is 0 only if all goal tiles are painted correctly.
        # If total_cost is 0, it means all goal tiles were already painted correctly.
        # If total_cost is > 0, it means there are unpainted goal tiles.
        # The large_penalty case handles unsolvable states.
        return total_cost
