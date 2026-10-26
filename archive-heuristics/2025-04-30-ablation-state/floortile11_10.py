from fnmatch import fnmatch
# Assuming Heuristic base class is available in the environment at this path
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with fixed arity.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed). The number of args must match the number of parts in the fact.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing, for each
    unsatisfied goal tile, the minimum cost required for any single robot to paint
    that tile. The cost for a single tile by a single robot includes the paint action,
    the cost to change color if necessary, and the estimated movement cost to get
    adjacent to the tile.

    # Assumptions
    - Tiles are arranged in a grid and named using the format 'tile_row_col',
      allowing Manhattan distance calculation based on row and column indices.
    - The grid is connected (implicitly handled by Manhattan distance as a lower bound).
    - All colors required by the goal are available in the domain.
    - The cost of any action (move, paint, change_color) is 1.

    # Heuristic Initialization
    - Extracts all tile names from the task facts and parses their row/column
      coordinates based on the 'tile_row_col' naming convention.
    - Identifies the set of goal conditions specifying which tiles need to be
      painted with which colors.
    - Identifies the set of available colors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal conditions of the form `(painted tile color)`.
    2. Determine which of these goal conditions are *not* satisfied in the current state.
    3. For each unsatisfied goal `(tile_T, color_C)`:
        a. Initialize the minimum cost for any robot to paint this tile to infinity.
        b. Iterate through each robot `R` in the state:
            i. Find the robot's current location `tile_X` and current color `color_R`.
            ii. Calculate the Manhattan distance between `tile_X` and `tile_T` using their pre-calculated coordinates.
            iii. Estimate the number of moves required for the robot to reach a tile *adjacent* to `tile_T`. If the robot is at `tile_T` (distance 0), it needs 1 move to get adjacent. Otherwise (distance > 0), it needs `distance - 1` moves.
            iv. Determine the cost to change the robot's color: 1 if `color_R` is not `color_C`, otherwise 0.
            v. The estimated cost for robot `R` to get ready to paint `tile_T` is the movement cost plus the color change cost.
            vi. Update the minimum cost for painting `tile_T` with the minimum found so far across all robots.
        c. Add 1 (for the paint action itself) plus the minimum robot readiness cost found in step 3b to the total heuristic value.
    4. The total heuristic value is the sum of costs calculated for each unsatisfied goal tile.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, available colors,
        and tile coordinates.
        """
        self.goals = task.goals
        static_facts = task.static
        all_facts = task.facts # Contains all possible ground facts, including type facts

        # 1. Extract goal painted tiles and colors
        self.goal_painted = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted" and len(parts) == 3:
                self.goal_painted.add((parts[1], parts[2]))

        # 2. Extract available colors (not strictly needed for this heuristic logic, but good practice)
        self.available_colors = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "available-color" and len(parts) == 2:
                 self.available_colors.add(parts[1])


        # 3. Build tile coordinates map from all possible tile objects
        self.tile_coords = {}
        tile_names_set = set()
        # Collect all tile names from task.facts (which includes type facts like (tile tile_0_1))
        for fact_str in all_facts:
            parts = get_parts(fact_str)
            # Check for facts like (tile tile_0_1)
            if len(parts) == 2 and parts[0] == 'tile':
                tile_names_set.add(parts[1])
            # Also collect from relational facts just in case type facts are missing
            # (e.g., from up/down/left/right, robot-at, painted, clear)
            elif len(parts) > 1 and parts[0] in ['up', 'down', 'left', 'right', 'robot-at', 'painted', 'clear']:
                 for part in parts[1:]:
                     if part.startswith('tile_'):
                         tile_names_set.add(part)


        for tile_name in tile_names_set:
            try:
                # Assuming tile names are strictly 'tile_row_col'
                _, r_str, c_str = tile_name.split('_')
                self.tile_coords[tile_name] = (int(r_str), int(c_str))
            except ValueError:
                # If a tile name doesn't match the expected format, it won't have coordinates.
                # This might cause issues later if such a tile is in a goal or robot location.
                # For this heuristic, we'll just skip it, assuming valid tiles follow the convention.
                pass


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Get current painted tiles
        current_painted = set()
        # 2. Get current robot locations and colors
        robot_locations = {}
        robot_colors = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted" and len(parts) == 3:
                current_painted.add((parts[1], parts[2]))
            elif parts[0] == "robot-at" and len(parts) == 3:
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]

        # 3. Identify unsatisfied goals
        unsatisfied_goals = self.goal_painted - current_painted

        # If all goals are satisfied, the heuristic is 0.
        if not unsatisfied_goals:
            return 0

        total_cost = 0

        # 4. Calculate cost for each unsatisfied goal tile
        for target_tile, target_color in unsatisfied_goals:
            # Cost for the paint action itself is 1, added at the end of the loop iteration.

            # Get coordinates of the target tile
            target_coords = self.tile_coords.get(target_tile)
            if target_coords is None:
                 # This should not happen if __init__ collected all tile names correctly
                 # and the goal refers to a valid tile. Handle defensively.
                 return float('inf') # Problem likely unsolvable or misparsed


            min_robot_readiness_cost_for_tile = float('inf')

            # Find the minimum cost for any robot to get ready to paint this tile
            for robot_name, robot_location in robot_locations.items():
                robot_color = robot_colors.get(robot_name) # Get color, default None if robot_has fact is missing

                # Get coordinates of the robot's current location
                robot_coords = self.tile_coords.get(robot_location)
                if robot_coords is None:
                    # This should not happen if __init__ collected all tile names correctly
                    # and robot-at refers to a valid tile. Handle defensively.
                    continue # Skip this robot for this tile

                # Calculate Manhattan distance from robot to target tile
                dist_to_target = abs(target_coords[0] - robot_coords[0]) + abs(target_coords[1] - robot_coords[1])

                # Moves needed to get to a tile adjacent to the target tile
                # If robot is at target (dist=0), needs 1 move to get adjacent.
                # If robot is adjacent (dist=1), needs 0 moves.
                # If robot is further (dist>1), needs dist-1 moves.
                moves_to_adjacent = 1 if dist_to_target == 0 else dist_to_target - 1

                # Cost to change color
                color_change_cost = 0
                # Only need to change color if the robot's current color is not the target color
                if robot_color != target_color:
                    color_change_cost = 1 # Assume 1 action to change color

                # Total cost for this robot to get ready to paint this tile
                robot_readiness_cost = moves_to_adjacent + color_change_cost

                # Update minimum robot readiness cost for this tile
                min_robot_readiness_cost_for_tile = min(min_robot_readiness_cost_for_tile, robot_readiness_cost)

            # Add the cost for this tile (paint action + minimum robot readiness cost)
            # If no robots exist or no robot location was found, min_robot_readiness_cost_for_tile remains inf.
            if min_robot_readiness_cost_for_tile == float('inf'):
                 total_cost = float('inf')
                 break # No robot can paint this tile, problem likely unsolvable

            total_cost += 1 + min_robot_readiness_cost_for_tile # 1 for the paint action

        return total_cost
