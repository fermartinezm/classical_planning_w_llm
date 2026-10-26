# Assuming Heuristic base class is available in a module named heuristics
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic base class if running standalone for testing
    class Heuristic:
        def __init__(self, task):
            self._goals = task.goals
            self._static = task.static
        def __call__(self, node):
            raise NotImplementedError
        @property
        def goals(self):
            return self._goals
        @property
        def static(self):
            return self._static


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def parse_tile_name(tile_name):
    """Parses 'tile_r_c' into (r, c) tuple of integers."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
        else:
            # If tile name format is unexpected, this indicates a problem with the instance
            # or domain definition not matching assumptions. Raise an error.
            raise ValueError(f"Unexpected tile name format: {tile_name}")
    except (ValueError, IndexError) as e:
        # Catch potential errors during int conversion or splitting
        raise ValueError(f"Failed to parse tile name {tile_name}: {e}")


def manhattan_distance(pos1, pos2):
    """Calculates Manhattan distance between two (row, col) tuples."""
    r1, c1 = pos1
    r2, c2 = pos2
    return abs(r1 - r2) + abs(c1 - c2)

# The heuristic class itself
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all
    goal tiles that are not yet painted correctly. It calculates the minimum
    cost for each unpainted goal tile independently, considering the closest
    robot and the need to change color, and sums these minimum costs.

    # Assumptions
    - Tiles are arranged in a grid, and movement cost between adjacent tiles is 1.
    - Manhattan distance is used as a proxy for movement cost between tiles,
      ignoring dynamic obstacles (`clear` predicate) for simplicity and efficiency.
    - Goal tiles that are not yet painted with the correct color are assumed to
      be `clear` and paintable. If a goal tile is painted with the wrong color,
      the problem might be unsolvable under the given domain rules (no unpaint action).
      This heuristic assumes such states are not encountered or handled elsewhere.
    - Robots can change color instantly if the target color is available (1 action).
    - The cost of painting a tile is 1 action.
    - The heuristic calculates the cost for each unpainted goal tile independently,
      finding the minimum cost among all robots for that specific tile. This
      ignores potential synergies or conflicts when multiple robots paint multiple tiles.

    # Heuristic Initialization
    - Extracts the goal conditions from the task, specifically identifying which
      tiles need to be painted and with which color.
    - Stores these goal requirements in a dictionary mapping tile names to the
      required color.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted tile color)`.
    2. For each such goal fact, check if the corresponding `(painted tile color)`
       predicate is true in the current state.
    3. Collect the set of goal tiles that are not yet painted with the correct
       color according to the goal.
    4. If no tiles need painting, the heuristic is 0 (goal state).
    5. Find the current location (tile name) and color of each robot from the
       current state facts.
    6. If there are tiles to paint but no robots, the problem is unsolvable,
       return infinity.
    7. Initialize the total heuristic value to 0.
    8. For each unpainted goal tile `T` requiring color `C`:
        a. Parse the tile name `T` to get its grid coordinates `T_pos = (tr, tc)`.
           If parsing fails, return infinity (invalid state).
        b. Initialize the minimum cost for this tile `min_cost_for_tile` to infinity.
        c. For each robot `R`:
            i. Get the robot's current tile name `R_loc_name` and color `R_color`.
            ii. Parse `R_loc_name` to get its grid coordinates `R_pos = (rr, rc)`.
                If parsing fails, return infinity (invalid state).
            iii. Calculate the estimated movement cost for robot `R` to reach the
                 vicinity of tile `T` using Manhattan distance: `move_cost = manhattan_distance(R_pos, T_pos)`.
            iv. Calculate the color change cost: 1 if robot `R`'s current color is
                not the required goal color `C`, otherwise 0.
            v. Calculate the total estimated cost for robot `R` to contribute to
               painting tile `T`: `cost_this_robot = move_cost + color_cost + 1`
               (where +1 is for the final paint action itself).
            vi. Update `min_cost_for_tile = min(min_cost_for_tile, cost_this_robot)`.
        d. Add `min_cost_for_tile` to the `total_heuristic`.
    9. Return the `total_heuristic`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        super().__init__(task)

        # Store goal requirements: map tile_name -> required_color
        self.goal_tiles = {}
        for goal in self.goals:
            # We only care about (painted tile color) goals
            parts = get_parts(goal)
            if parts[0] == "painted" and len(parts) == 3:
                tile_name, color = parts[1], parts[2]
                self.goal_tiles[tile_name] = color
            # Ignore other potential goal predicates if any

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state # state is a frozenset of fact strings

        # Identify unpainted goal tiles
        # A tile needs painting if it's a goal tile AND the goal painted fact
        # is NOT in the current state.
        tiles_to_paint = {} # Maps tile_name -> goal_color
        for goal_tile_name, goal_color in self.goal_tiles.items():
            required_fact = f"(painted {goal_tile_name} {goal_color})"
            if required_fact not in state:
                tiles_to_paint[goal_tile_name] = goal_color

        # If no tiles need painting, it's a goal state
        if not tiles_to_paint:
            return 0

        # Find current robot locations and colors
        robot_info = {} # Maps robot_name -> {'location': tile_name, 'color': color_name}
        # Iterate through the state facts to find robot info
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at' and len(parts) == 3:
                robot_name, tile_name = parts[1], parts[2]
                if robot_name not in robot_info:
                    robot_info[robot_name] = {}
                robot_info[robot_name]['location'] = tile_name
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot_name, color_name = parts[1], parts[2]
                if robot_name not in robot_info:
                    robot_info[robot_name] = {}
                robot_info[robot_name]['color'] = color_name
            # Assuming robot-at and robot-has facts exist for all robots in valid states

        # If there are tiles to paint but no robots, the problem is unsolvable
        if tiles_to_paint and not robot_info:
             return float('inf')

        total_heuristic = 0

        # Calculate minimum cost for each tile that needs painting
        for tile_name, goal_color in tiles_to_paint.items():
            try:
                tile_pos = parse_tile_name(tile_name)
            except ValueError as e:
                # Handle parsing error - indicates a problem with the instance
                print(f"Heuristic parsing error for tile {tile_name}: {e}")
                return float('inf') # Indicate an invalid state/tile name

            min_cost_for_tile = float('inf')

            # Consider each robot as a potential candidate to paint this tile
            for robot_name, info in robot_info.items():
                # Ensure robot has both location and color info (should be true in valid states)
                if 'location' in info and 'color' in info:
                    robot_loc_name = info['location']
                    robot_color = info['color']

                    try:
                        robot_pos = parse_tile_name(robot_loc_name)
                    except ValueError as e:
                         print(f"Heuristic parsing error for robot location {robot_loc_name}: {e}")
                         return float('inf') # Indicate an invalid state/robot location name

                    # Estimated moves to get near the tile
                    # Using Manhattan distance between robot's tile and target tile
                    move_cost = manhattan_distance(robot_pos, tile_pos)

                    # Cost to change color if needed
                    color_cost = 1 if robot_color != goal_color else 0

                    # Total estimated cost for this robot to paint this specific tile
                    # move_cost + color_change_cost + paint_action_cost
                    cost_this_robot = move_cost + color_cost + 1

                    min_cost_for_tile = min(min_cost_for_tile, cost_this_robot)
                # else: Robot info is incomplete, skip this robot for this tile calculation

            # Add the minimum cost for this unpainted tile to the total heuristic
            # min_cost_for_tile should be finite here because we checked for no robots earlier
            total_heuristic += min_cost_for_tile

        return total_heuristic
