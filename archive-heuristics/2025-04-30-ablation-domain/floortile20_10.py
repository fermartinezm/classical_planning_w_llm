# from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is provided

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
         return [] # Return empty list for invalid fact strings
    return fact[1:-1].split()

# Helper function to parse tile names
def get_tile_coords(tile_name):
    """
    Parses a tile name like 'tile_row_col' into (row, col) integer coordinates.
    Assumes tile names follow this pattern. Returns None if parsing fails.
    """
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            # Convert row and column parts to integers
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col parts are not valid integers
            return None
    # Handle cases where the name doesn't start with 'tile_' or doesn't have 3 parts
    return None

# Helper function for Manhattan distance
def manhattan_distance(coords1, coords2):
    """Calculates the Manhattan distance between two (row, col) integer coordinates."""
    # Assumes coords1 and coords2 are valid (row, col) tuples
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[2])


# Define the heuristic class
# class floortileHeuristic(Heuristic): # Uncomment this line if inheriting from Heuristic
class floortileHeuristic: # Use this if Heuristic base is not provided or for standalone
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums up the estimated cost for each unpainted
    goal tile independently. The estimated cost for a single tile is the sum of:
    1. The cost of the paint action (1).
    2. The cost of a pick-color action (1), if the robot chosen to paint the tile
       does not already have the required color.
    3. The minimum Manhattan distance from any robot's current location to the
       goal tile location (representing movement cost).

    # Assumptions
    - Tiles are named using the pattern 'tile_row_col', allowing Manhattan distance calculation.
    - Robots can only hold one color at a time, and picking a new color replaces the old one.
    - All colors required by the goal are available (implied by static 'available-color' facts).
    - Tiles are initially 'clear' if unpainted and become 'not clear' when painted.
      The heuristic assumes an unpainted goal tile is ready to be painted.
    - Tiles painted with the wrong color are not considered (assumed not to occur
      on optimal paths or handled implicitly by search).
    - The cost of any action (move, pick, paint) is 1.

    # Heuristic Initialization
    - Extracts the goal conditions (which tiles need which colors).
    - Identifies all robots present in the problem instance based on initial state facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify the current location and color held by each robot in the state.
    3. Iterate through each goal condition specifying a tile and its target color (e.g., `(painted tile_1_1 white)`).
    4. For each goal `(painted T C)`:
       a. Check if the fact `(painted T C)` is already true in the current state. If it is, this goal is satisfied, and the cost for this goal is 0. Continue to the next goal.
       b. If the goal `(painted T C)` is not satisfied:
          i. This tile `T` needs to be painted with color `C`.
          ii. Calculate the minimum estimated cost to paint this tile using any available robot.
          iii. Get the coordinates `(tr, tc)` for tile `T`. If parsing fails, return infinity as the heuristic value (indicating an invalid goal tile).
          iv. Initialize `min_cost_for_this_goal = float('inf')`.
          v. For each robot `R`:
              - Get the robot's current tile location `Tr` from the state. If the robot's location is unknown, skip this robot for this goal tile.
              - Get the coordinates `(rtr, rtc)` for tile `Tr`. If parsing fails, skip this robot for this goal tile.
              - Get the robot's current color `Cr` from the state. If the robot has no color, `Cr` is considered `None`.
              - Calculate the estimated movement cost: `manhattan_distance((rtr, rtc), (tr, tc))`.
              - Determine the estimated pick-color cost: 1 if robot `R` does not have color `C` (i.e., `Cr != C`), and 0 if robot `R` already has color `C` (i.e., `Cr == C`).
              - The estimated cost for robot `R` to paint tile `T` is: movement_cost + pick_color_cost + 1 (for the paint action itself).
              - Update `min_cost_for_this_goal = min(min_cost_for_this_goal, cost_for_this_robot)`.
          vi. If after checking all robots, `min_cost_for_this_goal` is still infinity, it means no robot could be used to paint this tile (e.g., no robots found, or all robot locations invalid). Return infinity as the heuristic value (indicating an unreachable goal).
          vii. Add `min_cost_for_this_goal` to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying robots.
        """
        # super().__init__(task) # Uncomment this line if inheriting from Heuristic
        self.goals = task.goals  # Goal conditions (frozenset of facts)

        # Identify all robots present in the problem instance from initial state facts
        # This assumes all relevant robots are mentioned in robot-at or robot-has in the initial state.
        self.robots = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if len(parts) >= 2 and (parts[0] == 'robot-at' or parts[0] == 'robot-has'):
                 self.robots.add(parts[1])

        # Store goal tiles and their required colors for quick lookup
        # { 'tile_name': 'color_name', ... }
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color
            # Note: Other goal types are ignored by this heuristic.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of facts)

        # If the goal is already reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        # Track current robot locations and colors
        robot_locations = {} # { robot_name: tile_name }
        robot_colors = {}    # { robot_name: color_name }

        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif len(parts) == 3 and parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            # Note: Facts about painted tiles or clear tiles are not explicitly stored
            # in separate dictionaries as they are checked directly against the state set.

        total_cost = 0

        # Iterate through each tile that needs to be painted according to the goal
        for goal_tile, required_color in self.goal_tiles.items():
            # Check if this specific goal (tile painted with required color) is met
            goal_fact = f'(painted {goal_tile} {required_color})'
            if goal_fact in state:
                continue # This goal is already satisfied

            # This tile needs painting. Estimate the cost.
            tile_coords = get_tile_coords(goal_tile)
            if tile_coords is None:
                 # If goal tile name is invalid, this goal is likely unreachable or problem is malformed.
                 # Return infinity to indicate this state is likely a dead end.
                 return float('inf')

            min_cost_for_this_goal = float('inf')

            # Find the minimum cost using any robot
            for robot in self.robots:
                robot_location = robot_locations.get(robot)
                if robot_location is None:
                    # Robot location unknown in this state. Cannot use this robot for estimation.
                    continue

                robot_coords = get_tile_coords(robot_location)
                if robot_coords is None:
                    # If robot location tile name is invalid, cannot use this robot.
                    # This might indicate an issue with the state representation.
                    continue

                # Calculate movement cost
                move_cost = manhattan_distance(robot_coords, tile_coords)

                # Calculate pick color cost
                current_color = robot_colors.get(robot)
                # Cost is 1 if the robot doesn't have the required color
                # If current_color is None (robot has no color), current_color != required_color is True.
                pick_cost = 1 if current_color != required_color else 0

                # Total estimated cost for this robot to paint this tile
                # move + pick (if needed) + paint
                cost_for_this_robot = move_cost + pick_cost + 1

                min_cost_for_this_goal = min(min_cost_for_this_goal, cost_for_this_robot)

            # If after checking all robots, min_cost_for_this_goal is still infinity,
            # it means no robot could be used to paint this tile (e.g., no robots found, or all robot locations invalid).
            # For a solvable problem, there must be at least one robot with a valid location.
            # If not, this state is likely unsolvable. Return infinity.
            if min_cost_for_this_goal == float('inf'):
                 return float('inf')

            total_cost += min_cost_for_this_goal

        return total_cost
