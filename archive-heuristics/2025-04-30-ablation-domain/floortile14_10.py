# Add necessary imports
from heuristics.heuristic_base import Heuristic
import math # For float('inf')

# Helper function to extract the components of a PDDL fact string.
# Example: "(predicate_name object1 object2)" -> ["predicate_name", "object1", "object2"]
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # print(f"Warning: get_parts received non-fact string: {fact}")
        return [] # Return empty list for invalid facts

    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

# Helper function to parse tile names like "tile_X_Y" into coordinates.
# Assumes tile names follow the format "tile_<row>_<col>".
def parse_tile_name(tile_name):
    """Parses a tile name string 'tile_X_Y' into integer coordinates (X, Y)."""
    try:
        parts = tile_name.split('_')
        # Assuming format is tile_row_col
        # Based on PDDL examples like (up tile_1_1 tile_0_1), the first number
        # seems to represent the row index, and the second the column index.
        # Movement predicates define adjacency:
        # (up tile_r1_c1 tile_r2_c2) implies r1 = r2 + 1, c1 = c2
        # (down tile_r1_c1 tile_r2_c2) implies r1 = r2 - 1, c1 = c2
        # (left tile_r1_c1 tile_r2_c2) implies r1 = r2, c1 = c2 + 1
        # (right tile_r1_c1 tile_r2_c2) implies r1 = r2, c1 = c2 - 1
        # We map tile_r_c to coordinates (r, c).
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (IndexError, ValueError):
        # Handle unexpected tile name formats if necessary, or return None/raise error
        # print(f"Warning: Could not parse tile name format: {tile_name}")
        return None # Indicate parsing failure

# Helper function to calculate Manhattan distance between two tiles given their names and coordinate map.
def manhattan_distance(tile1_name, tile2_name, tile_coords):
    """Calculates the Manhattan distance between two tiles given their names and coordinate map."""
    coords1 = tile_coords.get(tile1_name)
    coords2 = tile_coords.get(tile2_name)
    if coords1 is None or coords2 is None:
        # This indicates a problem with parsing or state data - the tile coordinates
        # were not successfully extracted during initialization.
        # print(f"Warning: Coordinates not found for {tile1_name} or {tile2_name}")
        return float('inf') # Indicate unreachable

    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[2])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the estimated cost for each unsatisfied
    goal independently, considering the cost of painting, acquiring the correct
    color, and moving a robot to the tile. It is designed for greedy best-first
    search and does not guarantee admissibility.

    # Assumptions
    - Robots can move between adjacent tiles (up, down, left, right) with a cost of 1 per step.
      The grid structure is defined by the static predicates (up, down, left, right).
    - Robots can hold at most one color at a time.
    - Picking up a color costs 1 action if the robot is free, and 2 actions
      (drop current color + pickup new color) if the robot holds a different color.
    - Painting a tile costs 1 action.
    - The 'clear' predicate is treated as static based on the provided examples
      and does not impose dynamic constraints on robot movement or painting in this heuristic calculation.
    - All colors required by the goal are available for pickup.
    - At least one robot exists in the initial state.

    # Heuristic Initialization
    - Extracts the target color for each tile specified in the goal conditions.
    - Parses the grid structure from static 'up', 'down', 'left', 'right' predicates
      and goal facts to map tile names to grid coordinates (row, col).
    - Identifies all robot names from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify all goal conditions of the form `(painted ?t ?c)`. Store these as
        a mapping from tile to target color in the constructor (`__init__`).
    2.  From the static facts defining the grid connectivity (`up`, `down`, `left`, `right`)
        and the tiles mentioned in the goals, build a mapping from each tile name
        (e.g., "tile_1_5") to its grid coordinates (row, col). This mapping is stored
        in the constructor and used to calculate Manhattan distances between tiles.
    3.  Identify all robot names present in the initial state and store them in the constructor.
    4.  In the `__call__` method, for the given state:
        a.  Determine the current location of each robot by finding facts like `(robot-at ?r ?t)`
            in the current state.
        b.  Determine the color currently held by each robot (if any) by finding facts like
            `(robot-has ?r ?c)` in the current state.
        c.  Identify the set of goal tiles that are *not* currently painted with their
            target color according to the state. This is done by checking if the fact
            `(painted t c)` for a goal tile `t` and its target color `c` is present
            in the current state.
        d.  Initialize the total heuristic cost to 0.
        e.  For each tile `t` and its target color `c` in the set of unsatisfied goals:
            i.  Add 1 to the total cost. This represents the final `paint` action required
                for this specific goal tile.
            ii. Calculate the minimum cost for *any* robot `r` to reach a state where it
                is ready to paint tile `t` with color `c`. This minimum cost is found by
                considering each robot `r` and summing two components:
                -   The cost for robot `r` to acquire color `c` at its *current* location.
                    This cost is 0 if `r` already has `c`, 1 if `r` is free (has no color),
                    and 2 if `r` has a different color (needs a `drop` action followed by a `pickup` action).
                -   The Manhattan distance from robot `r`'s current location to tile `t`.
                    This represents the minimum number of `move` actions required.
            iii. Add this minimum cost (over all robots) to the total heuristic cost.
                This step is where the non-admissibility comes from, as it calculates
                the minimum cost for each goal independently, potentially double-counting
                robot movements or color acquisitions if one robot/action helps satisfy
                multiple goals.
        f.  Return the total heuristic cost. If any goal tile is unreachable by all robots
            (e.g., due to missing coordinate data), return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, tile coordinates,
        and robot names from the task.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # 1. Extract goal paintings (tile -> color)
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # 2. Build tile coordinates map (tile_name -> (row, col))
        self.tile_coords = {}
        # Collect all tile names first from static facts and goals
        all_tile_names = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            # Look for predicates that define connectivity between tiles
            if parts and parts[0] in ["up", "down", "left", "right"] and len(parts) == 3:
                all_tile_names.add(parts[1])
                all_tile_names.add(parts[2])
        # Also include tiles mentioned in the goals
        for tile in self.goal_paintings.keys():
             all_tile_names.add(tile)

        # Parse coordinates for all identified tiles
        for tile_name in all_tile_names:
             coords = parse_tile_name(tile_name)
             if coords is not None:
                 self.tile_coords[tile_name] = coords
             # else: print(f"Could not parse coordinates for tile: {tile_name}") # Optional warning

        # 3. Identify robot names from the initial state
        self.robots = set()
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3:
                robot_name = parts[1]
                self.robots.add(robot_name)

        # Store available colors (not strictly needed for this heuristic logic, but good practice)
        # self.available_colors = {get_parts(fact)[1] for fact in self.static_facts if get_parts(fact)[0] == "available-color"}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 4a. Determine current robot locations
        current_robot_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3:
                robot_name, tile_name = parts[1], parts[2]
                current_robot_locations[robot_name] = tile_name

        # 4b. Determine current robot colors
        current_robot_colors = {}
        # Initialize all robots as having no color
        for robot in self.robots:
             current_robot_colors[robot] = None
        # Update for robots that have a color
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-has" and len(parts) == 3:
                robot_name, color_name = parts[1], parts[2]
                current_robot_colors[robot_name] = color_name

        # 4c. Identify unsatisfied goals
        current_painted_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile_name, color_name = parts[1], parts[2]
                current_painted_tiles.add((tile_name, color_name))

        unpainted_goals = [(tile, color) for tile, color in self.goal_paintings.items()
                           if (tile, color) not in current_painted_tiles]

        # If all goals are met, heuristic is 0
        if not unpainted_goals:
            return 0

        # 4d. Initialize total cost
        total_cost = 0

        # 4e. For each unsatisfied goal
        for tile_to_paint, target_color in unpainted_goals:
            # i. Add cost for paint action
            total_cost += 1

            # ii. Calculate minimum cost for any robot to be ready for this goal
            min_robot_ready_cost = float('inf')

            # If there are no robots, this goal is unreachable from the start
            if not self.robots:
                 return float('inf')

            for robot in self.robots:
                robot_current_tile = current_robot_locations.get(robot)
                if robot_current_tile is None:
                    # Robot location not found in state? Problematic state representation.
                    # Treat this robot as unable to satisfy the goal from this state.
                    continue

                # Cost for robot to acquire target_color at its current location
                robot_has_color = current_robot_colors.get(robot)

                if robot_has_color == target_color:
                    color_acquisition_cost = 0
                elif robot_has_color is not None: # Has a different color
                    color_acquisition_cost = 2 # Drop + Pickup
                else: # Has no color
                    color_acquisition_cost = 1 # Pickup

                # Distance cost for robot to move to the tile
                move_cost = manhattan_distance(robot_current_tile, tile_to_paint, self.tile_coords)

                # If move_cost is inf, this robot cannot reach the tile (e.g., tile coords not parsed)
                if move_cost == float('inf'):
                    continue

                # Total cost for this robot to be ready for this specific goal
                robot_ready_cost = color_acquisition_cost + move_cost

                # Update minimum cost over all robots for this goal
                min_robot_ready_cost = min(min_robot_ready_cost, robot_ready_cost)

            # Add the minimum robot ready cost for this goal to the total
            # If min_robot_ready_cost is still inf, it means no robot can reach this goal tile
            if min_robot_ready_cost == float('inf'):
                 # This goal is unreachable by any robot whose location was found.
                 # The state is likely unsolvable or malformed.
                 return float('inf')

            total_cost += min_robot_ready_cost

        # 4f. Return total heuristic cost
        return total_cost
