# Helper functions (get_parts, parse_tile_coords, manhattan_distance)
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_tile_coords(tile_name):
    """Parses tile name 'tile_R_C' into (row, col) integers."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Return None if the object name doesn't match the tile pattern
    return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coords1 = parse_tile_coords(tile1_name)
    coords2 = parse_tile_coords(tile2_name)
    if coords1 is None or coords2 is None:
        # This indicates an issue, maybe one of the objects is not a tile
        # Return a large value or infinity as a fallback
        return float('inf')
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

# Heuristic class
# Assuming heuristics.heuristic_base provides a base class named Heuristic
from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing the estimated costs for each individual goal tile that is not
    yet painted correctly. The cost for each unpainted goal tile includes
    the paint action itself, the estimated movement cost for the robot to
    reach the tile, and the estimated cost for the robot to acquire the
    necessary color.

    # Assumptions
    - The heuristic assumes a grid structure for tiles where movement costs
      correspond to Manhattan distance between tile coordinates derived from
      tile names (e.g., 'tile_R_C'). The static facts defining adjacency
      (up, down, left, right) imply this structure, but the heuristic
      calculates distance directly from tile names rather than building
      a graph from these facts.
    - The cost to acquire a color depends only on the robot's current color
      and the color needed for the specific tile being considered, ignoring
      potential efficiencies from planning color acquisition for multiple tiles.
    - The heuristic currently considers the state of only one robot (the first
      one found in the state facts) for calculating movement and color costs.
      If multiple robots exist, this heuristic calculates the cost based on
      one robot's state and sums independent costs per tile, which might not
      accurately reflect multi-robot coordination.

    # Heuristic Initialization
    - Extracts the goal conditions from the task to identify which tiles need
      to be painted and with which colors. This information is stored in
      `self.goal_tiles`. Static facts are not explicitly processed in init
      as coordinate parsing is done on demand.

    # Step-By-Step Thinking for Computing Heuristic
    1. Find the current location (`robot-at`) and color (`robot-has`) of a robot
       from the state facts. The heuristic uses the state of the first robot
       information encountered. If the robot has no color, its color state is
       represented internally as 'none'.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each goal condition `(painted tile color)` that was
       extracted during initialization.
    4. For each goal `(painted T C)`:
       - Check if the fact `(painted T C)` is present in the current state.
       - If the tile `T` is *not* painted with the required color `C` in the
         current state (i.e., the goal for this tile is unsatisfied):
         - Add 1 to the total cost (representing the `paint` action needed
           specifically for this tile).
         - If a robot's location was successfully identified in step 1:
           - Calculate the Manhattan distance between the robot's current tile
             and tile `T` using their parsed coordinates. Add this distance
             to the total cost (representing the estimated movement cost to
             reach this tile).
           - Determine the color acquisition cost for this tile:
             - If the robot currently has color `C`, the cost is 0.
             - If the robot currently has no color ('none'), the cost is 1
               (to pick up color `C`).
             - If the robot currently has a color different from `C`, the cost
               is 2 (to drop the current color and pick up color `C`).
           - Add this color acquisition cost to the total cost.
         - If no robot location was identified (unexpected in valid states),
           add a large penalty to indicate a potentially unreachable state
           or an issue with state parsing.
    5. Return the final total cost. The cost will be 0 if and only if all
       goal tiles are painted correctly, matching the goal state condition.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        Static facts are not explicitly used for this heuristic's calculation
        as tile coordinates are parsed directly from names.
        """
        self.goals = task.goals
        # Extract goal tiles and their required colors
        self.goal_tiles = {} # {tile_name: color}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to paint all goal tiles correctly.
        """
        state = node.state

        # Find robot location and color. Assumes at least one robot exists
        # and takes the state of the first robot information encountered.
        robot_location = None
        robot_color = 'none' # Sentinel value for no color
        robot_info_found = False # Track if we found any robot info

        # Iterate through state facts to find robot location and color
        # We assume facts for the same robot will appear together or we just take the first one found
        # In a multi-robot scenario, a more sophisticated heuristic might track all robots
        # and assign tasks, but this simple heuristic focuses on the work remaining per tile.
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "robot-at":
                # Assuming args[0] is robot name, args[1] is location
                robot_location = args[1]
                robot_info_found = True
            elif predicate == "robot-has":
                 # Assuming args[0] is robot name, args[1] is color
                robot_color = args[1]
                robot_info_found = True
            # Optimization: If we found both location and color for *a* robot, we can stop.
            # This is a simplification for the multi-robot case.
            if robot_location is not None and robot_color != 'none' and robot_info_found:
                 break


        total_cost = 0

        # Iterate through goal tiles and calculate cost for unsatisfied ones
        for tile, required_color in self.goal_tiles.items():
            # Check if the tile is already painted with the correct color
            is_painted_correctly = f"(painted {tile} {required_color})" in state

            if not is_painted_correctly:
                # This tile needs painting
                total_cost += 1 # Cost for the paint action

                # Add movement cost if robot location is known
                if robot_location:
                    # Ensure the goal object is actually a tile before calculating distance
                    if parse_tile_coords(tile) is not None:
                         total_cost += manhattan_distance(robot_location, tile)
                    else:
                         # This goal is not about painting a tile? Or tile name is malformed?
                         # Add a penalty or skip, depending on desired robustness.
                         # Assuming valid goal tiles for simplicity.
                         pass # Assuming valid tile names in goals
                else:
                    # Should not happen in valid states with a robot, but as a fallback
                    # add a penalty to make states with unknown robot location less desirable.
                    # This state is likely unreachable or invalid.
                    total_cost += 1000 # Penalty if robot location is unknown

                # Add color cost
                if robot_color == 'none':
                    total_cost += 1 # Need to pick up the color
                elif robot_color != required_color:
                    total_cost += 2 # Need to drop current color and pick up required color
                # If robot_color == required_color, color cost is 0

        # The heuristic is 0 if all goals are satisfied.
        # The loop above calculates cost only for unsatisfied goals.
        # If there are no unsatisfied goals, total_cost remains 0.
        # This satisfies the requirement that h=0 only for goal states.

        return total_cost
