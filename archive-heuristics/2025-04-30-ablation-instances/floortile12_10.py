# Assuming heuristics.heuristic_base provides the Heuristic base class
# from heuristics.heuristic_base import Heuristic

# Helper functions (outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_R_C' into (R, C) integer coordinates."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Log a warning or handle unexpected format
            # print(f"Warning: Unexpected tile name format: {tile_name}")
            return None
    except (ValueError, IndexError):
        # Log an error or handle parsing failure
        # print(f"Error parsing tile name {tile_name}: {e}")
        return None

def min_dist_to_adjacent(robot_coords, target_coords):
    """
    Calculates the minimum Manhattan distance from robot_coords to any tile
    adjacent to target_coords.
    """
    if robot_coords is None or target_coords is None:
        return float('inf')

    rr, rc = robot_coords
    tr, tc = target_coords

    # Distances to the four potential adjacent tiles:
    # (tr+1, tc), (tr-1, tc), (tr, tc+1), (tr, tc-1)
    # Note: We don't need to check if these adjacent tiles exist in the grid,
    # as Manhattan distance is a theoretical grid distance.
    dist1 = abs(rr - (tr + 1)) + abs(rc - tc) # To tile below target
    dist2 = abs(rr - (tr - 1)) + abs(rc - tc) # To tile above target
    dist3 = abs(rr - tr) + abs(rc - (tc + 1)) # To tile right of target
    dist4 = abs(rr - tr) + abs(rc - (tc - 1)) # To tile left of target

    return min(dist1, dist2, dist3, dist4)


# Import the base class as required
from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the estimated costs for each unsatisfied goal tile,
    considering the paint action itself, the movement required to get adjacent to the tile,
    and the color changes needed.

    # Assumptions
    - The grid structure is implied by tile names like 'tile_R_C'.
    - The robot is the only entity that can paint tiles.
    - Tiles are painted once and remain painted (no unpainting actions).
    - Goal tiles are initially either unpainted or painted with the correct color
      (i.e., no goal tile is initially painted with the wrong color).
    - The cost of each action (move, paint, change_color) is 1.
    - The minimum movement cost to get adjacent to a target tile T from robot location L
      is estimated as the minimum Manhattan distance from L to any of T's four neighbors.
    - The color change cost is estimated based on the number of distinct colors
      required for the currently unsatisfied goal tiles.

    # Heuristic Initialization
    - Stores the set of goal facts from the task.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Identify Robot State:** Find the robot's current location (tile name) and the color it is currently holding by examining the state facts (`robot-at` and `robot-has` predicates). Parse the robot's tile name to get its (row, column) coordinates.
    2. **Initialize Cost and Trackers:** Initialize the total heuristic cost `h` to 0. Create an empty set `colors_needed` to keep track of the distinct colors required for tiles that are not yet painted correctly according to the goal.
    3. **Identify Unsatisfied Painted Goals:** Iterate through each goal fact provided in `self.goals`. If a goal fact is of the form `(painted T C)` and this exact fact string is *not* present in the current state, then tile `T` is not yet painted with the required color `C`. Add `(T, C)` to a list of unsatisfied goal tiles and add `C` to the `colors_needed` set.
    4. **Calculate Cost for Each Unsatisfied Tile:** Iterate through the list of unsatisfied goal tiles `(tile_name, goal_color)`:
       a. Add 1 to `h` for the required `paint` action for this tile.
       b. Parse the `tile_name` to get its (row, column) coordinates `target_coords`.
       c. Calculate the minimum Manhattan distance from the robot's current coordinates (`robot_coords`) to *any* of the four tiles adjacent to `target_coords`. This is the estimated movement cost to get into a position from which the tile can be painted. Add this movement cost to `h`.
    5. **Calculate Color Change Cost:** After processing all unsatisfied tiles, determine the cost associated with changing colors.
       a. If the `colors_needed` set is empty (meaning all painted goals are satisfied), the color change cost is 0.
       b. If `colors_needed` is not empty:
          i. If the robot's current color (`robot_color`) is one of the colors in `colors_needed`, the robot can start painting tiles of that color immediately. It will need to perform at least `len(colors_needed) - 1` color changes to be able to paint tiles of all other needed colors. Add `max(0, len(colors_needed) - 1)` to `h`.
          ii. If the robot's current color (`robot_color`) is *not* in `colors_needed`, the robot must first change to one of the needed colors. It will then need to perform at least `len(colors_needed) - 1` further changes to cover all needed colors. The total minimum changes is `1 + (len(colors_needed) - 1) = len(colors_needed)`. Add `len(colors_needed)` to `h`.
    6. **Return Total Cost:** The final value of `h` is the estimated total number of actions (paint, move, color change) required to reach the goal state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals  # Goal conditions are a frozenset of facts

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify robot's current location and color
        robot_location_str = None
        robot_color = None
        # Assuming one robot, find its state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'robot-at' and len(parts) == 3:
                # parts[1] is robot name, parts[2] is tile name
                robot_location_str = parts[2]
            elif parts[0] == 'robot-has' and len(parts) == 3:
                # parts[1] is robot name, parts[2] is color
                robot_color = parts[2]

        # Check if robot state was found (should always be the case in valid states)
        if robot_location_str is None or robot_color is None:
             # This indicates an unexpected state structure
             # print("Error: Robot location or color not found in state.")
             return float('inf') # Return infinity as this state is likely invalid/unreachable

        robot_coords = parse_tile_name(robot_location_str)
        if robot_coords is None:
             # Handle parsing error for robot location tile
             # print(f"Error: Could not parse robot tile name {robot_location_str}")
             return float('inf') # Return infinity


        # 2. Initialize heuristic cost and needed colors tracker
        total_cost = 0
        colors_needed = set()

        # 3. Identify unsatisfied painted goals
        unsatisfied_painted_goals = [] # List of (tile_name, goal_color)

        for goal_fact in self.goals:
             # Check if the goal fact is a painted predicate
             parts = get_parts(goal_fact)
             if parts and parts[0] == 'painted' and len(parts) == 3:
                  # This is a painted tile goal fact
                  # Check if this exact fact string is NOT in the current state
                  if goal_fact not in state:
                       # This goal fact is not satisfied
                       tile_name = parts[1]
                       goal_color = parts[2]
                       unsatisfied_painted_goals.append((tile_name, goal_color))
                       colors_needed.add(goal_color)

        # 4. Calculate cost for each unsatisfied tile
        for tile_name, goal_color in unsatisfied_painted_goals:
             # Add 1 for the paint action
             total_cost += 1

             # Parse target tile coordinates
             target_coords = parse_tile_name(tile_name)
             if target_coords is None:
                  # Handle parsing error for target tile
                  # print(f"Error: Could not parse target tile name {tile_name}")
                  return float('inf') # Return infinity

             # Calculate minimum movement cost to get adjacent
             move_cost = min_dist_to_adjacent(robot_coords, target_coords)
             total_cost += move_cost

        # 5. Calculate color change cost
        if colors_needed: # Only add color cost if there are tiles to paint
             if robot_color in colors_needed:
                  # Robot has one of the needed colors
                  total_cost += max(0, len(colors_needed) - 1)
             else:
                  # Robot has a color not needed
                  total_cost += len(colors_needed)

        # 6. Return the total heuristic cost
        return total_cost
