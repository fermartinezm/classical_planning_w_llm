# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic
# Define a dummy Heuristic class for standalone testing if needed
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

import math
from fnmatch import fnmatch

# Helper functions (adapted from examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Return as is if it doesn't look like a standard PDDL fact string
         return [fact_str]
    return fact_str[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_2 black)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-specific helper for parsing tile names
def parse_tile_coords(tile_name):
    """Parses a tile name string like 'tile_R_C' into a (row, col) tuple of integers."""
    try:
        # Expecting format 'tile_R_C'
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected format, maybe return None
            return None
    except (ValueError, IndexError):
        # Handle cases where conversion to int fails or parts are missing
        return None

# Domain-specific helper for Manhattan distance
def manhattan_distance(tile1_name, tile2_name):
    """Calculates the Manhattan distance between two tiles given their names."""
    coords1 = parse_tile_coords(tile1_name)
    coords2 = parse_tile_coords(tile2_name)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance if parsing failed
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles
    with the correct color. It sums the number of unsatisfied painted goal facts,
    an estimate of color changes, and the minimum movement cost to reach any
    tile involved in an unsatisfied painted goal.

    # Assumptions
    - The goal is defined primarily by a set of `(painted tile color)` facts.
    - Tiles cannot be repainted once painted (based on action preconditions).
      Wrongly painted tiles are counted as unsatisfied goals but the heuristic
      doesn't explicitly model the steps needed to fix them (e.g., clearing).
    - The robot always possesses exactly one color.
    - Movement cost is approximated by Manhattan distance on the grid. The heuristic
      ignores the `clear` precondition for movement distance calculation, assuming
      paths are generally available.

    # Heuristic Initialization
    - Extracts all goal conditions of the form `(painted tile color)` into a
      set for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Check for Goal State:** The heuristic must return 0 *only* when the goal state is reached.
        Check if the set of goal facts (`self.goals`) is a subset of the current state facts (`node.state`). If it is, return 0.

    2.  **Identify Robot State:** If not the goal state, determine the robot's current location (`robot_loc_str`)
        and the color it is currently holding (`robot_color_str`) by examining the
        facts in the current state. This assumes a single robot and that it always
        has a location and a color. If robot state cannot be determined (e.g., missing facts), return infinity.

    3.  **Identify Unsatisfied Painted Goals:** Iterate through all goal conditions
        that specify a tile should be painted with a certain color (`(painted tile color)`), which were extracted during initialization.
        For each such goal fact, check if it is present in the current state. Collect
        all goal `(painted ...)` facts that are *not* present in the current state
        into a list of `(tile_name, required_color)` tuples. These represent the
        unsatisfied painted goals.

    4.  **Handle No Unsatisfied Painted Goals (but not goal state):** If the list of unsatisfied painted goals is empty,
        but the check in step 1 determined that the full goal is *not* met, it implies
        there are other goal conditions (not related to painted tiles, or perhaps
        complex interactions not captured by the simple painted fact check) that are
        not satisfied. In this specific case, return a small positive value (e.g., 1)
        to indicate that the state is not the goal, even though the painted tile
        conditions covered by this heuristic seem met. This prevents the heuristic
        from incorrectly returning 0.

    5.  **Calculate Heuristic Components:** If there are unsatisfied painted goals,
        calculate the estimated cost:
        *   **Painting Cost:** Add the number of unsatisfied painted goals (`len(unsatisfied_painted_goals)`).
            This is a lower bound on the number of `paint` actions required, assuming
            each unsatisfied goal tile needs one paint action.
        *   **Color Change Cost:** Determine the set of distinct colors required for
            the unsatisfied painted goals. Estimate the number of `change_color` actions
            needed. This is approximated by the number of distinct colors needed,
            minus 1 if the robot already holds one of these needed colors (as the
            first set of paintings can potentially use the current color without a change).
        *   **Movement Cost:** Estimate the cost to move the robot closer to the
            tiles involved in the unsatisfied painted goals. Calculate the Manhattan
            distance from the robot's current location to the tile associated with
            *each* unsatisfied painted goal. Add the *minimum* of these distances
            to the cost. This estimates the cost to reach the vicinity of the nearest
            tile that needs attention.

    6.  **Sum Costs:** The total heuristic value is the sum of the Painting Cost,
        Color Change Cost, and Movement Cost components.

    7.  **Return Heuristic Value:** Return the calculated total cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals

        # Store goal facts related to painted tiles for quick lookup.
        self.goal_painted_facts = {
            goal_fact for goal_fact in self.goals
            if match(goal_fact, 'painted', '*', '*')
        }

        # Static facts are available but not strictly needed for this heuristic's calculation
        # self.static = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check for Goal State. The heuristic must be 0 ONLY at the goal state.
        if self.goals <= state:
             return 0

        # 2. Identify Robot State.
        robot_loc_str = None
        robot_color_str = None
        # Assuming only one robot and it always has a location and color
        for fact in state:
            if match(fact, 'robot-at', '*', '*'):
                robot_loc_str = get_parts(fact)[2]
            elif match(fact, 'robot-has', '*', '*'):
                 robot_color_str = get_parts(fact)[2]
            # Break early if both found
            if robot_loc_str and robot_color_str:
                 break

        # Should not happen in valid problem states, but as a safeguard
        if not robot_loc_str or not robot_color_str:
             # Cannot compute heuristic without robot state
             return float('inf')


        # 3. Identify Unsatisfied Painted Goals.
        unsatisfied_painted_goals = [] # List of (tile_name, required_color) tuples

        for goal_fact in self.goal_painted_facts:
             if goal_fact not in state:
                  # Extract tile and color from the goal fact string
                  parts = get_parts(goal_fact)
                  if len(parts) == 3: # Ensure it's a valid painted fact
                      unsatisfied_painted_goals.append((parts[1], parts[2]))


        # 4. Handle No Unsatisfied Painted Goals (but not goal state).
        if not unsatisfied_painted_goals:
             # We already checked self.goals <= state at the top.
             # If we are here, it's not the goal state, but all painted goals are met.
             # This implies there are other goal conditions not met.
             # A heuristic of 1 is a simple non-zero value to guide search away from this non-goal state.
             return 1


        # 5. Calculate Heuristic Components.
        cost = 0

        # Add cost for painting/fixing unsatisfied tiles.
        cost += len(unsatisfied_painted_goals)

        # Estimate cost for color changes
        colors_needed = {color for (tile, color) in unsatisfied_painted_goals}
        # Number of color changes = Number of distinct colors needed - 1 (if current color is one of them)
        cost += len(colors_needed) - (1 if robot_color_str in colors_needed else 0)

        # Estimate cost for movement (distance to the closest tile in unsatisfied painted goals)
        min_dist = float('inf')
        robot_coords = parse_tile_coords(robot_loc_str)

        # Only calculate distance if robot location was parsed successfully
        if robot_coords is not None:
            for tile, color in unsatisfied_painted_goals:
                dist = manhattan_distance(robot_loc_str, tile)
                min_dist = min(min_dist, dist)

        # Add the minimum distance (cost to reach the vicinity of the first tile)
        # Only add if a finite minimum distance was found (i.e., there were unsatisfied goals and robot loc parsed)
        if min_dist != float('inf'):
             cost += min_dist

        # 6. Return the total calculated cost.
        return cost
