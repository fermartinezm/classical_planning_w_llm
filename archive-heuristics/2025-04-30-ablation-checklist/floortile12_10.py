from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(robot-at robot1 tile_0_1)" -> ["robot-at", "robot1", "tile_0_1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_str):
    """
    Parses a tile name string like 'tile_row_col' into a (row, col) tuple of integers.
    Assumes tile names follow the format 'tile_R_C'.
    """
    try:
        parts = tile_str.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected format, maybe log a warning or raise an error
            print(f"Warning: Unexpected tile name format: {tile_str}")
            return None # Or raise ValueError
    except (ValueError, IndexError):
        print(f"Warning: Could not parse tile name: {tile_str}")
        return None # Or raise ValueError

def manhattan_distance(tile1_str, tile2_str):
    """
    Calculates the Manhattan distance between two tiles given their string names.
    Returns infinity if parsing fails.
    """
    coords1 = parse_tile_name(tile1_str)
    coords2 = parse_tile_name(tile2_str)

    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal
    tiles with their target colors. It sums the estimated cost for each
    unsatisfied goal tile, considering the minimum cost among all robots.

    # Assumptions
    - The grid structure is implicitly defined by tile names like 'tile_row_col'.
    - Movement cost is approximated by Manhattan distance.
    - The heuristic ignores the 'clear' predicate requirement for intermediate
      movement tiles, only considering the target tile's painted status.
    - If a goal tile is painted with the wrong color, the problem is considered
      unsolvable from that state (heuristic returns infinity).
    - Each paint action requires the robot to be adjacent to the tile. The
      Manhattan distance calculation is a lower bound on moves to get *near*
      the tile, plus 1 for the final move to an adjacent tile or adjusting.
      For simplicity, we use the Manhattan distance to the tile itself as the
      movement cost component, plus 1 for the paint action.

    # Heuristic Initialization
    - Stores the goal conditions.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted ?tile ?color)`.
    2. For each such goal fact `(painted T C)`:
       a. Check if `(painted T C)` is already true in the current state. If yes, this goal is satisfied, cost is 0 for this tile.
       b. Check if `(painted T C')` is true for any `C' != C` in the current state. If yes, the tile is painted with the wrong color, and the problem is likely unsolvable. Return infinity.
       c. If the goal is not satisfied and the tile is not painted with the wrong color (it must be 'clear' based on domain rules for painting), estimate the cost to paint it with color `C`.
       d. To estimate the cost for tile `T` and color `C`, consider all robots.
       e. For each robot `R`:
          i. Find its current location `R_loc` and current color `R_color` from the state.
          ii. Calculate the Manhattan distance `dist` between `R_loc` and `T`.
          iii. Calculate the estimated cost for *this* robot to paint *this* tile:
              - Movement cost: `dist` (approximating moves to get near the tile).
              - Color change cost: 1 if `R_color` is not `C`, otherwise 0.
              - Paint action cost: 1.
              - Total robot cost for this tile: `dist + (1 if R_color != C else 0) + 1`.
       f. The estimated cost for tile `T` is the minimum of the costs calculated for each robot.
    3. Sum the estimated costs for all unsatisfied, paintable goal tiles.
    4. Return the total sum.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals  # Goal conditions (frozenset of strings)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        total_cost = 0  # Initialize the heuristic cost.

        # Extract robot information from the current state
        robots_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot_name, tile_name = parts[1], parts[2]
                if robot_name not in robots_info:
                    robots_info[robot_name] = {}
                robots_info[robot_name]['location'] = tile_name
            elif parts[0] == 'robot-has':
                robot_name, color_name = parts[1], parts[2]
                if robot_name not in robots_info:
                    robots_info[robot_name] = {}
                robots_info[robot_name]['color'] = color_name
            # Note: 'free-color' predicate is not used in this heuristic as 'robot-has' is sufficient

        # Iterate through the goal conditions
        for goal_fact_str in self.goals:
            goal_parts = get_parts(goal_fact_str)

            # We only care about 'painted' goals for this heuristic
            if goal_parts[0] != 'painted':
                continue

            goal_tile, goal_color = goal_parts[1], goal_parts[2]

            # Check if the goal tile is already painted correctly
            if goal_fact_str in state:
                continue # Goal already satisfied

            # Check if the goal tile is painted with the wrong color
            is_wrongly_painted = False
            for state_fact_str in state:
                state_parts = get_parts(state_fact_str)
                if state_parts[0] == 'painted' and state_parts[1] == goal_tile:
                    # Found a painted fact for the goal tile
                    if state_parts[2] != goal_color:
                        is_wrongly_painted = True
                        break # Tile is painted with the wrong color

            if is_wrongly_painted:
                # If a goal tile is painted with the wrong color, it's unsolvable
                # in this domain as there's no unpaint action.
                return float('inf')

            # The goal tile needs to be painted. Estimate the minimum cost across all robots.
            min_cost_for_tile = float('inf')

            if not robots_info:
                 # No robots available to paint
                 return float('inf')

            for robot_name, info in robots_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                if robot_location is None or robot_color is None:
                    # Robot state is incomplete, cannot use this robot
                    continue

                # Calculate movement cost (Manhattan distance)
                move_cost = manhattan_distance(robot_location, goal_tile)

                # Calculate color change cost
                color_change_cost = 0
                if robot_color != goal_color:
                    color_change_cost = 1 # Assume 1 action to change color

                # Paint action cost
                paint_cost = 1 # 1 action to paint

                # Total estimated cost for this robot to paint this tile
                # Note: move_cost is distance to the tile, paint is from adjacent.
                # This is a relaxation, but dist + 1 is a common pattern.
                estimated_robot_cost = move_cost + color_change_cost + paint_cost

                # Update minimum cost for this tile
                min_cost_for_tile = min(min_cost_for_tile, estimated_robot_cost)

            # If no robot could potentially paint the tile (e.g., no robots, or parsing failed)
            if min_cost_for_tile == float('inf'):
                 return float('inf') # Problem is likely unsolvable

            total_cost += min_cost_for_tile

        return total_cost

# Example usage (assuming you have a Task object and a Node object)
# from task import Task, Operator # Assuming task.py contains these classes
# from node import Node # Assuming node.py contains Node class
#
# # Create a dummy task and state for demonstration
# # This part would typically be handled by the planner's loading mechanism
# dummy_task = Task(
#     name="dummy_floortile_task",
#     facts=set(), # Not needed for heuristic
#     initial_state=frozenset({'(robot-at robot1 tile_0_1)', '(robot-has robot1 black)', '(clear tile_1_1)', '(painted tile_1_2 white)'}),
#     goals=frozenset({'(painted tile_1_1 white)', '(painted tile_1_2 black)'}),
#     operators=[], # Not needed for heuristic
#     static=frozenset({'(available-color white)', '(available-color black)', '(up tile_1_1 tile_0_1)', '(down tile_0_1 tile_1_1)'}) # Add necessary static facts for parsing/distance
# )
#
# dummy_node = Node(state=dummy_task.initial_state, parent=None, action=None, cost=0)
#
# # Instantiate and call the heuristic
# heuristic = floortileHeuristic(dummy_task)
# h_value = heuristic(dummy_node)
# print(f"Heuristic value: {h_value}")

# Example 1 from problem description:
# Initial state: robot1 at tile_0_1, has black. tile_1_1 clear. Goal: painted tile_1_1 white.
# Goal (painted tile_1_1 white) is not in state.
# tile_1_1 is not painted with wrong color (it's clear).
# Robot1 at tile_0_1 (0,1), has black. Goal color white.
# Distance tile_0_1 to tile_1_1 = abs(0-1) + abs(1-1) = 1.
# Color needs change (black != white). Cost 1.
# Paint cost 1.
# Robot1 cost for tile_1_1 = 1 (move) + 1 (color) + 1 (paint) = 3.
# Min cost for tile_1_1 = 3.
# Total heuristic = 3.

# Example 2 (partial check): Goal (painted tile_1_2 black). Initial state has (painted tile_1_2 white).
# Goal (painted tile_1_2 black) is not in state.
# State has (painted tile_1_2 white). Goal color is black, state color is white. Wrong color!
# Heuristic should return infinity.

# Example state from problem description:
# state = frozenset({'(clear tile_1_5)', '(clear tile_3_1)', '(clear tile_3_3)', '(clear tile_3_4)', '(clear tile_0_1)', '(clear tile_0_2)', '(clear tile_0_3)', '(clear tile_3_2)', '(painted tile_1_2 black)', '(painted tile_1_3 white)', '(painted tile_1_4 black)', '(painted tile_3_5 white)', '(painted tile_1_1 white)', '(clear tile_2_1)', '(clear tile_2_2)', '(clear tile_2_3)', '(clear tile_2_4)', '(clear tile_2_5)', '(painted tile_3_6 black)', '(painted tile_2_6 white)', '(painted tile_1_6 black)', '(clear tile_0_6)', '(robot-has robot1 white)', '(robot-at robot1 tile_0_4)', '(clear tile_0_5)'})
# goals = frozenset({'(painted tile_1_1 white)', '(painted tile_1_2 black)', '(painted tile_1_3 white)', '(painted tile_1_4 black)', '(painted tile_1_5 white)', '(painted tile_1_6 black)', '(painted tile_2_1 black)', '(painted tile_2_2 white)', '(painted tile_2_3 black)', '(painted tile_2_4 white)', '(painted tile_2_5 black)', '(painted tile_2_6 white)', '(painted tile_3_1 white)', '(painted tile_3_2 black)', '(painted tile_3_3 white)', '(painted tile_3_4 black)', '(painted tile_3_5 white)', '(painted tile_3_6 black)', '(painted tile_4_1 black)', '(painted tile_4_2 white)', '(painted tile_4_3 black)', '(painted tile_4_4 white)', '(painted tile_4_5 black)', '(painted tile_4_6 white)', '(painted tile_5_1 white)', '(painted tile_5_2 black)', '(painted tile_5_3 white)', '(painted tile_5_4 black)', '(painted tile_5_5 white)', '(painted tile_5_6 black)'})
#
# # Goals already satisfied in this state:
# # (painted tile_1_2 black) - Yes
# # (painted tile_1_3 white) - Yes
# # (painted tile_1_4 black) - Yes
# # (painted tile_3_5 white) - Yes
# # (painted tile_1_1 white) - Yes
# # (painted tile_3_6 black) - Yes
# # (painted tile_2_6 white) - Yes
# # (painted tile_1_6 black) - Yes
#
# # Goals not satisfied: All others in the goals set.
# # Are any of the unsatisfied goals painted with the wrong color?
# # Check state for painted tiles that are NOT in the satisfied list above.
# # State painted tiles: (painted tile_1_2 black), (painted tile_1_3 white), (painted tile_1_4 black), (painted tile_3_5 white), (painted tile_1_1 white), (painted tile_3_6 black), (painted tile_2_6 white), (painted tile_1_6 black)
# # All painted tiles in the state are goal tiles and are painted with the correct color. No wrong colors.
#
# # Robot info: robot1 at tile_0_4 (0,4), has white.
#
# # Calculate cost for an unsatisfied goal, e.g., (painted tile_1_5 white)
# # Goal tile: tile_1_5 (1,5). Goal color: white.
# # Robot1 at tile_0_4 (0,4), has white.
# # Distance tile_0_4 to tile_1_5 = abs(0-1) + abs(4-5) = 1 + 1 = 2.
# # Robot color white == goal color white. Color change cost 0.
# # Paint cost 1.
# # Robot1 cost for tile_1_5 = 2 (move) + 0 (color) + 1 (paint) = 3.
# # Min cost for tile_1_5 = 3. Add 3 to total_cost.
#
# # Calculate cost for another unsatisfied goal, e.g., (painted tile_2_1 black)
# # Goal tile: tile_2_1 (2,1). Goal color: black.
# # Robot1 at tile_0_4 (0,4), has white.
# # Distance tile_0_4 to tile_2_1 = abs(0-2) + abs(4-1) = 2 + 3 = 5.
# # Robot color white != goal color black. Color change cost 1.
# # Paint cost 1.
# # Robot1 cost for tile_2_1 = 5 (move) + 1 (color) + 1 (paint) = 7.
# # Min cost for tile_2_1 = 7. Add 7 to total_cost.
#
# # The heuristic would sum these minimum costs for all unsatisfied goals.
