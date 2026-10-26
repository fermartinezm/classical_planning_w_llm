# Need to import the base Heuristic class
# Assuming the structure is heuristics/heuristic_base.py
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: '(predicate arg1 arg2)' -> ['predicate', 'arg1', 'arg2']
    return fact[1:-1].split()

def get_coords(tile_name):
    """Parses tile name 'tile_r_c' into (row, col) tuple."""
    # Example: 'tile_1_1' -> ('1', '1') -> (1, 1)
    parts = tile_name.split('_')
    # Basic validation for expected format
    if len(parts) != 3 or parts[0] != 'tile':
        # This should not happen with valid problem instances for this domain
        # Raising an error might be better to catch unexpected inputs during development/testing.
        raise ValueError(f"Unexpected tile name format: {tile_name}")
    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except ValueError:
        # This should not happen with valid problem instances for this domain
        raise ValueError(f"Could not parse row/col from tile name: {tile_name}")


def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two (row, col) tuples."""
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet correctly painted. It sums the estimated cost for each unpainted
    goal tile independently, considering the closest robot and the need to change color.
    It is designed for greedy best-first search and is not admissible.

    # Assumptions
    - Tiles are named in the format 'tile_row_col', allowing coordinate extraction.
    - The grid is connected according to the up/down/left/right predicates, and movement cost is approximated by Manhattan distance on the (row, col) coordinates. This ignores dynamic obstacles (`clear` tiles).
    - If a tile is required by the goal to be painted with color C, and it is not currently painted with C, it must be 'clear' in the current state. Wrongly painted tiles (painted with C' != C when C is required) are assumed not to occur in solvable instances for tiles that are part of the goal and not yet correctly painted, as there's no unpaint action.
    - The cost of changing color is 1.
    - The cost of moving between adjacent tiles is 1.
    - The cost of painting a tile is 1.
    - Multiple robots can work in parallel, but the heuristic sums costs per tile, taking the minimum robot distance for each tile. This is a simplification.
    - Color change cost is added once per needed color if no robot has it, regardless of which robot does the change or how many tiles need that color. This is another simplification.

    # Heuristic Initialization
    - Extract the goal conditions (`self.goals`).
    - Identify all robot names from the initial state (`self.robots`).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the set of goal facts `(painted T C)` that are present in `task.goals` but not in the current `state`. Let this set be `unsatisfied_goals`.
    2. If `unsatisfied_goals` is empty, the current state is a goal state, so the heuristic value is 0.
    3. Determine the current location (tile) of each robot and the color each robot is holding by examining the facts in the current `state`. Store these in dictionaries `robot_locations` and `robot_colors`.
    4. Initialize the total heuristic value `total_cost = 0`.
    5. Create an empty set `colors_needed_for_painting` to keep track of all colors required by the tiles in `unsatisfied_goals`.
    6. Iterate through each goal fact `(painted T C)` in the `unsatisfied_goals` set:
       a. Parse the target tile `T` and its required color `C` from the goal fact.
       b. Add color `C` to the `colors_needed_for_painting` set.
       c. Parse the coordinates `(T_r, T_c)` from the target tile name `T` using the `get_coords` helper function.
       d. Find the minimum number of move actions required for *any* robot to reach a tile that is adjacent to `T`.
          - Initialize `min_moves_to_adjacent` to infinity.
          - For each robot `R` identified during initialization:
            # Ensure robot location is known in this state. If a robot somehow doesn't have a robot-at fact, skip it.
            if robot not in robot_locations:
                continue

            robot_tile = robot_locations[robot]
            try:
                robot_coords = get_coords(robot_tile)
            except ValueError as e:
                # If robot's tile name is malformed, this state is problematic.
                print(f"Error parsing robot tile coordinates for robot {robot}: {e}")
                return float('inf') # Indicate problematic state

            # Calculate Manhattan distance between robot and target tile
            d = manhattan_distance(robot_coords, target_coords)

            # Calculate moves needed for THIS robot to get adjacent to T
            moves = 0
            if d == 0: # Robot is at the target tile
                moves = 1 # Needs 1 move to get to an adjacent tile
            elif d > 1: # Robot is further than adjacent (d >= 2)
                moves = d - 1 # Needs d-1 moves to reach an adjacent tile
            # If d == 1, robot is already adjacent, moves = 0

            # Update minimum moves across all robots for this target tile
            min_moves_to_adjacent = min(min_moves_to_adjacent, moves)

       # If min_moves_to_adjacent is still infinity, it means no robots were found or processed.
       # This state is likely unsolvable or malformed.
       if min_moves_to_adjacent == float('inf'):
           return float('inf')

       # Add the minimum moves required for the closest robot to get adjacent
       total_cost += min_moves_to_adjacent

       # e. Add 1 for the paint action itself
       total_cost += 1

    # 7. After processing all unsatisfied goals, consider color changes
    # 8. For each color C that is needed for at least one unpainted tile
    for color in colors_needed_for_painting:
        # a. Check if any robot currently holds color C
        has_color = False
        for robot in self.robots:
            if robot_colors.get(robot) == color:
                has_color = True
                break # Found a robot with this color

        # b. If no robot has color C, add 1 for the change_color action
        # This assumes one change action is sufficient to make the color available
        # to a robot that needs it, which is a simplification.
        if not has_color:
            total_cost += 1

    # 9. Return the total heuristic value
    return total_cost
