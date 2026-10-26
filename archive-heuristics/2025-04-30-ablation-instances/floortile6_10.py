from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

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

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_r_c' into (row, column) integers."""
    try:
        parts = tile_name.split('_')
        # Expecting ['tile', 'row_str', 'col_str']
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
        else:
            return None
    except ValueError:
        return None

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet correctly painted. It sums the estimated cost for each
    unpainted goal tile independently. The cost for a single tile includes
    movement to a paintable position (adjacent tile), changing color if needed,
    and the paint action itself.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates
      connecting tiles in a grid-like manner. Tile names follow the format 'tile_row_col'.
      Manhattan distance on row/column indices is used as a proxy for movement cost.
    - The robot must be at a tile adjacent to the target tile to paint it.
    - Tiles that need to be painted are currently 'clear'.
    - Tiles are only painted once correctly according to the goal.
    - Action costs are uniform (1).
    - The heuristic ignores the 'clear' predicate requirements for movement and painting
      targets, assuming free movement on the grid for distance calculation.

    # Heuristic Initialization
    - Extracts the required color for each goal tile from the task's goal conditions.
      Stores this mapping in `self.goal_colors`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Find the robot's current location (tile) and the color it is currently holding
       by examining the current state facts. Assume a single robot named 'robot1'.
    3. Parse the robot's tile name to get its (row, column) coordinates (rr, rc).
       If parsing fails or robot location is not found, return infinity.
    4. Iterate through each goal condition specifying a tile that needs to be painted
       with a specific color (tile_Y, color_C) from `self.goal_colors`.
    5. For each (tile_Y, color_C) pair:
       a. Check if the fact `(painted tile_Y color_C)` is already present in the current state.
       b. If the tile is *not* painted correctly in the current state:
          i. Parse the tile_Y name to get its (row, column) coordinates (ry, cy).
             If parsing fails, add a large penalty and skip this tile.
          ii. Calculate the Manhattan distance D between the robot's current location (rr, rc)
              and the target tile's location (ry, cy): D = abs(rr - ry) + abs(rc - cy).
          iii. Estimate the minimum number of move actions required to get the robot
               to a tile adjacent to tile_Y (from which it can paint tile_Y).
               - If D = 0 (robot is at tile_Y), it needs 1 move to get off tile_Y to an adjacent tile.
               - If D = 1 (robot is adjacent to tile_Y), it needs 0 moves (already in position).
               - If D > 1, it needs D - 1 moves to reach an adjacent tile.
               Let this be `moves_to_adjacent`.
          iv. Estimate the cost for changing color: 1 action if the robot's current color
              is not color_C (and robot_color is known), and 0 actions otherwise. Let this be `color_cost`.
          v. The paint action itself costs 1.
          vi. The estimated cost for this specific unpainted goal tile is `moves_to_adjacent + color_cost + 1`.
          vii. Add this estimated cost to the `total_cost`.
    6. Return the `total_cost`. This sum represents the estimated total effort
       needed for all unpainted goal tiles, ignoring potential synergies (like
       painting multiple tiles of the same color in one trip or using the same
       movement for multiple tiles) and ignoring the 'clear' predicate constraints.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        # Store goal locations and required colors for each tile.
        # Example: {'tile_1_2': 'black', 'tile_1_3': 'white', ...}
        self.goal_colors = {}
        for goal in task.goals:
            # Assuming goal facts are like "(painted tile_r_c color)"
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile_name, color_name = parts[1], parts[2]
                self.goal_colors[tile_name] = color_name

        # Static facts are not explicitly needed for this heuristic's calculation logic,
        # as grid structure is implicitly handled by Manhattan distance on tile names.
        # Available colors are also not needed explicitly as we only check the robot's current color.
        # task.static # Example of accessing static facts if needed

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # 1. Initialize total cost
        total_cost = 0

        # 2. Find robot's current location and color
        robot_tile = None
        robot_color = None
        # Assuming a single robot named 'robot1' based on domain file
        robot_name = 'robot1' # Could potentially extract this from objects if needed

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            if parts[0] == "robot-at" and len(parts) == 3 and parts[1] == robot_name:
                robot_tile = parts[2]
            elif parts[0] == "robot-has" and len(parts) == 3 and parts[1] == robot_name:
                robot_color = parts[2]

        # 3. Parse robot's coordinates
        if robot_tile is None:
             # Robot location not found, cannot proceed with distance calculation.
             # This indicates an invalid state based on domain predicates.
             # Return infinity to prune this branch.
             return float('inf')

        robot_coords = parse_tile_name(robot_tile)
        if robot_coords is None:
             # Cannot parse robot tile name, cannot compute distance.
             # Return infinity.
             return float('inf')

        rr, rc = robot_coords

        # 4. Iterate through goal tiles
        for goal_tile, goal_color in self.goal_colors.items():
            # 5a. Check if the tile is already painted correctly
            is_painted_correctly = f"(painted {goal_tile} {goal_color})" in state

            # 5b. If not painted correctly
            if not is_painted_correctly:
                # i. Parse goal tile coordinates
                goal_coords = parse_tile_name(goal_tile)
                if goal_coords is None:
                    # Cannot parse goal tile name. This goal is unhandleable.
                    # Add a large penalty to discourage states leading here.
                    total_cost += 1000 # Arbitrary large penalty
                    continue # Skip to next goal tile

                ry, cy = goal_coords

                # ii. Calculate Manhattan distance
                D = abs(rr - ry) + abs(rc - cy)

                # iii. Estimate moves to get to an adjacent tile
                # If robot is at the goal tile (D=0), it needs 1 move away.
                # If robot is adjacent (D=1), it needs 0 moves.
                # If robot is further (D>1), it needs D-1 moves.
                if D == 0:
                    moves_to_adjacent = 1
                else: # D >= 1
                    moves_to_adjacent = D - 1

                # iv. Estimate color change cost
                # Assuming robot_has fact is always present if robot_color is not None
                color_cost = 0
                # Check if robot_color was successfully found and is different from goal color
                if robot_color is not None and robot_color != goal_color:
                     color_cost = 1
                # Note: If robot_color is None (e.g., free-color state, though unused),
                # we might need a cost to acquire *any* color first.
                # Based on domain, robot always has a color.

                # v. Paint action cost is 1
                paint_cost = 1

                # vi. Estimated cost for this tile
                tile_cost = moves_to_adjacent + color_cost + paint_cost

                # vii. Add to total cost
                total_cost += tile_cost

        # 6. Return total cost
        # The heuristic is 0 iff all goal tiles are painted correctly,
        # because the loop only adds cost for unpainted goal tiles.
        return total_cost
