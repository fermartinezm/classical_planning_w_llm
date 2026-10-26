from fnmatch import fnmatch
# Assuming Heuristic base class is available in the specified path
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list or handle error for invalid fact format
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts
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

# Helper function to parse tile coordinates
def parse_tile_coords(tile_name):
    """
    Parses a tile name string like 'tile_R_C' into a tuple (R, C) of integers.
    Assumes tile names follow this format.
    Returns (0, 0) for unexpected formats, though valid instances should conform.
    """
    try:
        # Split by '_' and take the last two parts
        parts = tile_name.split('_')
        # Ensure there are at least 3 parts (e.g., 'tile', 'R', 'C')
        if len(parts) < 3:
             # In a real system, you might log a warning or raise an error here.
             # print(f"Warning: Tile name format incorrect for '{tile_name}'.")
             return (0, 0) # Default or error value
        row = int(parts[-2])
        col = int(parts[-1])
        return (row, col)
    except (ValueError, IndexError) as e:
        # Log a warning or handle the error appropriately in a real system
        # print(f"Warning: Could not parse tile name '{tile_name}': {e}. Assuming (0,0).")
        return (0, 0) # Default or error value

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet painted correctly. It calculates the minimum cost for each
    unpainted goal tile independently, considering the closest robot and the
    color change cost, and sums these minimum costs.

    # Assumptions
    - All tiles required in the goal are initially 'clear' and remain so until painted
      with the correct color.
    - Robots always have a color (the 'free-color' predicate is ignored as it's unused).
    - All colors required in the goal are available via the 'change_color' action.
    - The grid structure allows movement between adjacent tiles as defined by
      'up', 'down', 'left', 'right' predicates.
    - Pathfinding for movement assumes clear paths (ignores other robots or obstacles).
    - The cost of each action (move, paint, change_color) is 1.
    - Goal facts only specify 'painted' predicates.

    # Heuristic Initialization
    - Extracts the target color for each tile that needs to be painted in the goal state.
      This is stored in `self.goal_paintings` as a dictionary mapping tile names to colors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location and color held by each robot present in the state.
    2. Identify which goal tiles are not yet painted with their required color
       in the current state. These are the 'unpainted goal tiles'.
    3. Initialize the total heuristic value to 0.
    4. For each unpainted goal tile (let's say tile T needs color C):
       a. Calculate the minimum cost for *any* robot to paint tile T with color C.
       b. For each robot R found in the state:
          i. Determine the cost for robot R to obtain color C: 1 if R currently
             holds a different color, 0 otherwise.
          ii. Determine the number of moves required for robot R to reach a tile
              adjacent to tile T. This is calculated using Manhattan distance
              between the robot's current tile and the target tile T.
              - Let `dist` be the Manhattan distance between the robot's tile and tile T.
              - If `dist == 0` (robot is at tile T), it needs 1 move to get adjacent.
              - If `dist > 0` (robot is not at tile T), it needs `dist - 1` moves to get adjacent.
              This can be calculated as `dist - 1 if dist >= 1 else 1`.
          iii. The cost of the paint action itself is 1.
          iv. The total cost for robot R to paint tile T is (color cost) + (move cost) + (paint cost).
          v. Update the minimum cost for tile T with the minimum cost found so far across all robots.
       c. Add the minimum cost for tile T to the total heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal painting requirements.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are available in task.static but not explicitly used
        # in this heuristic's calculation, as grid structure and available colors
        # are implicitly handled by parsing tile names and assuming color changes are possible.

        # Store goal locations and required colors for each tile.
        # We assume goal facts are only of the form (painted tile color)
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color
            # If the goal contains other facts, they are ignored by this heuristic.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify current robot locations and colors
        robot_info = {} # Map robot name to {'location': tile_name, 'color': color_name}
        robots_in_state = set() # Keep track of robots found in state
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3:
                predicate, obj1, obj2 = parts
                if predicate == "robot-at":
                    robot, tile = obj1, obj2
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['location'] = tile
                    robots_in_state.add(robot)
                elif predicate == "robot-has":
                    robot, color = obj1, obj2
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['color'] = color
                    robots_in_state.add(robot)

        # Filter out robots whose info is incomplete (shouldn't happen in valid states)
        valid_robots = {r for r, info in robot_info.items() if 'location' in info and 'color' in info}

        # 2. Identify unpainted goal tiles
        unpainted_goals = [] # List of (tile, required_color) tuples
        for tile, required_color in self.goal_paintings.items():
            # Check if the tile is already painted with the correct color
            if f"(painted {tile} {required_color})" not in state:
                 # Add to unpainted list if not painted correctly
                 unpainted_goals.append((tile, required_color))

        # If all goal tiles are painted correctly, the heuristic is 0.
        if not unpainted_goals:
            return 0

        # If there are unpainted goals but no valid robots, the state is likely unsolvable
        # (unless a robot appears later, which is not typical in STRIPS).
        # Return infinity in this case.
        if not valid_robots:
             return float('inf')


        # 3. Calculate total heuristic
        total_h = 0

        # 4. For each unpainted goal tile, find the minimum cost across all valid robots
        for tile, required_color in unpainted_goals:
            min_cost_for_tile = float('inf')
            target_r, target_c = parse_tile_coords(tile)

            for robot in valid_robots:
                robot_loc = robot_info[robot]['location']
                robot_color = robot_info[robot]['color']
                robot_r, robot_c = parse_tile_coords(robot_loc)

                # Cost to get the correct color
                color_cost = 1 if robot_color != required_color else 0

                # Calculate Manhattan distance between robot and target tile
                dist = abs(robot_r - target_r) + abs(robot_c - target_c)

                # Calculate moves needed to reach a tile adjacent to the target tile
                # If robot is at target (dist=0), needs 1 move to adjacent.
                # If robot is adjacent (dist=1), needs 0 moves.
                # If robot is further (dist>1), needs dist-1 moves.
                move_cost = dist - 1 if dist >= 1 else 1 # Simplified: 1 if dist==0 else dist-1

                # Cost of the paint action
                paint_cost = 1

                # Total cost for this robot to paint this tile
                cost_for_robot = color_cost + move_cost + paint_cost

                # Update minimum cost for this tile
                min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)

            # Add the minimum cost for this tile to the total heuristic
            # min_cost_for_tile should not be infinity here because we checked valid_robots earlier
            total_h += min_cost_for_tile

        # 5. Return total heuristic value
        return total_h
