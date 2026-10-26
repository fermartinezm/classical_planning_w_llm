from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check is just element-wise matching up to the length of args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """
    Parse a tile name like 'tile_row_col' into (row, col) integers.
    Assumes tile names follow the format 'tile_R_C' where R and C are integers.
    """
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return int(parts[1]), int(parts[2])
        else:
            # Handle unexpected format - maybe log a warning or raise an error
            # For this heuristic, returning (0,0) or similar might be okay
            # but indicates a potential issue with input data format.
            # Let's return None to indicate failure.
            return None
    except (ValueError, IndexError):
        # Handle cases where row/col are not integers or parts are missing
        return None

def manhattan_distance(pos1, pos2):
    """
    Calculate the Manhattan distance between two grid positions (row, col).
    """
    if pos1 is None or pos2 is None:
        return float('inf') # Cannot calculate distance if parsing failed
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing
    the estimated costs for each unpainted goal tile. The cost for a single
    tile includes:
    1. The paint action itself.
    2. The cost to change a robot's color if the required color is not
       currently held by any robot.
    3. The minimum movement cost for any robot to get adjacent to the tile.

    # Assumptions
    - Tiles are named in the format 'tile_row_col'.
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - If a goal tile is painted with the wrong color, the problem is likely
      unsolvable in this domain (no unpaint action), so we return infinity.
    - The heuristic is non-admissible and designed for greedy best-first search.
      It sums costs per tile independently, ignoring potential synergies
      (e.g., one move serving multiple tiles, one color change serving multiple tiles).
      It also ignores the 'clear' precondition for movement, using simple
      Manhattan distance as an estimate.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which colors.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted ?tile ?color)`.
    2. Filter these goal facts to find the set of tiles that are *not*
       painted correctly in the current state. These are the "unpainted goal tiles".
    3. If any goal tile is painted with a *wrong* color in the current state,
       return infinity (or a very large number), as this state is likely
       a dead end.
    4. Initialize the heuristic value `h` to 0.
    5. Add the number of unpainted goal tiles to `h`. This accounts for the
       paint action needed for each tile.
    6. Determine the set of colors required by the unpainted goal tiles.
    7. Determine the set of colors currently held by the robots.
    8. Count the number of required colors that are not held by any robot.
       Add this count to `h`. This estimates the minimum number of color
       change actions needed across all robots.
    9. For each unpainted goal tile:
       a. Parse its name to get its (row, col) coordinates.
       b. Find the minimum Manhattan distance from *any* robot's current
          location to *any* tile adjacent to the unpainted goal tile.
          The distance from robot at (R_r, R_c) to get adjacent to tile (T_r, T_c)
          is estimated as `max(0, abs(R_r - T_r) + abs(R_c - T_c) - 1)`.
       c. Add this minimum distance to `h`. This estimates the movement cost
          needed for this specific tile, summed independently.
    10. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.

        # Pre-parse goal facts to easily access target color for each goal tile
        self.goal_painted_tiles = {} # Map tile_name -> goal_color
        for goal in self.goals:
             predicate, *args = get_parts(goal)
             if predicate == "painted":
                 if len(args) == 2:
                     tile, color = args
                     self.goal_painted_tiles[tile] = color
                 else:
                     print(f"Warning: Unexpected goal format: {goal}")


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state  # Current world state (frozenset of fact strings)
        task_static = node.task.static # Static facts (frozenset of fact strings)

        h = 0 # Initialize heuristic value

        unpainted_goal_tiles = {} # Map tile_name -> goal_color for tiles needing paint
        required_colors = set()   # Set of colors needed for unpainted goal tiles

        # Check goal tiles
        for tile, goal_color in self.goal_painted_tiles.items():
            painted_correctly = f"(painted {tile} {goal_color})" in state

            if not painted_correctly:
                # Check if painted with the wrong color - indicates likely unsolvable state
                is_painted_wrongly = False
                for fact in state:
                    if match(fact, "painted", tile, "*"):
                        _, painted_tile, painted_color = get_parts(fact)
                        if painted_tile == tile and painted_color != goal_color:
                            is_painted_wrongly = True
                            break # Found wrong color for this goal tile

                if is_painted_wrongly:
                    # If a goal tile is painted the wrong color, it cannot be fixed
                    # with the available actions (no unpaint). Return infinity.
                    return float('inf')

                # If not painted correctly and not painted wrongly, it needs painting
                unpainted_goal_tiles[tile] = goal_color
                required_colors.add(goal_color)

        # Component 1: Cost for paint actions (1 per unpainted goal tile)
        h += len(unpainted_goal_tiles)

        # If all goal tiles are painted correctly, the goal is reached.
        # The heuristic should be 0 in this case.
        if len(unpainted_goal_tiles) == 0:
             return 0

        # Component 2: Cost for color changes
        # Find colors currently held by robots
        robot_colors = set()
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                 _, robot, color = get_parts(fact)
                 robot_colors.add(color)

        # Count required colors that no robot currently has
        colors_to_acquire = required_colors - robot_colors
        h += len(colors_to_acquire) # Add 1 for each color type that needs to be acquired

        # Component 3: Estimate movement cost
        # Find robot locations
        robot_locations = {} # Map robot_name -> tile_name
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, location = get_parts(fact)
                robot_locations[robot] = location

        # Calculate minimum movement cost for each unpainted tile
        for tile, goal_color in unpainted_goal_tiles.items():
            tile_pos = parse_tile_name(tile)
            if tile_pos is None:
                 # Cannot calculate distance for this tile, skip or handle error
                 continue # Or return infinity if any unparseable tile makes it unsolvable

            min_dist_adj = float('inf')

            for robot, location in robot_locations.items():
                robot_pos = parse_tile_name(location)
                if robot_pos is None:
                    # Cannot calculate distance from this robot, skip or handle error
                    continue # Or return infinity

                # Distance from robot's current tile to the target tile
                dist_to_T = manhattan_distance(robot_pos, tile_pos)

                # Estimated moves to get to a tile adjacent to the target tile
                # If dist_to_T is 0, robot is on the tile, needs 1 move to adjacent.
                # If dist_to_T is 1, robot is adjacent, needs 0 moves to adjacent.
                # If dist_to_T > 1, needs dist_to_T - 1 moves to get adjacent.
                dist_to_adj = max(0, dist_to_T - 1)

                min_dist_adj = min(min_dist_adj, dist_to_adj)

            if min_dist_adj != float('inf'):
                 h += min_dist_adj
            # Note: If min_dist_adj is inf, it means there are no robots or
            # unparseable locations, which might indicate an unsolvable state,
            # but the earlier check for unpainted tiles already handles the goal=0 case.
            # For non-goal states, inf here means no robot can reach, likely unsolvable.
            # However, adding inf to h will result in inf, which is correct.

        return h

    # Include helper methods within the class as they use self.
    def get_parts(self, fact):
        """Extract the components of a PDDL fact."""
        return fact[1:-1].split()

    def match(self, fact, *args):
        """Check if a PDDL fact matches a given pattern."""
        parts = self.get_parts(fact)
        if len(parts) < len(args):
            return False
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def parse_tile_name(self, tile_name):
        """Parse 'tile_row_col' into (row, col) integers."""
        try:
            parts = tile_name.split('_')
            if len(parts) == 3 and parts[0] == 'tile':
                return int(parts[1]), int(parts[2])
            else:
                # print(f"Warning: Could not parse tile name {tile_name}") # Optional warning
                return None
        except (ValueError, IndexError):
            # print(f"Warning: Could not parse tile name {tile_name}") # Optional warning
            return None

