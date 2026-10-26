from fnmatch import fnmatch
# Assume Heuristic base class is available in heuristics.heuristic_base
# For standalone execution or testing, you might need a dummy Heuristic class:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(painted tile_1_1 white)" -> ["painted", "tile_1_1", "white"]
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """
    Parses a tile name like 'tile_row_col' into a (row, col) tuple of integers.
    Returns None if the format is unexpected.
    """
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col are not integers
            print(f"Warning: Could not parse tile coordinates from name {tile_name}")
            return None
    # Handle unexpected format
    print(f"Warning: Unexpected tile name format {tile_name}")
    return None

def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles given their names.
    Returns infinity if tile names cannot be parsed.
    """
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)

    if coords1 is None or coords2 is None:
        # Cannot calculate distance for invalid tile names
        return float('inf')

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to paint all goal tiles that are not
    yet painted correctly. For each such tile, it calculates the minimum cost
    among all robots to reach an adjacent tile, change color if needed, and paint.
    The total heuristic is the sum of these minimum costs for all unpainted goal tiles.

    # Assumptions:
    - Tiles are arranged in a grid, and names like 'tile_row_col' can be parsed
      to get coordinates for Manhattan distance calculation.
    - Movement cost between adjacent tiles is 1.
    - Changing color costs 1.
    - Painting costs 1.
    - The Manhattan distance is a reasonable estimate for movement cost, ignoring
      potential blockages from painted tiles.
    - Tiles that are goal tiles but painted with the wrong color are treated
      as needing to be painted (even though the domain actions don't allow repainting
      a non-clear tile; this might indicate an unsolvable state or a domain simplification).
      We assume valid instances don't require repainting wrongly painted goal tiles.
      Thus, we only count goal tiles that are *not* painted with the correct color.
    - Robots always have *a* color to change from if they need a different one.

    # Heuristic Initialization
    - Extracts the required color for each goal tile from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted according to the goal, but are
       not currently painted with the correct color in the state.
    2. For each such tile T needing color C:
       a. Determine the current location and color of each robot.
       b. For each robot R:
          i. Calculate the cost to change R's color to C (1 if needed, 0 otherwise).
          ii. Calculate the estimated movement cost for R to reach a tile adjacent to T.
              The robot needs to be *at* a tile X such that T is adjacent to X.
              The minimum number of moves from the robot's current location L_R
              to *any* tile X adjacent to T is estimated by the Manhattan distance
              between L_R and T.
          iii. Add the cost of the paint action (1).
          iv. The total cost for robot R to paint tile T is (color_cost + move_cost + paint_cost).
       c. Find the minimum cost among all robots to paint tile T.
    3. Sum the minimum costs calculated for each unpainted goal tile.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        # Store goal paintings: {tile_name: color_name}
        self.goal_paintings = {}
        for goal in task.goals:
            # Goal facts are typically like (painted tile_1_1 white)
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Static facts are not strictly needed for this heuristic's calculation
        # as tile coordinates are parsed from names, and adjacency is implicit
        # in the Manhattan distance on the grid implied by naming.
        # self.static = task.static

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to paint all goal tiles correctly.
        """
        state = node.state

        # 1. Identify current robot locations and colors, and current tile paintings
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}
        current_paintings = {} # {tile_name: color_name}
        # clear_tiles = set() # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == "painted":
                tile, color = parts[1], parts[2]
                current_paintings[tile] = color
            # elif parts[0] == "clear":
            #     tile = parts[1]
            #     clear_tiles.add(tile)

        total_heuristic_cost = 0

        # 2. Iterate through goal tiles and calculate cost for unpainted ones
        for goal_tile, goal_color in self.goal_paintings.items():
            # Check if the tile is already painted with the correct color
            if current_paintings.get(goal_tile) == goal_color:
                continue # This goal is already satisfied

            # This tile needs painting with goal_color
            min_cost_for_this_tile = float('inf')

            # Calculate the cost for each robot to paint this tile
            for robot_name, robot_location in robot_locations.items():
                current_robot_color = robot_colors.get(robot_name) # Get current color

                # Cost to change color if needed
                # The 'change_color' action requires having *some* color first.
                # We assume robots always have *a* color or the problem is structured
                # such that they can acquire one (e.g., from initial state).
                # The heuristic simplifies this to just checking if the *current*
                # color is the *goal* color. If not, a change is needed.
                color_change_cost = 0
                if current_robot_color != goal_color:
                     color_change_cost = 1 # Cost of change_color action

                # Estimated movement cost to get adjacent to the goal tile
                # The robot needs to move from robot_location to a tile X adjacent to goal_tile.
                # The minimum moves on a grid is the Manhattan distance between robot_location and goal_tile.
                move_cost = manhattan_distance(robot_location, goal_tile)

                # Cost of the paint action itself
                paint_cost = 1

                # Total cost for this robot to paint this specific tile
                robot_cost = color_change_cost + move_cost + paint_cost

                # Update minimum cost for this tile across all robots
                min_cost_for_this_tile = min(min_cost_for_this_tile, robot_cost)

            # Add the minimum cost required to paint this tile to the total heuristic
            # If min_cost_for_this_tile is still infinity, it means there are no robots,
            # which implies an unsolvable state. We can keep infinity or return a large number.
            # Assuming there's at least one robot, this won't be infinity unless tile names are unparseable.
            if min_cost_for_this_tile == float('inf'):
                 # This case might occur if tile names are malformed or no robots exist.
                 # For valid problems, this shouldn't happen. Return inf to indicate unsolvable.
                 return float('inf')
            else:
                 total_heuristic_cost += min_cost_for_this_tile

        return total_heuristic_cost

