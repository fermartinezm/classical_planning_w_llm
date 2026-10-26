from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For float('inf')

# Utility functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_tile_coords(tile_name):
    """Parses tile name 'tile_r_c' to get (row, col) tuple."""
    parts = tile_name.split('_')
    if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
         # Should not happen in valid floortile problems
         raise ValueError(f"Unexpected tile name format: {tile_name}")
    return (int(parts[1]), int(parts[2]))

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles."""
    r1, c1 = get_tile_coords(tile1_name)
    r2, c2 = get_tile_coords(tile2_name)
    return abs(r1 - r2) + abs(c1 - c2)

def min_dist_to_adjacent(from_tile_name, target_tile_name):
    """
    Calculates minimum Manhattan distance (number of moves) from from_tile
    to any tile adjacent to target_tile, considering the robot cannot paint
    the tile it is currently on.
    """
    if from_tile_name == target_tile_name:
        # Robot is at the target tile, must move to an adjacent tile (1 move)
        return 1
    else:
        # Robot is not at the target tile. Minimum moves to get adjacent is max(0, dist - 1).
        dist = manhattan_distance(from_tile_name, target_tile_name)
        return max(0, dist - 1)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles with the correct colors. It considers the number of tiles
    that need painting, the movement cost for robots to reach those tiles,
    and the color changes required by robots.

    # Assumptions
    - Tiles form a grid structure where adjacency corresponds to Manhattan distance 1.
    - Tile names are in the format 'tile_row_col'.
    - Unpainted goal tiles are currently 'clear'. (Solvability assumption)
    - Movement is possible between adjacent clear tiles (heuristic ignores 'clear'
      precondition for movement cost estimation, using only grid distance).
    - Color changes are possible if the color is available (heuristic assumes
      needed colors are available, based on problem definition).

    # Heuristic Initialization
    - Extracts the set of goal conditions, specifically the required painted states.
    - Static facts (like adjacency) are implicitly used by the tile coordinate
      parsing and distance calculation functions, assuming a grid structure.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of three components:
    1.  **Paint Actions:** Count the number of goal facts `(painted T C)` that are
        not true in the current state. Each such tile needs one `paint` action.
        This is a lower bound on the number of paint actions.
    2.  **Movement Cost:** For each tile `T` that needs painting, calculate the
        minimum Manhattan distance from any robot's current location to *any*
        tile adjacent to `T`. Sum these minimum distances over all tiles that
        need painting. This component estimates the total movement effort.
        It likely overestimates as one movement might position a robot
        advantageously for multiple tiles.
    3.  **Color Change Cost:** For each robot, identify the set of colors
        required by the unpainted goal tiles that this robot is *closest* to.
        Count how many of these required colors are different from the color
        the robot currently holds. Sum this count over all robots. This component
        estimates the number of color changes needed. It likely overestimates
        as a robot might acquire a color and paint multiple tiles, or another
        robot might paint tiles needing that color.

    The total heuristic is the sum of these three components.
    h = (Number of unpainted goal tiles)
      + (Sum over unpainted goal tiles of min_dist(any_robot, adjacent_to_tile))
      + (Sum over robots of |{colors needed by closest tiles} - {robot's current color}|)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # Store goal conditions, specifically the required painted states.
        # We assume goals are conjunctions of (painted tile color) facts.
        self.goal_painted_facts = {
            fact for fact in task.goals if get_parts(fact)[0] == "painted"
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        unpainted_goal_tiles = set() # Stores (tile_name, color_name) tuples
        for goal_fact in self.goal_painted_facts:
            # Check if the goal fact is NOT in the current state
            if goal_fact not in state:
                # Extract tile and color from the goal fact '(painted tile color)'
                parts = get_parts(goal_fact)
                if len(parts) == 3: # Ensure it's a painted fact
                    unpainted_goal_tiles.add((parts[1], parts[2]))

        h = len(unpainted_goal_tiles)

        if h == 0:
            # Goal is reached
            return 0

        # Extract current robot locations and colors
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at" and len(parts) == 3:
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]

        # Component 2 & 3 preparation: Find closest robot(s) and min distance for each unpainted tile
        tile_min_dist_info = {} # {tile_name: (min_dist, [robot_name1, robot_name2, ...])}

        for target_tile, target_color in unpainted_goal_tiles:
            min_dist_for_tile = float('inf')
            closest_robots_list = []

            for robot_name, robot_loc in robot_locations.items():
                dist = min_dist_to_adjacent(robot_loc, target_tile)

                if dist < min_dist_for_tile:
                    min_dist_for_tile = dist
                    closest_robots_list = [robot_name]
                elif dist == min_dist_for_tile:
                    closest_robots_list.append(robot_name)

            if min_dist_for_tile != float('inf'):
                 tile_min_dist_info[target_tile] = (min_dist_for_tile, closest_robots_list)

        # Component 2: Movement cost
        total_movement_cost = sum(min_dist for min_dist, _ in tile_min_dist_info.values())
        h += total_movement_cost

        # Component 3: Color change cost
        robot_needed_colors = {robot_name: set() for robot_name in robot_locations}

        # For each unpainted tile, add its color requirement to the set of needed colors
        # for the robot(s) closest to it.
        for target_tile, target_color in unpainted_goal_tiles:
             if target_tile in tile_min_dist_info:
                 _, closest_robots_list = tile_min_dist_info[target_tile]
                 for robot_name in closest_robots_list:
                     robot_needed_colors[robot_name].add(target_color)

        total_color_change_cost = 0
        for robot_name, needed_colors_set in robot_needed_colors.items():
            current_color = robot_colors.get(robot_name)

            # Count the number of colors in `needed_colors_set` that are *not* `current_color`.
            # This estimates how many times this robot might need to change color
            # if it were to handle the tiles it's closest to.
            colors_different_from_current = needed_colors_set - {current_color}
            total_color_change_cost += len(colors_different_from_current)

        h += total_color_change_cost

        return h
