# Assume this import is available in the environment where the heuristic is run
# from heuristics.heuristic_base import Heuristic

# Helper functions
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 city1-1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_coords(tile_name):
    """Parses 'tile_r_c' string into (row, col) tuple."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            return int(parts[1]), int(parts[2])
        except ValueError:
            return None # Invalid format
    return None # Invalid format


def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two (row, col) tuples."""
    r1, c1 = coords1
    r2, c2 = coords2
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance
    return abs(r1 - r2) + abs(c1 - c2)


# Assume Heuristic base class is defined elsewhere and imported
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         pass


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost by summing, for each unpainted goal tile,
    the cost of painting it (1) plus the minimum cost for any robot
    to get the correct color and reach the goal tile's location.
    Movement cost is estimated using Manhattan distance to the tile itself.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and robot names.
        """
        self.goals = task.goals  # Goal conditions.
        self.initial_state = task.initial_state # Use initial state to find robots

        # Extract goal tiles and their target colors
        self.goal_tiles = {} # Map tile name to target color
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

        # Extract robot names from initial state
        self.robots = set()
        for fact in self.initial_state:
            predicate, *args = get_parts(fact)
            if predicate == "robot-at":
                robot_name = args[0]
                self.robots.add(robot_name)
            elif predicate == "robot-has":
                 robot_name = args[0]
                 self.robots.add(robot_name)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Get current robot locations and colors
        robot_state = {} # Map robot name to {'location': tile_name, 'color': color_name}
        for robot in self.robots:
             robot_state[robot] = {'location': None, 'color': None} # Initialize

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "robot-at":
                robot_name, location = args
                if robot_name in robot_state:
                    robot_state[robot_name]['location'] = location
            elif predicate == "robot-has":
                robot_name, color = args
                if robot_name in robot_state:
                    robot_state[robot_name]['color'] = color

        total_cost = 0  # Initialize action cost counter.

        # Iterate through all goal tiles
        for goal_tile, target_color in self.goal_tiles.items():
            # Check if the tile is already painted correctly in the current state
            is_painted_correctly = False
            for fact in state:
                predicate, *args = get_parts(fact)
                if predicate == "painted" and args[0] == goal_tile:
                    if args[1] == target_color:
                        is_painted_correctly = True
                        break # Found correct painting, move to next goal tile
                    # Note: We don't explicitly handle wrong colors, assuming solvable problems
                    # mean tiles needing painting are clear or not yet painted correctly.

            # If the tile is not painted correctly, calculate its contribution
            if not is_painted_correctly:
                # Cost for the paint action itself
                tile_paint_cost = 1

                min_robot_prep_cost = float('inf')

                # Calculate the cost for each robot to prepare to paint this tile
                goal_tile_coords = parse_tile_coords(goal_tile)
                if goal_tile_coords is None:
                     # Cannot parse goal tile coordinates, return inf
                     return float('inf')

                for robot_name, state_info in robot_state.items():
                    robot_location = state_info['location']
                    robot_color = state_info['color']

                    if robot_location is None or robot_color is None:
                         # This robot's state is incomplete, skip it
                         continue

                    robot_coords = parse_tile_coords(robot_location)
                    if robot_coords is None:
                         # This robot's location is unparseable, skip it
                         continue

                    # Calculate movement cost from robot's current location to the goal tile's location
                    # Using distance to the tile itself as a proxy for proximity cost
                    movement_cost = manhattan_distance(robot_coords, goal_tile_coords)

                    # Calculate color change cost
                    color_change_cost = 0
                    if robot_color != target_color:
                        color_change_cost = 1 # Need to change color

                    # Total cost for this robot to get ready to paint this tile
                    # (movement to tile location + color change)
                    robot_prep_cost = color_change_cost + movement_cost

                    min_robot_prep_cost = min(min_robot_prep_cost, robot_prep_cost)

                # Add the cost for this unpainted tile to the total heuristic
                if min_robot_prep_cost == float('inf'):
                     # This means no valid robot state was found to paint this tile.
                     # In a valid problem with robots, this shouldn't happen unless
                     # all robots are in states with unparseable locations/colors.
                     # Return inf to indicate this state is likely problematic/unsolvable.
                     return float('inf')

                total_cost += tile_paint_cost + min_robot_prep_cost

        return total_cost
