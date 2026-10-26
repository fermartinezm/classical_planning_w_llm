from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_tile_coords(tile_name):
    """Parses tile name 'tile_R_C' into (R, C) integer coordinates."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            return (int(parts[1]), int(parts[2]))
        except ValueError:
            # Handle cases where R or C are not integers if necessary
            # print(f"Warning: Could not parse integer coordinates from tile name {tile_name}")
            return None
    # print(f"Warning: Unexpected tile name format: {tile_name}")
    return None

def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two (R, C) coordinate tuples."""
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost based on:
    1. Number of tiles that need painting.
    2. Number of tiles that need clearing (robot on them).
    3. Number of colors that need to be acquired by any robot.
    4. Sum of minimum Manhattan distances from each robot to its closest unpainted goal tile.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing goal conditions."""
        self.goals = task.goals
        # Static facts are not explicitly used in this heuristic's calculation,
        # but the tile naming convention relies on the grid structure implied by static facts.
        # self.static = task.static # Not strictly needed for this heuristic logic

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Identify unsatisfied goal tiles and their target colors
        unsatisfied_goals = set()
        for goal in self.goals:
            # Goal is typically (painted tile_X_Y color)
            if match(goal, "painted", "*", "*"):
                tile, color = get_parts(goal)[1:]
                # Check if the goal predicate is NOT in the current state
                if goal not in state:
                     unsatisfied_goals.add((tile, color))

        # If all goals are satisfied, heuristic is 0
        if not unsatisfied_goals:
            return 0

        # 2. Check for unsolvable state (tile painted with wrong color)
        # Iterate through all tiles mentioned in unsatisfied goals
        for tile, target_color in unsatisfied_goals:
             # Check if this tile is painted with *any* color in the current state
             for fact in state:
                 if match(fact, "painted", tile, "*"):
                     painted_color = get_parts(fact)[2]
                     if painted_color != target_color:
                         # Tile is painted with the wrong color, likely unsolvable
                         return float('inf')
             # Note: If a tile is clear, it won't match "(painted tile *)", so this check is correct.

        h = 0

        # 3. Identify robot locations and colors
        robot_locs = {} # {robot_name: tile_name}
        robot_colors = {} # {robot_name: color_name}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, loc = get_parts(fact)[1:]
                robot_locs[robot] = loc
            elif match(fact, "robot-has", "*", "*"):
                robot, color = get_parts(fact)[1:]
                robot_colors[robot] = color
            # Ignore (free-color) predicate as it's not used in actions

        # 4. Identify needed colors and held colors
        needed_colors = {color for tile, color in unsatisfied_goals}
        held_colors = set(robot_colors.values())

        # 5. Calculate color acquisition cost
        # Cost is 1 for each needed color that no robot currently holds.
        colors_to_acquire = needed_colors - held_colors
        h += len(colors_to_acquire)

        # 6. Add cost for paint actions
        # Each unsatisfied goal requires one paint action.
        h += len(unsatisfied_goals)

        # 7. Add cost for clearing occupied tiles
        # If a tile needs painting but is occupied by a robot, that robot must move off (1 action).
        occupied_goal_tiles = {
            tile for tile, color in unsatisfied_goals
            if "(clear " + tile + ")" not in state # If not clear, it must be occupied by a robot
        }
        h += len(occupied_goal_tiles)

        # 8. Add movement cost
        # Estimate movement cost as the sum of minimum Manhattan distances
        # from each robot to its closest unsatisfied goal tile.
        unsatisfied_tile_coords = {get_tile_coords(tile) for tile, color in unsatisfied_goals}
        unsatisfied_tile_coords.discard(None) # Remove any tiles that couldn't be parsed

        if unsatisfied_tile_coords: # Only calculate movement if there are targets
            for robot, r_loc in robot_locs.items():
                robot_coords = get_tile_coords(r_loc)
                if robot_coords is not None:
                    min_dist_for_robot = float('inf')
                    for target_coords in unsatisfied_tile_coords:
                        dist = manhattan_distance(robot_coords, target_coords)
                        min_dist_for_robot = min(min_dist_for_robot, dist)

                    # Add the minimum distance for this robot to the total heuristic
                    if min_dist_for_robot != float('inf'):
                         h += min_dist_for_robot

        return h
