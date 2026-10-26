# Assuming heuristics.heuristic_base provides the Heuristic base class
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_str):
    """Parses a tile name string like 'tile_r_c' into a (row, col) tuple."""
    # Assumes tile names are in the format 'tile_row_col'
    parts = tile_str.split('_')
    if len(parts) != 3 or parts[0] != 'tile':
        # Handle unexpected format, maybe return None or raise error
        # For this domain, assuming the format is consistent
        raise ValueError(f"Unexpected tile name format: {tile_str}")
    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except ValueError:
        # Handle cases where row/col parts are not integers
        raise ValueError(f"Unexpected tile name format: {tile_str}")


def manhattan_distance(tile1_str, tile2_str):
    """Calculates the Manhattan distance between two tiles."""
    try:
        r1, c1 = parse_tile_name(tile1_str)
        r2, c2 = parse_tile_name(tile2_str)
        return abs(r1 - r2) + abs(c1 - c2)
    except ValueError:
        # If tile names cannot be parsed, distance is undefined/infinite
        return float('inf')


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost based on the number of unpainted goal tiles,
    the number of color changes needed, and the estimated movement cost
    (sum of Manhattan distances from the robot to the nearest paint locations).
    Considers multiple robots by taking the minimum heuristic value among them.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Builds the paint_locations map and stores goal tiles.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations and colors for each tile
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

        # Build the map from a tile to the set of tiles the robot can paint it from
        # e.g., to paint tile_1_1 (below tile_0_1), robot must be at tile_0_1
        self.paint_locations = {}
        all_tiles = set()

        # Find all tiles involved in grid relations and goals
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                all_tiles.add(parts[1]) # tile_y
                all_tiles.add(parts[2]) # tile_x

        all_tiles.update(self.goal_tiles.keys())

        # Initialize paint_locations for all relevant tiles
        for tile in all_tiles:
            self.paint_locations[tile] = set()

        # Populate paint_locations based on which tile the robot must be AT
        # to paint an adjacent tile.
        # Action paint_up ?r ?y ?x ?c requires (up ?y ?x) and (robot-at ?r ?x)
        # So, to paint ?y, robot must be at ?x where (up ?y ?x) is true.
        # This means ?x is a paint location for ?y.
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "up": # (up tile_y tile_x) -> robot at tile_x paints tile_y
                tile_y, tile_x = parts[1], parts[2]
                self.paint_locations[tile_y].add(tile_x)
            elif parts[0] == "down": # (down tile_y tile_x) -> robot at tile_x paints tile_y
                tile_y, tile_x = parts[1], parts[2]
                self.paint_locations[tile_y].add(tile_x)
            elif parts[0] == "left": # (left tile_y tile_x) -> robot at tile_x paints tile_y
                tile_y, tile_x = parts[1], parts[2]
                self.paint_locations[tile_y].add(tile_x)
            elif parts[0] == "right": # (right tile_y tile_x) -> robot at tile_x paints tile_y
                tile_y, tile_x = parts[1], parts[2]
                self.paint_locations[tile_y].add(tile_x)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Build sets of clear and painted tiles from the current state
        clear_tiles = set()
        painted_tiles = {} # {tile: color}
        robots_state = {} # {robot_name: {'loc': tile_name, 'color': color_name}}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "clear":
                clear_tiles.add(parts[1])
            elif parts[0] == "painted":
                painted_tiles[parts[1]] = parts[2]
            elif parts[0] == "robot-at":
                robot_name, robot_loc = parts[1], parts[2]
                if robot_name not in robots_state:
                    robots_state[robot_name] = {}
                robots_state[robot_name]['loc'] = robot_loc
            elif parts[0] == "robot-has":
                robot_name, robot_color = parts[1], parts[2]
                if robot_name not in robots_state:
                    robots_state[robot_name] = {}
                robots_state[robot_name]['color'] = robot_color

        if not robots_state: # Should not happen in a valid state
            return float('inf')

        # Identify unsatisfied goal tiles and check for unsolvability
        pending_goals = {} # {tile: color}
        for tile, goal_color in self.goal_tiles.items():
            current_color = painted_tiles.get(tile)

            if current_color is not None:
                # Tile is painted
                if current_color != goal_color:
                    # Painted with the wrong color - unsolvable
                    return float('inf')
                # Else: Painted correctly - goal satisfied for this tile
            else:
                # Tile is not painted. It must be clear to be a pending goal.
                if tile in clear_tiles:
                    pending_goals[tile] = goal_color
                else:
                    # Tile is not painted, not clear, and is a goal tile.
                    # This state is inconsistent with domain rules (a tile must be clear or painted).
                    # Assuming valid states, this shouldn't happen. If it does, unsolvable.
                    return float('inf')


        # If all goal tiles are painted correctly, heuristic is 0
        if not pending_goals:
            return 0

        # Calculate heuristic for each robot assuming it does the whole job
        # The state heuristic is the minimum of these per-robot heuristics
        min_robot_h = float('inf')

        for robot_name, robot_info in robots_state.items():
            robot_loc = robot_info.get('loc')
            robot_color = robot_info.get('color')

            if robot_loc is None or robot_color is None:
                 # Robot state is incomplete, should not happen in valid states
                 continue

            h_robot = 0

            # 1. Cost for paint actions: One action per pending tile
            h_robot += len(pending_goals)

            # 2. Cost for color changes
            required_colors = set(pending_goals.values())
            num_color_changes = 0
            if required_colors:
                if robot_color not in required_colors:
                    # Need to get the first required color
                    num_color_changes += 1
                    # Then potentially switch between other required colors
                    if len(required_colors) > 1:
                        num_color_changes += len(required_colors) - 1
                else:
                    # Already have one of the required colors, need to switch between others
                    if len(required_colors) > 1:
                        num_color_changes += len(required_colors) - 1
            h_robot += num_color_changes

            # 3. Estimated movement cost
            # Sum of Manhattan distances from robot's current location
            # to the nearest paint location for each pending tile.
            movement_cost = 0
            robot_h_is_inf = False # Flag to break outer loop for this robot

            for tile, color in pending_goals.items():
                min_dist_to_paint_loc = float('inf')

                # Check if the tile has defined paint locations
                if tile not in self.paint_locations or not self.paint_locations[tile]:
                     # This pending tile cannot be painted from anywhere. Unsolvable.
                     robot_h_is_inf = True
                     break # Exit inner loop for this robot

                for paint_loc in self.paint_locations[tile]:
                    # Use Manhattan distance, ignoring 'clear' constraint for simplicity/efficiency
                    try:
                        dist = manhattan_distance(robot_loc, paint_loc)
                        min_dist_to_paint_loc = min(min_dist_to_paint_loc, dist)
                    except ValueError:
                        # Handle case where robot_loc or paint_loc name is malformed
                        robot_h_is_inf = True
                        break # Exit inner loop for this robot

                if robot_h_is_inf: # Check if error occurred in inner loop
                    break # Exit outer loop for this robot

                # If after checking all paint locations, min_dist is still inf,
                # it implies an issue (e.g., robot_loc malformed or no valid paint_loc found).
                if min_dist_to_paint_loc == float('inf'):
                     robot_h_is_inf = True
                     break # Exit outer loop for this robot

                movement_cost += min_dist_to_paint_loc

            if robot_h_is_inf:
                 h_robot = float('inf')
            else:
                 h_robot += movement_cost

            min_robot_h = min(min_robot_h, h_robot)

        # If min_robot_h is still inf, it means either no robots were found
        # or all robots were in an unsolvable state (e.g., pending tile has no paint locs,
        # or robot_loc/paint_loc names are malformed, or a goal tile was not clear).
        # If pending_goals was not empty, but min_robot_h is inf, it's unsolvable.
        # If pending_goals was empty, we returned 0 earlier.
        # So if we reach here and min_robot_h is inf, it's an unsolvable state.
        return min_robot_h if min_robot_h != float('inf') else float('inf')
