from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_coords(tile_name):
    """Parses a tile name like 'tile_row_col' into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            pass # Not a valid tile name format
    return None

def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two (row, col) coordinates."""
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the estimated minimum cost
    for each tile that needs to be painted according to the goal and is currently clear.
    The estimated cost for a single tile is the minimum cost for any robot to get the required color,
    move to an adjacent tile, and paint the target tile. If a goal tile is painted with the wrong
    color, the heuristic returns infinity, indicating a likely dead end.

    # Assumptions
    - The goal only requires painting tiles that are initially clear or become clear (although no action clears a painted tile).
      Therefore, we primarily focus on goal tiles that are currently in a 'clear' state.
    - If a goal tile is painted with the wrong color, the state is considered a dead end (heuristic returns infinity).
    - Tile names follow the format 'tile_row_col' where row and col are integers.
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates corresponds to Manhattan distance.

    # Heuristic Initialization
    - Extract goal conditions (`(painted ?t ?c)`).
    - Parse static facts and initial state to identify all tile objects and their names.
    - Build a mapping from tile names to (row, col) coordinates by parsing tile names.
    - Build an adjacency list/set for the tile grid based on `up`, `down`, `left`, `right` predicates from static facts.
    - Identify all robot objects.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize total heuristic cost to 0.
    2. Identify all goal facts of the form `(painted ?t ?c)`.
    3. For each goal fact `(painted t c)`:
        a. Check if `(painted t c)` is already true in the current state. If yes, this goal is satisfied for this tile; continue to the next goal fact.
        b. Check if `(clear t)` is true in the current state.
        c. If `(clear t)` is false:
            - This means the tile `t` is painted with *some* color or is otherwise not clear.
            - Check if `(painted t c')` is true in the state for *any* color `c'`.
            - If `(painted t c')` is found for any `c'`, and `c'` is *not* the goal color `c`, then the tile is painted incorrectly. Based on the domain, this is a dead end. Return `math.inf`.
            - If `(clear t)` is false, but no `(painted t c')` is found (or it's painted with the correct color, which was handled in 3a), this state is inconsistent or a dead end. For simplicity and robustness, if a goal tile is not clear and not painted with the correct color, we treat it as a dead end. Return `math.inf`.
        d. If `(clear t)` IS true (and `(painted t c)` is false):
            - This tile `t` needs to be painted with color `c`.
            - Get the coordinates of tile `t` using the pre-computed `self.tile_coords`.
            - Find all tiles `x` adjacent to `t` using the pre-computed `self.tile_adj`.
            - Find the coordinates of these adjacent tiles.
            - Find all robots `r` and their current locations `loc_r` from the state by checking facts like `(robot-at r loc_r)`.
            - Initialize `min_cost_for_tile = math.inf`.
            - For each robot `r`:
                - Get the robot's current location `loc_r` from the state. If the robot's location is not found (shouldn't happen in valid states), skip this robot or return infinity.
                - Get the coordinates of `loc_r` using `self.tile_coords`.
                - Calculate `color_cost = 1` if `(robot-has r c)` is NOT in the state for robot `r`, else `0`.
                - Calculate `min_dist_to_adjacent = math.inf`.
                - For each adjacent tile `x_name` of `t` (found in `self.tile_adj.get(t, set())`):
                    # Ensure the adjacent tile name is valid and has coordinates
                    if x_name in self.tile_coords:
                        adj_coord = self.tile_coords[x_name]
                        dist = manhattan_distance(robot_coords, adj_coord)
                        min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

                # If there's at least one reachable adjacent tile
                if min_dist_to_adjacent != math.inf:
                    # Cost for this robot: move to adjacent + change color (if needed) + paint
                    cost_for_this_robot = min_dist_to_adjacent + color_cost + 1
                    min_cost_for_tile = min(min_cost_for_tile, cost_for_this_robot)

            # If no robot could potentially paint this tile (e.g., all robots at invalid locations or no adjacent tiles found)
            if min_cost_for_tile == math.inf:
                 # This tile needs painting but is unreachable by any robot from an adjacent tile.
                 # This implies a dead end or an unsolvable subproblem.
                 return math.inf # Dead end

            total_cost += min_cost_for_tile

    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        self.tile_coords = {}
        self.tile_adj = {}
        self.robots = set() # Store robot names
        self.colors = set() # Store color names

        # Collect all potential object names from initial state, goals, and static facts
        all_facts = set(initial_state) | set(self.goals) | set(static_facts)
        for fact in all_facts:
            parts = get_parts(fact)
            for part in parts:
                if part.startswith('tile_'):
                    coords = parse_tile_coords(part)
                    if coords is not None:
                        self.tile_coords[part] = coords
                        if part not in self.tile_adj:
                             self.tile_adj[part] = set()
                elif part.startswith('robot'): # Assuming robot names start with 'robot'
                     self.robots.add(part)
                # Assuming colors are arguments to (robot-has) or (available-color)
                elif match(fact, "robot-has", "*", part) or match(fact, "available-color", part):
                     self.colors.add(part)


        # Build adjacency list from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                # Predicates are (direction tile_to tile_from)
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # So, tile_1_1 is adjacent to tile_0_1
                dir_pred, tile1, tile2 = parts
                if tile1 in self.tile_adj and tile2 in self.tile_adj:
                     self.tile_adj[tile1].add(tile2)
                     self.tile_adj[tile2].add(tile1)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        total_cost = 0  # Initialize action cost counter.

        # Find current robot locations and colors
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}
        painted_tiles = {}   # {tile_name: color_name}

        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot_name, tile_name = get_parts(fact)
                robot_locations[robot_name] = tile_name
            elif match(fact, "robot-has", "*", "*"):
                 _, robot_name, color_name = get_parts(fact)
                 robot_colors[robot_name] = color_name
            elif match(fact, "painted", "*", "*"):
                 _, tile_name, color_name = get_parts(fact)
                 painted_tiles[tile_name] = color_name


        # Identify goal tiles that are not yet painted correctly
        unsatisfied_painted_goals = {} # {tile_name: color_name}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile_name, color_name = get_parts(goal)
                # Check if the goal is already satisfied
                if goal not in state:
                    unsatisfied_painted_goals[tile_name] = color_name

        # Calculate cost for each unsatisfied goal tile
        for tile_name, goal_color in unsatisfied_painted_goals.items():
            # Check the current state of the tile
            is_clear = f'(clear {tile_name})' in state
            current_painted_color = painted_tiles.get(tile_name)

            # If painted wrong or not clear/correctly painted, it's a dead end in this domain
            if current_painted_color is not None and current_painted_color != goal_color:
                 # Tile is painted with the wrong color
                 return math.inf # Dead end
            elif not is_clear and current_painted_color is None:
                 # Tile is not clear and not painted (inconsistent state or dead end)
                 return math.inf # Dead end
            # Note: If is_clear is false and current_painted_color == goal_color,
            # the goal would have been satisfied and not in unsatisfied_painted_goals.

            # If the tile is clear and needs painting
            if is_clear:
                min_cost_for_tile = math.inf

                # Find adjacent tiles and their coordinates
                adjacent_tiles = self.tile_adj.get(tile_name, set())
                adjacent_coords = []
                for adj_t_name in adjacent_tiles:
                    if adj_t_name in self.tile_coords:
                        adjacent_coords.append(self.tile_coords[adj_t_name])

                if not adjacent_coords:
                    # Tile needs painting but has no valid adjacent tiles with coordinates
                    return math.inf # Dead end

                # Calculate min cost for any robot to paint this tile
                for robot_name in self.robots:
                    if robot_name not in robot_locations:
                        # Robot location unknown - inconsistent state?
                        continue # Skip this robot

                    robot_loc_name = robot_locations[robot_name]
                    if robot_loc_name not in self.tile_coords:
                         # Robot location is not a valid tile? Inconsistent state.
                         continue # Skip this robot

                    robot_coords = self.tile_coords[robot_loc_name]

                    # Cost to get the correct color
                    # Assumes any available color can be obtained anywhere with 1 action if not held
                    color_cost = 1 if robot_colors.get(robot_name) != goal_color else 0

                    # Minimum distance from robot to any adjacent tile
                    min_dist_to_adjacent = math.inf
                    for adj_coord in adjacent_coords:
                        dist = manhattan_distance(robot_coords, adj_coord)
                        min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

                    if min_dist_to_adjacent != math.inf:
                        # Cost for this robot: move to adjacent + change color (if needed) + paint
                        cost_for_this_robot = min_dist_to_adjacent + color_cost + 1
                        min_cost_for_tile = min(min_cost_for_tile, cost_for_this_robot)

                # If no robot could potentially paint this tile (e.g., all robots at invalid locations or no adjacent tiles found)
                if min_cost_for_tile == math.inf:
                     # This tile needs painting but is unreachable by any robot from an adjacent tile.
                     # This implies a dead end or an unsolvable subproblem.
                     return math.inf # Dead end

                total_cost += min_cost_for_tile

        return total_cost
