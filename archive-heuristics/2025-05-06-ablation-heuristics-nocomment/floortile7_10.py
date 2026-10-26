from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For infinity

def get_parts(fact):
    """Helper to parse a PDDL fact string into a list of parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_X_Y' into (X, Y) coordinates."""
    parts = tile_name.split('_')
    # Assuming format is always tile_row_col
    row = int(parts[1])
    col = int(parts[2])
    return (row, col)

def manhattan_dist(coord1, coord2):
    """Calculates Manhattan distance between two (row, col) coordinates."""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        Estimates the cost to reach the goal by summing up the estimated cost
        for each goal tile that is not yet painted correctly. The estimated cost
        for a single unpainted goal tile is 1 (for the paint action) plus the
        minimum cost for any robot to get into a position to paint it. The cost
        for a robot to paint a tile includes the Manhattan distance from the
        robot's current location to the closest tile adjacent to the target tile,
        plus 1 if the robot needs to change color to match the target tile's
        required color. This heuristic is not admissible as it calculates costs
        for each unpainted tile independently, potentially double-counting robot
        movement or color changes that contribute to multiple goals.

    Assumptions:
        - The tile names follow the format 'tile_X_Y' where X and Y are integers
          representing row and column.
        - The grid is connected based on 'up', 'down', 'left', 'right' predicates.
        - Manhattan distance on the grid is a reasonable estimate for movement cost,
          ignoring the 'clear' precondition for intermediate tiles.
        - Changing color is always possible if the target color is available.
        - There is at least one robot and at least one available color in solvable problems.

    Heuristic Initialization:
        - Stores the goal facts.
        - Parses static facts to identify available colors.
        - Parses static facts defining 'up', 'down', 'left', 'right' relations
          to build a map of tile neighbors and a map of tile names to (row, col)
          coordinates.
        - Stores the required color for each goal tile.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the heuristic value `h` to 0.
        2. Identify the set of goal facts `(painted T C)` that are not true in the current state. These are the unpainted goal tiles.
        3. If there are no unpainted goal tiles, the state is a goal state, return `h = 0`.
        4. Identify the current location and color of each robot from the state facts. Map robot locations to (row, col) coordinates using the precomputed map.
        5. For each unpainted goal tile `(T, C)`:
            a. Add 1 to `h` (representing the cost of the paint action for this tile).
            b. Get the (row, col) coordinates for tile `T`.
            c. Find the set of tile names that are neighbors of `T` using the precomputed neighbor map.
            d. Initialize `min_total_cost_for_tile` to infinity. This will track the minimum cost for *any* robot to get into position and have the correct color to paint tile `T`.
            e. For each robot `R` at coordinates `(Rx, Ry)` with color `R_color`:
                i. Initialize `min_dist_R_to_neighbors` to infinity. This will track the minimum Manhattan distance from robot `R` to any neighbor of tile `T`.
                ii. For each neighbor tile `N` of `T`:
                    - Get the (row, col) coordinates for neighbor `N`.
                    - Calculate the Manhattan distance between `(Rx, Ry)` and `N`'s coordinates.
                    - Update `min_dist_R_to_neighbors` with the minimum distance found so far.
                iii. Calculate the estimated cost for robot `R` to paint tile `T`: `cost_for_robot = min_dist_R_to_neighbors` (movement cost).
                iv. If `R_color` is not equal to the required color `C`:
                    - Add 1 to `cost_for_robot` (cost for changing color).
                v. Update `min_total_cost_for_tile` with the minimum of its current value and `cost_for_robot`.
            f. Add `min_total_cost_for_tile` to `h`. This adds the estimated minimum movement and color change cost required for this specific unpainted tile.
        6. Return the final value of `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static

        self.goal_tiles = {}  # {tile_name: color}
        self.tile_coords = {} # {tile_name: (row, col)}
        self.tile_neighbors = {} # {tile_name: set(neighbor_names)}
        self.available_colors = set()
        self.all_tiles = set()

        # Parse goals
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile_name, color = parts[1], parts[2]
                self.goal_tiles[tile_name] = color

        # Parse static facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "available-color":
                self.available_colors.add(parts[1])
            elif predicate in ["up", "down", "left", "right"]:
                t1, t2 = parts[1], parts[2]
                self.all_tiles.add(t1)
                self.all_tiles.add(t2)
                self.tile_neighbors.setdefault(t1, set()).add(t2)
                self.tile_neighbors.setdefault(t2, set()).add(t1)

        # Parse tile coordinates from names
        for tile_name in self.all_tiles:
             self.tile_coords[tile_name] = parse_tile_name(tile_name)

    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        unpainted_goal_tiles = set()
        for goal_fact_str in self.goals:
            if goal_fact_str not in state:
                 # Goal fact is (painted T C)
                 parts = get_parts(goal_fact_str)
                 tile_name = parts[1]
                 color = parts[2]
                 unpainted_goal_tiles.add((tile_name, color))

        # If all goal tiles are painted, but goal is not reached, something is wrong
        # (e.g., other goal conditions exist, but domain file suggests only painted facts)
        # Based on domain/instance examples, goals are only painted facts.
        # So if unpainted_goal_tiles is empty, goals <= state should be true.
        # The check at the beginning handles the goal state.
        if not unpainted_goal_tiles:
             # This case should ideally be covered by the initial goal check,
             # but as a safeguard if goal contains non-painted facts.
             # If there are no unpainted tiles but goal isn't met, heuristic should be > 0.
             # However, given the domain, this implies the state IS a goal state.
             # So this branch should not be reached if the first check passes.
             # Let's return 0 as per the first check's logic.
             return 0


        robot_info = {} # {robot_name: {'location': tile_name, 'color': color}}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot_name, tile_name = parts[1], parts[2]
                robot_info.setdefault(robot_name, {})['location'] = tile_name
            elif parts[0] == "robot-has":
                robot_name, color = parts[1], parts[2]
                robot_info.setdefault(robot_name, {})['color'] = color

        h = 0

        # For each unpainted goal tile, estimate the cost to paint it
        for tile_name, required_color in unpainted_goal_tiles:
            # Cost for the paint action itself
            h += 1

            tile_coords = self.tile_coords.get(tile_name)
            if tile_coords is None:
                 # Should not happen in valid problems, but handle defensively
                 # If a goal tile doesn't exist in the static grid definition, it's unreachable.
                 # Return infinity or a very large number? For greedy search, just a large number is fine.
                 # Let's assume valid problems where goal tiles are part of the grid.
                 # If it happens, it indicates an issue with the problem definition or parsing.
                 # Returning a large number signals this state is likely bad.
                 return math.inf # Or a large constant like 1000000

            neighbors_of_tile = self.tile_neighbors.get(tile_name, set())

            if not neighbors_of_tile:
                 # A tile with no neighbors cannot be painted (paint action requires adjacent tile).
                 # If this tile is a goal tile and unpainted, the problem is unsolvable from here.
                 return math.inf # Or a large constant

            min_total_cost_for_tile = math.inf

            # Find the minimum cost for any robot to get into position and color
            if not robot_info:
                 # No robots, cannot paint. Unsolvable.
                 return math.inf # Or a large constant

            for robot_name, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                if robot_location is None or robot_color is None:
                    # Robot info is incomplete in state, skip this robot
                    continue

                robot_coords = self.tile_coords.get(robot_location)
                if robot_coords is None:
                    # Robot is at a location not in the grid? Unsolvable from here.
                    return math.inf # Or a large constant

                min_dist_robot_to_neighbors = math.inf

                # Find minimum distance from robot to any neighbor of the target tile
                for neighbor_tile in neighbors_of_tile:
                    neighbor_coords = self.tile_coords.get(neighbor_tile)
                    if neighbor_coords is None:
                         # Neighbor tile not in grid? Problem definition issue.
                         continue # Skip this neighbor

                    dist = manhattan_dist(robot_coords, neighbor_coords)
                    min_dist_robot_to_neighbors = min(min_dist_robot_to_neighbors, dist)

                # If min_dist_robot_to_neighbors is still inf, this robot cannot reach any neighbor.
                # This might happen if the grid is disconnected or neighbors weren't parsed correctly.
                # If it's inf, this robot cannot paint this tile.
                if min_dist_robot_to_neighbors == math.inf:
                    cost_for_robot = math.inf
                else:
                    # Cost for this robot to paint this tile = movement cost + color change cost
                    cost_for_robot = min_dist_robot_to_neighbors

                    if robot_color != required_color:
                        # Assuming the required_color is always available if it's in the goal
                        # (checked in __init__ from available-color facts, but not strictly necessary
                        # for heuristic if we assume goal colors are always available).
                        # Change color action cost
                        cost_for_robot += 1

                min_total_cost_for_tile = min(min_total_cost_for_tile, cost_for_robot)

            # If min_total_cost_for_tile is still inf after checking all robots,
            # no robot can reach a neighbor of this tile. Unsolvable from here.
            if min_total_cost_for_tile == math.inf:
                 return math.inf # Or a large constant

            # Add the minimum movement/color cost required for this tile
            h += min_total_cost_for_tile

        return h
