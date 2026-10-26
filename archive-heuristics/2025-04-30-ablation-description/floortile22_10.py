from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

# Utility functions to parse PDDL facts
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
    # Ensure we don't try to match more args than parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing up
    estimated costs for each individual unpainted goal tile. The estimated cost
    for a single tile includes the paint action itself, the cost to clear the
    tile if occupied, and the minimum cost for any robot to get the required
    color and move adjacent to the tile. It returns infinity if any goal tile
    is painted with the wrong color or is otherwise unreachable.

    # Assumptions
    - The domain represents a grid structure where tiles are named 'tile_row_col'.
    - Adjacency relations (up, down, left, right) define the grid connections.
    - All actions have a unit cost of 1.
    - Tiles, once painted, cannot be unpainted or repainted.
    - If a tile needs painting but is occupied by a robot, that robot must move off.
    - The heuristic assumes that if a tile is part of a goal and is mentioned in adjacency facts, it has valid coordinates and adjacent tiles defined in the static facts.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted and with which colors.
    - Parses static facts (`up`, `down`, `left`, `right`) to build:
        - An adjacency map (`tile -> set of adjacent tiles`).
        - A coordinate map (`tile -> (row, col)`) based on the 'tile_row_col' naming convention, used for Manhattan distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a state is calculated as follows:

    1. Identify all goal conditions of the form `(painted tile_X color_Y)`.
    2. Initialize `unpainted_goals` list and `unsolvable_due_to_wrong_color` flag.
    3. For each goal `(tile_X, color_Y)`:
        a. Check if `(painted tile_X color_Y)` is true in the current state. If yes, this goal is met.
        b. If not met, check if `tile_X` is painted with *any* color `Z` where `Z != Y`. If yes, set `unsolvable_due_to_wrong_color` to True and break.
        c. If not met and not painted wrong, add `(tile_X, color_Y)` to `unpainted_goals`.
    4. If `unsolvable_due_to_wrong_color` is True, return infinity.
    5. If `unpainted_goals` is empty, return 0 (goal state).
    6. Get current robot locations and colors from the state.
    7. Initialize the total heuristic value `total_h` to 0 and `unreachable_tile_found` flag to False.
    8. For each `(tile_X, color_Y)` in `unpainted_goals`:
        a. Initialize the cost for this specific tile `tile_h` to 0.
        b. Add 1 to `tile_h` for the paint action required for this tile.
        c. Check if `tile_X` is currently occupied by any robot. If it is, add 1 to `tile_h` (estimating the cost to move the robot off).
        d. Calculate the minimum cost for *any* robot to be prepared to paint `tile_X` with `color_Y`.
           - Initialize `min_robot_prep_cost` to infinity.
           - Get the set of tiles adjacent to `tile_X` from the precomputed `adjacent_map`.
           - If `tile_X` is not in the `coords_map` or has no adjacent tiles defined, this tile is unreachable. Set `unreachable_tile_found` to True and break the outer loop.
           - For each robot `R` and its current location `R_loc`:
               - Determine the `color_cost`: 1 if robot `R` does not have `color_Y` (i.e., `(robot-has R color_Y)` is not in the state), otherwise 0.
               - Find the minimum movement cost from `R_loc` to any adjacent tile `tile_Adj` using Manhattan distance. Initialize `min_move_cost_to_adjacent` to infinity.
               - For each `tile_Adj` adjacent to `tile_X`: calculate distance and update `min_move_cost_to_adjacent`.
               - If `min_move_cost_to_adjacent` is not infinity: Calculate `robot_prep_cost = color_cost + min_move_cost_to_adjacent` and update `min_robot_prep_cost`.
           - If `min_robot_prep_cost` is not infinity: Add `min_robot_prep_cost` to `tile_h`.
           - Else (no robot can get adjacent): Set `unreachable_tile_found` to True and break the outer loop.
        e. If `unreachable_tile_found` is True, break the outer loop.
        f. Add `tile_h` to `total_h`.
    9. If `unreachable_tile_found` is True, return infinity.
    10. Otherwise, return `total_h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Extract goal tiles and their required colors
        self.goal_tiles = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                if len(parts) == 3:
                    tile, color = parts[1:]
                    self.goal_tiles.add((tile, color))

        # Build adjacency map and coordinate map from static facts
        self.adjacent_map = {}
        self.coords_map = {}

        # Collect all tile names mentioned in adjacency facts
        all_tile_names = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                t1, t2 = parts[1:]
                self.adjacent_map.setdefault(t1, set()).add(t2)
                self.adjacent_map.setdefault(t2, set()).add(t1) # Grid is symmetric
                all_tile_names.add(t1)
                all_tile_names.add(t2)

        # Parse coordinates for collected tile names
        for tile_name in all_tile_names:
            try:
                parts = tile_name.split('_')
                # Assumes tile names are in the format 'tile_row_col'
                if len(parts) == 3 and parts[0] == 'tile':
                    row = int(parts[1])
                    col = int(parts[2])
                    self.coords_map[tile_name] = (row, col)
            except ValueError:
                # Ignore names that don't fit the expected pattern
                pass

    def get_coords(self, tile_name):
        """Get coordinates (row, col) for a tile name using the precomputed map."""
        return self.coords_map.get(tile_name)

    def get_manhattan_distance(self, tile1_name, tile2_name):
        """Calculate Manhattan distance between two tiles using their coordinates."""
        coords1 = self.get_coords(tile1_name)
        coords2 = self.get_coords(tile2_name)
        if coords1 is None or coords2 is None:
            # Cannot compute distance if coordinates are unknown
            return float('inf')
        return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


    def __call__(self, node):
        """Estimate the minimum cost to paint all goal tiles."""
        state = node.state

        unpainted_goals = []
        unsolvable_due_to_wrong_color = False

        # Identify unpainted goal tiles and check for unsolvable conditions (wrong color)
        for tile_X, color_Y in self.goal_tiles:
            is_painted_goal_color = f"(painted {tile_X} {color_Y})" in state

            if not is_painted_goal_color:
                # Check if it's painted the wrong color
                is_painted_wrong_color = False
                for fact in state:
                    if match(fact, "painted", tile_X, "?c"):
                        painted_color = get_parts(fact)[2]
                        if painted_color != color_Y:
                            is_painted_wrong_color = True
                            break
                if is_painted_wrong_color:
                    unsolvable_due_to_wrong_color = True
                    break # Found an unsolvable condition

                # If not painted correctly and not painted wrong, it needs painting
                unpainted_goals.append((tile_X, color_Y))

        # If unsolvable due to wrong color, return infinity
        if unsolvable_due_to_wrong_color:
            return float('inf')

        # If all goals are met, heuristic is 0
        if not unpainted_goals:
            return 0

        # Get current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "?r", "?loc"):
                robot, loc = get_parts(fact)[1:]
                robot_locations[robot] = loc
            elif match(fact, "robot-has", "?r", "?c"):
                robot, color = get_parts(fact)[1:]
                robot_colors[robot] = color

        total_h = 0
        unreachable_tile_found = False

        # Calculate cost components for each unpainted tile
        for tile_X, color_Y in unpainted_goals:
            tile_h = 0
            tile_h += 1 # Cost 1: The paint action itself

            # Cost 2: Clearing the tile if occupied
            is_occupied = False
            for robot, loc in robot_locations.items():
                if loc == tile_X:
                    is_occupied = True
                    break
            if is_occupied:
                tile_h += 1 # Need to move robot off

            # Cost 3 & 4: Getting the right color and getting adjacent
            min_robot_prep_cost = float('inf')
            adjacent_tiles = self.adjacent_map.get(tile_X, set())

            # Check if the tile is part of the known grid structure and has adjacent tiles
            if tile_X not in self.coords_map or not adjacent_tiles:
                 # If tile coordinates are unknown or it has no adjacent tiles, it's unreachable for painting.
                 unreachable_tile_found = True
                 break # Problem is unsolvable from here

            # Find the minimum preparation cost across all robots
            for robot, robot_loc in robot_locations.items():
                robot_color = robot_colors.get(robot)

                color_cost = 0
                if robot_color != color_Y:
                     color_cost = 1 # Need to change color

                min_move_cost_to_adjacent = float('inf')
                for tile_Adj in adjacent_tiles:
                    dist = self.get_manhattan_distance(robot_loc, tile_Adj)
                    min_move_cost_to_adjacent = min(min_move_cost_to_adjacent, dist)

                # Cost for this robot to be ready to paint this tile
                # This is cost to get color + cost to move adjacent
                # Only consider if movement to an adjacent tile is possible
                if min_move_cost_to_adjacent != float('inf'):
                    robot_prep_cost = color_cost + min_move_cost_to_adjacent
                    min_robot_prep_cost = min(min_robot_prep_cost, robot_prep_cost)

            # Add the minimum preparation cost found across all robots for this tile
            if min_robot_prep_cost != float('inf'):
                tile_h += min_robot_prep_cost
            else:
                 # No robot can get adjacent to this tile. Unreachable.
                 unreachable_tile_found = True
                 break # Problem is unsolvable from here

            total_h += tile_h

        if unreachable_tile_found:
             return float('inf') # Return infinity for unsolvable states

        return total_h
