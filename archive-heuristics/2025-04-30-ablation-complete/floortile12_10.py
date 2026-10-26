# Add necessary imports
from fnmatch import fnmatch
import math # Used for float('inf')
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

# Utility functions (copied from examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to handle cases where lengths differ, fnmatch handles '*'
    # The match is successful only if all provided args match the corresponding parts.
    # If the fact has more parts than args, the extra parts are ignored.
    # If the fact has fewer parts than args, the zip stops, and the match is true
    # if all available parts matched. This seems acceptable for pattern matching.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Helper function to parse tile coordinates
def get_coords(tile_name):
    """Parses tile name 'tile_row_col' into (row, col) tuple."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        pass # Handle cases where tile name format is unexpected
    return None # Indicate parsing failed

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the cost components for each unpainted goal tile:
    1. The paint action itself.
    2. The minimum movement cost for any robot to reach a tile adjacent to the target tile.
    3. A cost for acquiring the necessary color if no robot currently holds it (counted once per missing color type).
    It returns infinity if a goal tile is painted with the wrong color, indicating an unsolvable state.

    # Assumptions
    - Tiles are arranged in a grid and named 'tile_row_col'.
    - Movement is restricted to adjacent clear tiles (Manhattan distance applies to movement cost).
    - Painting a tile requires a robot to be on an adjacent tile, hold the correct color, and the target tile must be clear.
    - If a goal tile is painted with a color different from the goal color, the state is unsolvable (no unpaint action).
    - The goal only consists of 'painted' predicates.
    - All tiles involved in the problem (initial state, static facts, goals) follow the 'tile_row_col' naming convention.

    # Heuristic Initialization
    - Parses static facts and initial state to build a map of tile names to grid coordinates and precompute adjacency information.
    - Stores the goal conditions for easy lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles that are not currently painted with the correct color.
    2. Check for unsolvability: For each unpainted goal tile, verify that it is currently 'clear'. If any goal tile is not clear AND not painted with the goal color, the state is unsolvable, return infinity.
    3. Initialize total heuristic cost to 0.
    4. Add the number of unpainted goal tiles to the total cost (representing the paint actions).
    5. For each unpainted goal tile:
        a. Find its grid coordinates.
        b. Find the coordinates of all existing adjacent tiles.
        c. Find the current grid coordinates of all robots.
        d. Calculate the minimum Manhattan distance from any robot's current location to any of the adjacent tiles of the target tile.
        e. Add this minimum distance to the total cost.
    6. Identify the set of colors required by the unpainted goal tiles.
    7. Identify the set of colors currently held by any robot.
    8. Count how many colors are in the required set but not in the held set. Add this count to the total cost (representing color change actions).
    9. Return the total cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all tile objects

        # Store goal conditions for easy lookup: {tile_name: goal_color}
        self.goal_painted = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goal only contains 'painted' predicates based on examples
            if parts and parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color
            # If goal contains other predicates, they are ignored by this heuristic.
            # This is acceptable for a domain-dependent heuristic focused on painting.


        # Build tile coordinate map and adjacency list
        self.tile_coords = {} # Map tile name to (row, col)
        self.coords_tile = {} # Map (row, col) to tile name
        max_row = -1
        max_col = -1

        # Collect all potential tile names from static facts and initial state
        all_potential_tile_names = set()
        # Iterate through all facts in static and initial state to find objects of type tile
        # Look for predicates that involve tiles
        tile_predicates = {'robot-at', 'up', 'down', 'left', 'right', 'clear', 'painted'}
        for fact in static_facts | initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in tile_predicates:
                 # Check all parts, as tile can be in different positions depending on predicate
                 for part in parts:
                     if part.startswith('tile_'):
                         all_potential_tile_names.add(part)

        # Also add tiles from goal conditions, just in case
        all_potential_tile_names.update(self.goal_painted.keys())


        # Parse coordinates and find grid dimensions
        for tile_name in all_potential_tile_names:
            coords = get_coords(tile_name)
            if coords is not None:
                r, c = coords
                self.tile_coords[tile_name] = (r, c)
                self.coords_tile[(r, c)] = tile_name
                max_row = max(max_row, r)
                max_col = max(max_col, c)

        self.grid_rows = max_row + 1 if max_row >= 0 else 0
        self.grid_cols = max_col + 1 if max_col >= 0 else 0

        # Precompute adjacency list based on grid structure
        self.adj_tiles = {}
        for tile_name, (r, c) in self.tile_coords.items():
            adjacent = []
            # Check up
            if r > 0 and (r - 1, c) in self.coords_tile:
                adjacent.append(self.coords_tile[(r - 1, c)])
            # Check down
            if r < self.grid_rows - 1 and (r + 1, c) in self.coords_tile:
                adjacent.append(self.coords_tile[(r + 1, c)])
            # Check left
            if c > 0 and (r, c - 1) in self.coords_tile:
                adjacent.append(self.coords_tile[(r, c - 1)])
            # Check right
            if c < self.grid_cols - 1 and (r, c + 1) in self.coords_tile:
                adjacent.append(self.coords_tile[(r, c + 1)])
            self.adj_tiles[tile_name] = adjacent

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unsatisfied goal tiles and check for unsolvability
        unsatisfied_goals = [] # List of (tile, goal_color) tuples
        current_painted = {} # Map tile to current color if painted
        current_clear = set() # Set of clear tiles

        # Populate current painted and clear status from the state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'painted' and len(parts) == 3:
                current_painted[parts[1]] = parts[2]
            elif parts[0] == 'clear' and len(parts) == 2:
                current_clear.add(parts[1])

        for tile, goal_color in self.goal_painted.items():
            # Check if the tile is painted with the correct color in the current state
            is_painted_correctly = (tile in current_painted and current_painted[tile] == goal_color)

            if not is_painted_correctly:
                # This goal is not satisfied.
                # Check for unsolvability: Is the tile painted with the wrong color?
                # If it's not clear and not painted correctly, it's painted wrong.
                # A tile is either clear or painted. So, if it's not clear, it must be painted.
                if tile not in current_clear:
                     # Since it's not painted correctly (checked above) and not clear,
                     # it must be painted with a different color.
                     # State is unsolvable.
                     return math.inf

                # If it's not painted correctly but is clear, it needs painting.
                unsatisfied_goals.append((tile, goal_color))

        # If all goal tiles are painted correctly and no unsolvable state found
        if not unsatisfied_goals:
            return 0

        total_cost = 0

        # 2. Calculate paint cost
        # Each unpainted goal tile needs one paint action.
        total_cost += len(unsatisfied_goals)

        # 3. Calculate movement cost
        robot_locations = {} # Map robot name to tile name
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile

        # If there are unpainted goals but no robots, it's unsolvable.
        if not robot_locations and unsatisfied_goals:
             return math.inf


        for tile, goal_color in unsatisfied_goals:
            tile_coords = self.tile_coords.get(tile)
            # This should not happen if tile names are consistent and parsed correctly
            if tile_coords is None:
                 return math.inf # Unrecognized tile in goal

            min_dist_to_adjacent = math.inf
            possible_adj_tiles = self.adj_tiles.get(tile, [])

            # If a goal tile has no adjacent tiles in the grid, it cannot be painted
            # by moving from an adjacent tile. This state is likely unsolvable
            # if this tile needs painting.
            if not possible_adj_tiles:
                 return math.inf

            for robot, robot_tile in robot_locations.items():
                robot_coords = self.tile_coords.get(robot_tile)
                # This should not happen if robot locations are valid tiles
                if robot_coords is None:
                     continue

                for adj_tile in possible_adj_tiles:
                    adj_coords = self.tile_coords.get(adj_tile)
                    # This should not happen if adjacent tiles are valid
                    if adj_coords is None:
                         continue
                    # Calculate Manhattan distance
                    dist = abs(robot_coords[0] - adj_coords[0]) + abs(robot_coords[1] - adj_coords[1])
                    min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

            # If min_dist_to_adjacent is still infinity, it means no robot could reach
            # any adjacent tile. This implies unsolvability for this tile.
            if min_dist_to_adjacent == math.inf:
                 return math.inf

            # Add the minimum movement cost required for this specific tile
            total_cost += min_dist_to_adjacent

        # 4. Calculate color change cost
        needed_colors = {color for tile, color in unsatisfied_goals}
        held_colors = {parts[2] for fact in state if get_parts(fact)[0] == 'robot-has' and len(parts) == 3}

        missing_colors_count = 0
        for color in needed_colors:
            if color not in held_colors:
                missing_colors_count += 1

        # Add the cost for acquiring missing colors
        total_cost += missing_colors_count

        return total_cost
