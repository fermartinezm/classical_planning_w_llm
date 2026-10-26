from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_r_c' into a (row, col) tuple."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            # PDDL examples show tile_0_1 is above tile_1_1, etc.
            # So row 0 is at the top, row index increases downwards.
            # Column index increases rightwards.
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            pass # Not a valid tile name format
    return None # Indicate parsing failure

def manhattan_distance(tile1, tile2, tile_coords):
    """Computes the Manhattan distance between two tiles using precomputed coordinates."""
    coord1 = tile_coords.get(tile1)
    coord2 = tile_coords.get(tile2)
    if coord1 is None or coord2 is None:
        # This might happen if a tile is mentioned in state/goal but not static (malformed instance)
        # Or if parse_tile_name failed.
        # Returning infinity makes this path unattractive.
        return float('inf')
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the required number of actions (paint, color change, movement)
    to satisfy the unpainted goal tiles. It sums the number of unpainted goal tiles
    (for paint actions), the number of new colors needed by robots, and the minimum
    movement cost for robots to reach painting locations.

    # Assumptions
    - The tiles form a grid structure defined by 'up', 'down', 'left', 'right' predicates.
    - A tile painted with the wrong color cannot be repainted (makes the state unsolvable).
    - Robots always hold a color (no 'free-color' state to handle for color acquisition).
    - Movement cost is approximated by Manhattan distance, ignoring the 'clear' precondition for movement.

    # Heuristic Initialization
    - Parses static facts to build:
        - `goal_tiles`: A set of (tile, color) tuples representing the desired painted state.
        - `adjacency_map`: Maps each tile to a set of its adjacent tiles based on 'up', 'down', 'left', 'right' predicates.
        - `tile_coords`: Maps each tile name string to its (row, col) integer coordinates.
        - `available_colors`: A set of all colors available in the domain.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to identify:
       - `current_painted`: Set of (tile, color) tuples for painted tiles.
       - `current_clear`: Set of clear tiles.
       - `robot_locations`: Dictionary mapping robot name to its current tile location.
       - `robot_colors`: Dictionary mapping robot name to the color it currently holds.
    2. Identify `unpainted_goal_tiles`: The set of (tile, color) pairs from `goal_tiles` that are not present in `current_painted`.
    3. Check for impossible states: If any tile in `unpainted_goal_tiles` is currently painted with a *different* color in `current_painted`, the state is likely unsolvable. Return a large heuristic value (e.g., 1000).
    4. Initialize the heuristic value `h` to 0.
    5. Add cost for paint actions: Increment `h` by the number of tiles in `unpainted_goal_tiles`. Each requires one paint action.
    6. Add cost for color changes:
       - Identify the set of colors `needed_colors` required by the tiles in `unpainted_goal_tiles`.
       - Identify the set of colors `colors_held` currently held by robots.
       - The number of colors in `needed_colors` that are *not* in `colors_held` represents the minimum number of `change_color` actions needed to make those colors available to robots. Add this count to `h`.
    7. Add cost for movement:
       - Identify the set of all tiles adjacent to the tiles in `unpainted_goal_tiles`. These are the potential "painting spots".
       - For each painting spot, calculate the minimum Manhattan distance from *any* robot's current location to that spot.
       - Sum these minimum distances for all painting spots and add the total to `h`. This estimates the total movement effort required to position robots near the tiles that need painting.
    8. Return the calculated heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.goal_tiles = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                # Goal is (painted tile color)
                if len(args) == 2:
                    self.goal_tiles.add((args[0], args[1]))

        self.adjacency_map = {}
        self.tile_coords = {}
        self.available_colors = set()

        # Parse static facts to build adjacency map, tile coordinates, and available colors
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]

            if predicate in ["up", "down", "left", "right"]:
                # Example: (up tile_1_1 tile_0_1) means tile_1_1 is adjacent to tile_0_1
                if len(parts) == 3:
                    tile1, tile2 = parts[1], parts[2]
                    self.adjacency_map.setdefault(tile1, set()).add(tile2)
                    self.adjacency_map.setdefault(tile2, set()).add(tile1)

                    # Parse coordinates if not already done
                    if tile1 not in self.tile_coords:
                        coords1 = parse_tile_name(tile1)
                        if coords1: self.tile_coords[tile1] = coords1
                    if tile2 not in self.tile_coords:
                        coords2 = parse_tile_name(tile2)
                        if coords2: self.tile_coords[tile2] = coords2


            elif predicate == "available-color":
                if len(parts) == 2:
                    self.available_colors.add(parts[1])

        # Ensure all tiles mentioned in goals or adjacency map have coordinates
        all_mentioned_tiles = set()
        for tile, _ in self.goal_tiles:
            all_mentioned_tiles.add(tile)
        for tile in self.adjacency_map:
             all_mentioned_tiles.add(tile)
        for adj_set in self.adjacency_map.values():
             all_mentioned_tiles.update(adj_set)

        for tile in all_mentioned_tiles:
             if tile not in self.tile_coords:
                 coords = parse_tile_name(tile)
                 if coords: self.tile_coords[tile] = coords


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        current_painted = set()
        current_clear = set()
        robot_locations = {} # {robot: tile}
        robot_colors = {}    # {robot: color}

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "painted":
                if len(parts) == 3:
                    current_painted.add((parts[1], parts[2]))
            elif predicate == "clear":
                if len(parts) == 2:
                    current_clear.add(parts[1])
            elif predicate == "robot-at":
                if len(parts) == 3:
                    robot_locations[parts[1]] = parts[2]
            elif predicate == "robot-has":
                if len(parts) == 3:
                    robot_colors[parts[1]] = parts[2]
            # Ignore free-color based on assumption

        # Identify unpainted goal tiles
        unpainted_goal_tiles = {
            (tile, color) for (tile, color) in self.goal_tiles
            if (tile, color) not in current_painted
        }

        # Check for impossible states (goal tile painted with wrong color)
        for tile, goal_color in unpainted_goal_tiles:
            # Check if this tile is painted with *any* color
            for fact in state:
                 if match(fact, "painted", tile, "*"):
                     painted_tile, painted_color = get_parts(fact)[1:]
                     if painted_color != goal_color:
                         return 1000 # Large value indicating likely unsolvable

        # If all goal tiles are painted correctly, heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        h = 0

        # 1. Add cost for paint actions
        h += len(unpainted_goal_tiles)

        # 2. Add cost for color changes
        needed_colors = {color for (tile, color) in unpainted_goal_tiles}
        colors_held = set(robot_colors.values())
        colors_to_acquire = needed_colors - colors_held
        h += len(colors_to_acquire)

        # 3. Add cost for movement
        painting_spots = set()
        for tile, _ in unpainted_goal_tiles:
            # Ensure the tile exists in the adjacency map before trying to get neighbors
            if tile in self.adjacency_map:
                painting_spots.update(self.adjacency_map[tile])

        movement_cost = 0
        if painting_spots and robot_locations: # Only calculate movement if there are spots to reach and robots exist
            for spot in painting_spots:
                min_dist_to_spot = float('inf')
                # Ensure the spot tile has coordinates before calculating distance
                if spot not in self.tile_coords:
                     continue # Cannot calculate distance if coordinates are missing

                for robot, loc_r in robot_locations.items():
                    # Ensure robot location tile has coordinates
                    if loc_r not in self.tile_coords:
                         continue # Cannot calculate distance if coordinates are missing

                    dist = manhattan_distance(spot, loc_r, self.tile_coords)
                    min_dist_to_spot = min(min_dist_to_spot, dist)

                if min_dist_to_spot != float('inf'):
                    movement_cost += min_dist_to_spot

        h += movement_cost

        return h
