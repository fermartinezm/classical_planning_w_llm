import math
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments unless args contains wildcards
    # A simpler check: just zip and compare. fnmatch handles the wildcard logic.
    if len(parts) != len(args):
         # This handles cases like matching "(robot-at robot1 tile_0_1)" with ("robot-at", "*")
         # which would fail zip. Let's allow partial matching from the start.
         # However, the typical use is matching predicate and arguments, so len should match.
         # Let's stick to the original logic assuming full pattern match.
         if not all(arg == '*' for arg in args): # Allow pattern like ('at', '*') to match ('at', 'ball1', 'rooma')
              # Re-implementing match to be more flexible like the Gripper example
              # The Gripper example's match function is slightly different:
              # It zips parts and args and checks all. This implies len(parts) == len(args).
              # Let's use that interpretation as it's provided.
              # If the lengths don't match, it's not a match.
              return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """
    Parses a tile name string like 'tile_X_Y' into integer coordinates (X, Y).
    Assumes tile names are always in this format.
    """
    try:
        _, row_str, col_str = tile_name.split('_')
        return (int(row_str), int(col_str))
    except ValueError:
        # Handle unexpected tile name format if necessary, or assume valid input
        raise ValueError(f"Unexpected tile name format: {tile_name}")

def manhattan_distance(coords1, coords2):
    """Calculates the Manhattan distance between two coordinate pairs (r1, c1) and (r2, c2)."""
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing:
    1. The number of tiles that still need to be painted with their goal color.
    2. The number of distinct colors required for the unpainted goal tiles.
    3. The minimum Manhattan distance from the robot's current location to any tile
       adjacent to any of the unpainted goal tiles.

    # Assumptions:
    - Tiles are arranged in a grid and named 'tile_row_col'.
    - Adjacency predicates (up, down, left, right) define grid connections.
    - Once a tile is painted, it cannot be repainted (paint action requires 'clear').
    - The robot always holds a color (no 'free-color' state used in actions).

    # Heuristic Initialization
    - Extracts goal conditions to know which tiles need which colors.
    - Builds a map from tile names to grid coordinates.
    - Builds an adjacency map for tiles based on static 'up', 'down', 'left', 'right' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted ?tile ?color)`.
    2. For each such goal condition `(painted T C)`:
       - Check the current state:
         - If `(painted T C)` is true, this goal is met for this tile. Ignore.
         - If `(painted T C')` is true for any `C' != C`, the goal is unreachable. Return infinity.
         - If `(clear T)` is true, this tile `T` needs to be painted with color `C`. Add `(T, C)` to a list of unpainted goals. Add `C` to a set of required colors.
    3. Count the number of unpainted goal tiles (`N`).
    4. Count the number of distinct colors required for these tiles (`N_colors`).
    5. If `N` is 0, the goal is reached, return 0.
    6. Find the robot's current location (`R_loc`) from the state.
    7. Calculate the minimum Manhattan distance from `R_loc` to *any* tile `X` that is adjacent to *any* tile `T` in the list of unpainted goals. Let this minimum distance be `D_min`.
       - Iterate through each `(T, C)` in the unpainted goals list.
       - Get the coordinates for `T`.
       - Find all tiles `X` adjacent to `T` using the pre-computed adjacency map.
       - For each adjacent tile `X`, get its coordinates.
       - Calculate the Manhattan distance from the robot's coordinates to `X`'s coordinates.
       - Keep track of the overall minimum distance found across all unpainted goal tiles and all their adjacent tiles.
    8. The heuristic value is `N + N_colors + D_min`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, tile coordinates,
        and tile adjacency information from the task.
        """
        super().__init__(task) # Call the base class constructor

        self.goals = task.goals
        static_facts = task.static

        # Store goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

        # Build map from tile name to coordinates (row, col)
        self.tile_coords = {}
        # We need to find all tile objects first. They are listed in the instance file.
        # The task object doesn't directly give us the list of all objects by type.
        # We can infer tile names from the static facts (up, down, left, right)
        # or from the goal facts. Let's use the goal tiles and their neighbors
        # mentioned in static facts to build the coordinate map.
        # A more robust way would be to parse the objects section of the PDDL,
        # but we only have access to task.static and task.goals here.
        # Let's assume all tiles involved in goals or static adjacency facts
        # are relevant and follow the tile_X_Y format.

        all_relevant_tiles = set(self.goal_tiles.keys())
        self.adjacency_map = {} # {tile: [adjacent_tiles]}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                t1, t2 = parts[1], parts[2]
                all_relevant_tiles.add(t1)
                all_relevant_tiles.add(t2)
                # Build adjacency map bidirectionally
                self.adjacency_map.setdefault(t1, []).append(t2)
                self.adjacency_map.setdefault(t2, []).append(t1)

        # Populate tile_coords from relevant tiles
        for tile_name in all_relevant_tiles:
            try:
                self.tile_coords[tile_name] = parse_tile_name(tile_name)
            except ValueError:
                 # If a tile name doesn't match the expected format,
                 # we can't calculate distance for it. This might indicate
                 # an issue with the PDDL instance or the assumption.
                 # For this heuristic, we'll skip tiles that don't parse.
                 # A production heuristic might need more robust parsing.
                 print(f"Warning: Could not parse tile name {tile_name}. Skipping.")
                 continue # Skip this tile if parsing fails


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        unpainted_goals = [] # List of (tile, color) tuples for tiles needing painting
        required_colors = set() # Set of colors needed for unpainted tiles

        # Check goal conditions against the current state
        for tile, goal_color in self.goal_tiles.items():
            is_painted_correctly = False
            is_painted_wrongly = False
            is_clear = False

            # Check the state for the status of this tile
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "painted" and len(parts) == 3 and parts[1] == tile:
                    painted_color = parts[2]
                    if painted_color == goal_color:
                        is_painted_correctly = True
                    else:
                        is_painted_wrongly = True
                    break # Found painted status, no need to check other facts for this tile
                elif parts[0] == "clear" and len(parts) == 2 and parts[1] == tile:
                    is_clear = True
                    # Don't break yet, might also be painted (though domain suggests not)

            if is_painted_wrongly:
                # If a tile is painted with the wrong color, the goal is unreachable
                return float('inf')

            if not is_painted_correctly:
                 # If it's not painted correctly, it must be clear (based on domain logic)
                 # We explicitly check for clear just in case, but the logic implies
                 # if not painted, and not wrongly painted, it must be clear.
                 # Let's rely on the 'clear' predicate explicitly as per domain.
                 if is_clear:
                    unpainted_goals.append((tile, goal_color))
                    required_colors.add(goal_color)
                 # Note: If a tile is *neither* painted nor clear, something is wrong
                 # with the state representation or domain model. We assume valid states.


        # If all goal tiles are painted correctly, the goal is reached
        if not unpainted_goals:
            return 0

        # Heuristic component 1: Number of tiles left to paint
        h_tiles = len(unpainted_goals)

        # Heuristic component 2: Number of distinct colors needed
        h_colors = len(required_colors)

        # Heuristic component 3: Minimum distance to get to the vicinity of work
        # Find robot's current location
        robot_location = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_location = get_parts(fact)[2]
                break

        if robot_location is None:
             # Robot location not found - indicates an invalid state
             return float('inf') # Should not happen in valid planning problems

        # Get robot coordinates
        robot_coords = self.tile_coords.get(robot_location)
        if robot_coords is None:
             # Robot is at a tile we couldn't parse coordinates for
             return float('inf') # Should not happen with valid tile names

        # Calculate minimum distance from robot to any tile adjacent to any unpainted goal tile
        min_dist_to_adjacent_unpainted = float('inf')

        for tile_to_paint, _ in unpainted_goals:
            tile_coords = self.tile_coords.get(tile_to_paint)
            if tile_coords is None:
                 # Skip tiles we couldn't parse coordinates for
                 continue

            adjacent_tiles = self.adjacency_map.get(tile_to_paint, [])

            for adj_tile in adjacent_tiles:
                adj_coords = self.tile_coords.get(adj_tile)
                if adj_coords is None:
                     # Skip adjacent tiles we couldn't parse coordinates for
                     continue

                dist = manhattan_distance(robot_coords, adj_coords)
                min_dist_to_adjacent_unpainted = min(min_dist_to_adjacent_unpainted, dist)

        # If min_dist_to_adjacent_unpainted is still infinity, it means there are unpainted
        # goal tiles but no valid adjacent tiles found for them (e.g., parsing failed
        # for all relevant tiles/adjacencies). This shouldn't happen in valid problems
        # with the assumed tile naming and grid structure.
        if min_dist_to_adjacent_unpainted == float('inf'):
             # This case implies the unpainted goal tiles are somehow isolated or invalid
             # based on the static facts provided. Treat as unreachable.
             return float('inf')


        h_distance = min_dist_to_adjacent_unpainted

        # Total heuristic value
        # Summing these components provides a non-admissible estimate.
        # h = h_tiles + h_colors + h_distance
        # Let's refine the color cost slightly: if the robot already has one of the needed colors,
        # maybe the first color change is saved.
        # Find robot's current color
        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                break

        # Adjust color cost: if robot has a color needed for an unpainted tile,
        # we might save one color change action compared to needing to acquire
        # the first color.
        # A simple adjustment: if the robot's current color is one of the required colors,
        # subtract 1 from the number of required colors, minimum 0.
        # This is still a simplification, as the robot might need to switch back and forth.
        # Let's stick to the simpler count of distinct colors needed, as it's a reasonable
        # non-admissible estimate of the *variety* of tasks remaining.
        # h_colors = len(required_colors) # Already calculated

        # The sum h_tiles + h_colors + h_distance seems a reasonable non-admissible heuristic.
        # h_tiles: Lower bound on paint actions.
        # h_colors: Rough estimate of color changes needed (at least one per new color).
        # h_distance: Estimate of movement to get to the first painting task.

        return h_tiles + h_colors + h_distance

