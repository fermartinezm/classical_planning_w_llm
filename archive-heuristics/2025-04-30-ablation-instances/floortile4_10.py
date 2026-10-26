# from heuristics.heuristic_base import Heuristic # Assuming this base class is provided in the environment
from fnmatch import fnmatch

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Or raise an error, depending on expected input robustness
        return [] # Return empty list for invalid format
    return fact[1:-1].split()

# Helper function to match PDDL fact patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_0)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to parse tile names like 'tile_X_Y'
def parse_tile_name(tile_name):
    """Extract row and column integers from a tile name string like 'tile_X_Y'."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            # PDDL tile names are often 1-indexed in problems, but 0-indexed in internal representation?
            # The example state/static facts use tile_0_1, tile_1_1, etc. suggesting 0-indexed or 1-indexed depending on interpretation.
            # Let's assume the numbers directly correspond to grid indices for Manhattan distance.
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
    except (ValueError, IndexError):
        pass # Handle potential errors if name format is unexpected
    # Return None or raise error for invalid format
    return None # Or raise ValueError(f"Invalid tile name format: {tile_name}")

# Helper function to calculate Manhattan distance
def manhattan_distance(coords1, coords2):
    """Calculate the Manhattan distance between two grid coordinates (row, col)."""
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance for invalid coordinates
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated costs
    for each tile that is not yet painted with its required goal color. The estimated
    cost for a single unpainted goal tile includes the cost to change the robot's
    color (if needed), the estimated movement cost for the robot to reach a tile
    adjacent to the target tile, and the cost of the paint action itself.

    # Assumptions
    - The goal state requires specific tiles to be painted with specific colors.
    - Tiles are identified by names like 'tile_X_Y' where X is the row and Y is the column.
    - The grid is rectangular and connected by 'up', 'down', 'left', 'right' relations.
    - Tiles are only painted once to their goal color; a tile painted with the wrong
      color relative to the goal is considered a state from which the goal is
      unreachable (or is ignored by this heuristic).
    - The robot always holds exactly one color.
    - Movement cost is approximated by Manhattan distance to the nearest adjacent tile,
      ignoring the 'clear' precondition for intermediate tiles.

    # Heuristic Initialization
    - Parses goal facts to identify which tiles need to be painted and with which color.
    - Parses static facts to:
        - Identify available colors.
        - Build a mapping from tile names to their grid coordinates (row, col).
        - Build an adjacency map for tiles based on 'up', 'down', 'left', 'right' relations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is the goal state. If yes, return 0.
    2. Identify the robot's current location and the color it is holding.
    3. Identify which tiles are currently painted and with what color.
    4. Initialize the total heuristic cost to 0.
    5. Iterate through each tile specified in the goal that needs to be painted with a specific color C.
    6. For the current goal tile T and required color C:
       - Check if tile T is already painted with color C in the current state. If yes, this goal is satisfied for this tile; continue to the next goal tile.
       - If tile T is painted with a *different* color C' in the current state, this heuristic assumes this state is not on a path to the goal and effectively ignores this tile (or could add a large penalty, but ignoring is simpler for non-admissible).
       - If tile T is *not* painted (i.e., it is clear or robot is on it, but robot cannot be on a tile that needs painting) in the current state:
         - Calculate the cost components needed to paint this tile:
           - Paint action cost: 1 (for paint_up/down action).
           - Color change cost: 1 if the robot's current color is not C, otherwise 0. This assumes the required color C is available (which is checked during initialization).
           - Movement cost: Estimate the minimum number of moves required for the robot to reach *any* tile adjacent to T. This is approximated by finding the minimum Manhattan distance between the robot's current tile and any of T's adjacent tiles.
         - Add the sum of paint action cost, color change cost, and movement cost to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.task = task # Store task to check goal state
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goal facts: Map tile to required color
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                # Goal is (painted tile color)
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_paintings[tile] = color
                # else: print(f"Warning: Unexpected goal format: {goal}") # Optional warning

        # 2. Parse static facts
        self.available_colors = set()
        self.tile_coords = {} # Map tile name to (row, col)
        self.adjacency_map = {} # Map tile name to set of adjacent tile names

        all_tiles = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]

            if predicate == "available-color":
                if len(parts) == 2:
                    self.available_colors.add(parts[1])
                # else: print(f"Warning: Unexpected available-color format: {fact}") # Optional warning

            elif predicate in ["up", "down", "left", "right"]:
                if len(parts) == 3:
                    tile1, tile2 = parts[1], parts[2]
                    # Add symmetric adjacency
                    self.adjacency_map.setdefault(tile1, set()).add(tile2)
                    self.adjacency_map.setdefault(tile2, set()).add(tile1)

                    all_tiles.add(tile1)
                    all_tiles.add(tile2)
                # else: print(f"Warning: Unexpected adjacency fact format: {fact}") # Optional warning

        # Extract coordinates from all identified tiles
        for tile_name in all_tiles:
             coords = parse_tile_name(tile_name)
             if coords is not None:
                 self.tile_coords[tile_name] = coords
             # else: print(f"Warning: Could not parse coordinates for tile: {tile_name}") # Optional warning


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to paint all goal tiles correctly.
        """
        state = node.state  # Current world state.

        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # 2. Identify robot's current location and color
        robot_tile = None
        robot_color = None
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "robot-at" and len(parts) == 3:
                # Assuming one robot named 'robot1'
                if parts[1] == 'robot1':
                    robot_tile = parts[2]
            elif parts[0] == "robot-has" and len(parts) == 3:
                 # Assuming one robot named 'robot1'
                 if parts[1] == 'robot1':
                    robot_color = parts[2]
            # Note: free-color is not used in actions, robot always has a color.

        if robot_tile is None or robot_color is None:
             # This state is likely invalid or represents a failure state
             # Return infinity
             return float('inf')

        # 3. Identify current tile paintings
        current_paintings = {} # Map tile name to current color
        # We don't strictly need clear_tiles set if we assume any tile not painted is clear
        # clear_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                current_paintings[tile] = color
            # elif parts[0] == "clear" and len(parts) == 2:
            #     tile = parts[1]
            #     clear_tiles.add(tile)

        total_cost = 0  # Initialize action cost counter.

        # Get robot coordinates
        robot_coords = self.tile_coords.get(robot_tile)
        if robot_coords is None:
             # Robot is at an unknown tile? Should not happen in valid states.
             return float('inf')

        # 4. Iterate through each goal tile
        for goal_tile, required_color in self.goal_paintings.items():
            # 5. Check if the goal tile is painted correctly
            is_painted_correctly = (goal_tile in current_paintings and
                                    current_paintings[goal_tile] == required_color)

            if is_painted_correctly:
                continue # This tile goal is satisfied

            # If not painted correctly, it needs painting.
            # We assume solvable problems don't have tiles painted wrong color.
            # So, if not painted correctly, it must be clear and need painting.

            # Calculate cost components
            paint_cost = 1

            # Cost to get the required color
            color_cost = 0
            if robot_color != required_color:
                 # Need to change color. Assumes required_color is available.
                 # The domain guarantees available-color facts for all colors used.
                 color_cost = 1

            # Cost to move to an adjacent tile
            move_cost = float('inf')
            adjacent_tiles = self.adjacency_map.get(goal_tile, set())

            if not adjacent_tiles:
                 # Tile has no adjacent tiles? Should not happen in a grid.
                 return float('inf')

            for adj_tile in adjacent_tiles:
                 adj_coords = self.tile_coords.get(adj_tile)
                 if adj_coords is not None:
                     dist = manhattan_distance(robot_coords, adj_coords)
                     move_cost = min(move_cost, dist)

            if move_cost == float('inf'):
                 # Could not find path to any adjacent tile (e.g., missing coords)
                 return float('inf')

            # Total cost for this unpainted goal tile
            total_cost += paint_cost + color_cost + move_cost

        return total_cost

# Note: The Heuristic base class is assumed to be provided in the environment.
# If running this code standalone, you would need a definition like:
# class Heuristic:
#     def __init__(self, task): self.task = task # Need task to check goal
#     def __call__(self, node): pass
