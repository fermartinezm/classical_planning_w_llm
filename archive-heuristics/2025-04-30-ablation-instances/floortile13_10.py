# Add necessary imports
# Assuming Heuristic base class is available via 'from heuristics.heuristic_base import Heuristic'
# If not, a minimal base class might be needed, but the prompt implies it exists.
# from heuristics.heuristic_base import Heuristic

# Define helper functions outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_R_C' into (row, col) coordinates."""
    try:
        parts = tile_name.split('_')
        # Assuming format is always tile_row_col
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (ValueError, IndexError):
        # Handle unexpected tile name format if necessary, or assume consistent format
        # print(f"Warning: Could not parse tile name {tile_name}") # Avoid printing in heuristic
        return None

def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two coordinate pairs (r1, c1) and (r2, c2)."""
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

# Define the heuristic class
# Assuming Heuristic base class is available via 'from heuristics.heuristic_base import Heuristic'
# If not, a minimal base class might be needed, but the prompt implies it exists.
# from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles
    with the correct color. It sums three components: the number of unpainted
    goal tiles (for paint actions), an estimate of color changes needed, and
    an estimate of movement cost to reach the vicinity of unpainted tiles.

    # Assumptions
    - Tiles are arranged in a grid and named like 'tile_R_C'.
    - The goal is to paint specific tiles with specific colors.
    - Tiles painted with the wrong color cannot be repainted (based on domain actions).
      Therefore, the heuristic only considers goal tiles that are not yet painted
      with their target color.
    - All actions (move, paint, change_color) have a cost of 1.
    - There is a single robot named 'robot1'.

    # Heuristic Initialization
    - Extracts all tile names and their grid coordinates from static facts
      (using adjacency predicates like up, down, left, right) and goal facts
      to build a coordinate map.
    - Stores the goal conditions, specifically the required color for each goal tile.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts `(painted tile_X_Y color_Z)` that are not true
       in the current state. These are the unsatisfied painting goals.
    2. If there are no unsatisfied painting goals, the heuristic is 0 (goal state).
    3. If there are unsatisfied goals:
       a.  **Painting Cost:** Add 1 for each unsatisfied painting goal (each requires a `paint` action).
       b.  **Color Change Cost:** Identify the set of distinct colors required by the
           unsatisfied painting goals. Estimate the number of color changes needed.
           A simple estimate is the number of distinct colors needed, minus one
           if the robot currently holds one of those colors. Add this value to the heuristic.
       c.  **Movement Cost:** The robot needs to reach a tile adjacent to each
           unsatisfied goal tile. Estimate the minimum number of moves required
           to reach a tile adjacent to the *closest* unsatisfied goal tile from
           the robot's current location. Add this value to the heuristic.
           The minimum moves to get adjacent to a target tile from the robot's
           current tile is calculated using Manhattan distance: 1 if the robot
           is on the target tile, 0 if the robot is already adjacent, and
           Manhattan distance minus 1 otherwise.
    4. The total heuristic value is the sum of the painting cost, color change cost,
       and movement cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and grid structure."""
        self.goals = task.goals
        static_facts = task.static

        # Extract all tile names from static adjacency facts and goal facts
        tile_names = set()
        for fact in static_facts:
            parts = get_parts(fact)
            # Adjacency facts have 3 parts: predicate, tile1, tile2
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                tile_names.add(parts[1])
                tile_names.add(parts[2])

        # Add tiles mentioned in goal facts
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "painted" and len(parts) == 3:
                 tile_names.add(parts[1])


        # Map tile names to coordinates
        self.tile_coords = {}
        for tile_name in tile_names:
            coords = parse_tile_name(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords

        # Store goal colors for each goal tile
        self.goal_colors = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_colors[tile] = color

        # Assuming a single robot named 'robot1' based on example state
        # A more robust approach would extract robot names from initial state facts like (robot-at robot_name tile_name)
        self.robot_name = 'robot1'


    def manhattan_distance_tiles(self, tile1_name, tile2_name):
        """Calculates Manhattan distance between two tiles by name."""
        coords1 = self.tile_coords.get(tile1_name)
        coords2 = self.tile_coords.get(tile2_name)
        return manhattan_distance(coords1, coords2)

    def min_dist_to_adjacent(self, from_tile_name, target_tile_name):
        """
        Calculates the minimum moves from from_tile to any tile adjacent to target_tile.
        Assumes 1 move per Manhattan distance unit.
        Cost is 1 if from_tile is the same as target_tile (needs 1 move off),
        0 if from_tile is adjacent to target_tile,
        Manhattan distance - 1 otherwise.
        Returns float('inf') if target_tile is not in the grid.
        """
        dist = self.manhattan_distance_tiles(from_tile_name, target_tile_name)
        if dist == float('inf'):
            return float('inf') # Target tile not found in grid

        # If robot is on the target tile (dist 0), it needs 1 move to get adjacent.
        # If robot is adjacent (dist 1), it needs 0 moves to get adjacent.
        # If robot is further (dist > 1), it needs dist - 1 moves to get adjacent.
        return 1 if dist == 0 else max(0, dist - 1)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot's current location and color
        robot_loc = None
        robot_color = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at" and len(parts) == 3 and parts[1] == self.robot_name:
                robot_loc = parts[2]
            elif parts[0] == "robot-has" and len(parts) == 3 and parts[1] == self.robot_name:
                 robot_color = parts[2]

        # Identify unsatisfied painting goals
        unpainted_goals = [] # List of (tile_name, color)
        needed_colors = set()

        for tile_name, goal_color in self.goal_colors.items():
            # Check if the tile is painted with the correct color
            is_painted_correctly = f"(painted {tile_name} {goal_color})" in state

            if not is_painted_correctly:
                # We assume goal tiles are initially clear or unpainted.
                # If it's not painted correctly, it needs to be painted.
                unpainted_goals.append((tile_name, goal_color))
                needed_colors.add(goal_color)

        # If all goal tiles are painted correctly, the goal is reached
        if not unpainted_goals:
            return 0

        h = 0

        # 1. Painting Cost: Each unpainted goal tile needs one paint action.
        h += len(unpainted_goals)

        # 2. Color Change Cost: Estimate based on distinct colors needed.
        # If the robot already has one of the needed colors, it saves one change.
        # Cost is |needed_colors| if robot doesn't have a needed color,
        # |needed_colors| - 1 if robot has one of the needed colors.
        if robot_color in needed_colors:
            h += len(needed_colors) - 1
        else:
            h += len(needed_colors)

        # 3. Movement Cost: Estimate cost to reach the first unpainted tile.
        # Find the closest unpainted goal tile to the robot's current location.
        min_dist_to_any_unpainted_adj = float('inf')

        # Ensure robot location is valid and parsed
        if robot_loc and robot_loc in self.tile_coords:
             for tile_name, color in unpainted_goals:
                 dist = self.min_dist_to_adjacent(robot_loc, tile_name)
                 min_dist_to_any_unpainted_adj = min(min_dist_to_any_unpainted_adj, dist)

        # Add the minimum distance found, if any unpainted goals exist and robot location is valid
        if min_dist_to_any_unpainted_adj != float('inf'):
             h += min_dist_to_any_unpainted_adj
        # else: This case implies unpainted_goals was empty (handled above) or robot_loc is invalid.
        # If robot_loc is invalid but unpainted_goals exist, the problem might be malformed,
        # but we return the sum of paint and color costs calculated so far.

        return h
