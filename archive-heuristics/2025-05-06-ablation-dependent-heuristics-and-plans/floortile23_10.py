from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise an error
        # For robustness, return empty list or handle appropriately
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(tile_name):
    """Parses tile name 'tile_R_C' into integer coordinates (R, C)."""
    try:
        # Split by '_' and take parts after 'tile'
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
             return int(parts[1]), int(parts[2])
        else:
             # Handle unexpected tile name format
             return None
    except ValueError:
        # Handle cases where R or C are not integers
        return None

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It considers the cost of painting, moving robots off
    occupied goal tiles, changing robot colors, and robot movement to reach tiles.

    # Assumptions
    - Tiles are arranged in a grid, and movement cost can be estimated using
      Manhattan distance between tile coordinates derived from their names (tile_R_C).
    - Robots can move freely between adjacent clear tiles (Manhattan distance is a
      reasonable estimate ignoring obstacles).
    - A tile painted with the wrong color cannot be repainted and represents a dead end.
    - Robots always possess a color (no 'free-color' state for painting/changing).
    - The cost of changing color is 1, and a robot can change to any available color.
    - The heuristic does not account for multiple robots coordinating on the same tile
      or path conflicts.

    # Heuristic Initialization
    - Extracts all tile names and their coordinates from the initial state, static
      facts, and goal state to determine the grid boundaries (min/max row/col).
    - Stores the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles that are not currently painted with their required color.
       Let this set be `UnpaintedGoals`.
    2. Check for dead ends: If any tile in `UnpaintedGoals` is currently painted
       with a *different* color than required by the goal, the state is likely
       unsolvable. Return infinity.
    3. Calculate the base cost: Add 1 for each tile in `UnpaintedGoals`. This
       represents the paint action needed for each such tile.
    4. Calculate the occupation cost: For each tile in `UnpaintedGoals`, check if
       any robot is currently located on that tile. If yes, add 1 to the heuristic.
       This represents the cost of moving the robot off the tile to make it clear
       for painting.
    5. Calculate the color acquisition cost: Determine the set of colors required
       by the tiles in `UnpaintedGoals`. Determine the set of colors currently
       held by the robots. Count how many required colors are *not* currently
       held by any robot. Add this count to the heuristic. This estimates the
       minimum number of color changes needed across all robots.
    6. Calculate the movement cost: For each tile in `UnpaintedGoals`:
       a. Get the coordinates (row, col) of the tile.
       b. Determine the coordinates of all valid adjacent tiles within the grid bounds
          that actually exist in the problem.
       c. For each robot, calculate the Manhattan distance from its current location
          to each of the adjacent tiles found in step 6b.
       d. Find the minimum distance among all robots and all adjacent tiles for
          the current unpainted goal tile.
       e. Add this minimum distance to the heuristic.
    7. The total heuristic value is the sum of the base cost, occupation cost,
       color acquisition cost, and total movement cost (summed over all
       unpainted goal tiles).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and grid information."""
        self.goals = task.goals  # Goal conditions (frozenset of fact strings)

        # Extract all tile names and determine grid bounds
        all_tile_names = set()
        # Collect tile names from initial state, static facts, and goals
        for fact in task.initial_state | task.static | task.goals:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('tile_'):
                     all_tile_names.add(part)

        self.all_tile_coords = {get_coords(name) for name in all_tile_names if get_coords(name) is not None}

        if not self.all_tile_coords:
             # Handle case with no tiles (shouldn't happen in valid problems)
             # Set bounds to an invalid range so get_adjacent_coords returns empty
             self.min_row, self.max_row, self.min_col, self.max_col = 0, -1, 0, -1
        else:
            self.min_row = min(r for r, c in self.all_tile_coords)
            self.max_row = max(r for r, c in self.all_tile_coords)
            self.min_col = min(c for r, c in self.all_tile_coords)
            self.max_col = max(c for r, c in self.all_tile_coords)

        # Store goal painted facts for quick lookup {tile_name: color}
        self.goal_painted = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

    def get_adjacent_coords(self, r, c):
        """Returns a list of valid adjacent tile coordinates within grid bounds that exist."""
        adjacent = []
        possible_adj = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for ar, ac in possible_adj:
            # Check if the adjacent coordinates are within the determined grid bounds
            # AND if a tile with these coordinates actually exists in the problem
            if (self.min_row <= ar <= self.max_row and
                self.min_col <= ac <= self.max_col and
                (ar, ac) in self.all_tile_coords):
                 adjacent.append((ar, ac))
        return adjacent

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Extract relevant information from the current state
        current_painted = {} # Stores {tile_name: color} for painted tiles in state
        robot_locations = {} # Stores {robot_name: tile_name}
        robot_colors = {} # Stores {robot_name: color}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                current_painted[tile] = color
            elif predicate == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif predicate == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # 1. Identify unpainted goal tiles and check for wrong colors
        unpainted_goals = set() # Stores (tile_name, color)
        for goal_tile, goal_color in self.goal_painted.items():
            if goal_tile not in current_painted or current_painted[goal_tile] != goal_color:
                # Tile is not painted correctly
                if goal_tile in current_painted and current_painted[goal_tile] != goal_color:
                    # 2. Dead end: painted with wrong color
                    return float('inf')
                else:
                    # Not painted correctly (either clear or not mentioned as painted)
                    unpainted_goals.add((goal_tile, goal_color))

        # If all goals are painted correctly, heuristic is 0
        if not unpainted_goals:
            return 0

        h = 0

        # 3. Base cost (paint action for each unpainted goal tile)
        h += len(unpainted_goals)

        # 4. Occupation cost
        occupied_tiles = set(robot_locations.values())
        for tile_name, _ in unpainted_goals:
            if tile_name in occupied_tiles:
                h += 1 # Cost to move robot off

        # 5. Color acquisition cost
        needed_colors = {color for tile_name, color in unpainted_goals}
        current_robot_colors = set(robot_colors.values())
        h += len(needed_colors - current_robot_colors)

        # 6. Movement cost
        # If there are no robots but unpainted goals, the problem is unsolvable.
        # The movement cost calculation will handle this by min_dist_to_adj remaining inf.
        # We will return inf later if any tile requires an infinite movement cost.
        if not robot_locations and unpainted_goals:
             # If there are unpainted goals but no robots, it's unsolvable.
             return float('inf')

        for tile_name, _ in unpainted_goals:
            tile_coords = get_coords(tile_name)
            if tile_coords is None:
                 # Should not happen if tile names are consistent, but handle defensively
                 return float('inf') # Cannot calculate distance for this tile

            adj_coords = self.get_adjacent_coords(*tile_coords)
            if not adj_coords:
                 # If a tile needs painting but has no adjacent tiles defined in the grid,
                 # it's likely unsolvable.
                 return float('inf')

            min_dist_to_adj = float('inf')

            for robot_name, robot_tile_name in robot_locations.items():
                robot_coords = get_coords(robot_tile_name)
                if robot_coords is None:
                     # Should not happen if robot locations are consistent tile names
                     continue # Skip this robot if location parsing fails

                for ar, ac in adj_coords:
                    dist = abs(robot_coords[0] - ar) + abs(robot_coords[1] - ac)
                    min_dist_to_adj = min(min_dist_to_adj, dist)

            # If min_dist_to_adj is still inf after checking all robots and adjacent tiles,
            # it means no robot could reach any adjacent tile (e.g., no robots, or grid issue).
            # This implies unsolvability for this tile.
            if min_dist_to_adj == float('inf'):
                 return float('inf')

            # Add the minimum movement cost for this tile
            h += min_dist_to_adj

        return h
