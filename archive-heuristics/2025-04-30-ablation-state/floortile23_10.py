# Add necessary imports
from fnmatch import fnmatch
# Assuming heuristic_base.py is in a 'heuristics' directory relative to where this file is placed
# If the structure is different, the import path needs adjustment.
# Based on the provided example code structure, this import path seems plausible.
from heuristics.heuristic_base import Heuristic
from collections import deque # Use deque for efficient queue operations

# Helper functions from examples
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
    # Basic check for length compatibility, allowing wildcards
    if len(args) > len(parts) and '*' not in args:
         return False
    if len(parts) < len(args) and '*' not in args:
         return False # Cannot match if fact has fewer parts than pattern args (unless pattern uses wildcards carefully)

    # Use zip to handle potential length differences if wildcards are complex
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet painted correctly. It sums the minimum estimated cost for each
    unpainted goal tile independently, considering the closest robot with the
    required color (or the cost to change color). It assumes a grid structure
    defined by the static adjacency predicates.

    # Assumptions
    - Tiles form a grid structure defined by static up/down/left/right predicates.
    - Tile names like 'tile_row_col' do not necessarily encode coordinates directly;
      coordinates are derived from the adjacency predicates.
    - A tile painted with the wrong color (different from the goal color) is a dead end (unsolvable state).
    - Robots can move freely between clear adjacent tiles.
    - The cost of moving between adjacent tiles is 1.
    - The cost of changing color is 1.
    - The cost of painting a tile is 1.
    - The heuristic ignores potential conflicts (robots blocking each other,
      multiple robots needing the same color/tile simultaneously) and assumes
      each unpainted goal tile can be painted by the 'best' robot for that tile
      independently of other tiles and robots.

    # Heuristic Initialization
    - Extracts the goal conditions (`(painted tile color)` facts).
    - Builds a coordinate map (`self.tile_coords`) for all tiles by traversing the grid defined by
      static up/down/left/right predicates. It starts from an arbitrary tile, assigns it (0,0),
      and assigns coordinates to neighbors recursively based on the direction predicates.
      This map is used to calculate Manhattan distances between tiles.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Check for Dead Ends:** Iterate through all goal painting requirements (`(painted tile_G color_G)`). For each goal tile `tile_G`, check if it is currently painted with a *different* color (`(painted tile_G color_C)` where `color_C != color_G`). If such a fact exists in the current state, the state is unsolvable, and the heuristic returns `float('inf')`.

    2.  **Identify Current State Information:**
        - Determine the current location (`robot-at`) for each robot. Store this in a dictionary `robot_locations`.
        - Determine the current held color (`robot-has`) for each robot. Store this in a dictionary `robot_colors`.
        - Identify the set of all robots.

    3.  **Identify Unpainted Goal Tiles:** Iterate through the goal painting requirements (`self.goal_paintings`). For each required painting `(tile_G, color_G)`:
        - Check if the fact `(painted tile_G color_G)` is present in the current state.
        - If it is *not* present, and the tile is `clear` (which it must be if not painted correctly, based on the dead end check), add `(tile_G, color_G)` to a list of `needed_paintings`.

    4.  **Handle Trivial Cases:**
        - If the list of `needed_paintings` is empty, all goal tiles are painted correctly, so the state is a goal state. Return 0.
        - If the list of `needed_paintings` is not empty but there are no robots, the state is unsolvable. Return `float('inf')`.

    5.  **Calculate Total Heuristic Cost:** Initialize `total_cost = 0`.

    6.  **Calculate Cost Per Unpainted Tile:** For each `(tile_G, color_G)` in the `needed_paintings` list:
        - Find the coordinates `coords_G` for `tile_G` using the map built during initialization. If the tile is not in the map (disconnected grid), return `float('inf')`.
        - Initialize `min_cost_for_tile = float('inf')`. This will store the minimum cost for *any* robot to paint this specific tile.
        - For each robot `R`:
            - Get its current location `tile_R` and held color `color_H`. Find its coordinates `coords_R`. If the robot's location is not in the map, return `float('inf')`.
            - Calculate the cost for this robot `R` to acquire the correct color `color_G`: This is 1 if `color_H` is different from `color_G`, and 0 otherwise. (This is a simplification; it assumes the robot can instantly change color, ignoring the `available-color` precondition, but `available-color` is static and usually all colors are available).
            - Calculate the minimum number of moves required for robot `R` to reach *any* tile adjacent to `tile_G`. Let `d` be the Manhattan distance between `tile_R` and `tile_G`. The moves needed are: 1 if `d == 0` (robot is at `tile_G`), 0 if `d == 1` (robot is adjacent), and `d - 1` if `d > 1`. This can be expressed as `1` if `d == 0` else `max(0, d - 1)`.
            - The estimated cost for robot `R` to paint `tile_G` is the sum of the color change cost, the movement cost, and the paint action cost (which is 1).
            - Update `min_cost_for_tile` with the minimum of its current value and the calculated cost for robot `R`.
        - If `min_cost_for_tile` is still `float('inf')` after checking all robots (e.g., no robot can reach the tile), the state is unsolvable. Return `float('inf')`.
        - Add `min_cost_for_tile` to the `total_cost`.

    7.  **Return Heuristic Value:** Return the final `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the tile coordinate map
        and storing goal conditions.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the tile coordinate map from static facts
        self.tile_coords = {}
        q = deque() # Use deque for efficient queue operations
        visited = set()

        # Build adjacency list based on static facts
        # Map: tile -> [(neighbor_tile, direction_from_tile_to_neighbor)]
        adj_list = {}
        all_tiles = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                direction, tile1, tile2 = parts # (direction tile1 tile2) means tile1 is [direction] of tile2
                all_tiles.add(tile1)
                all_tiles.add(tile2)

                # The predicate (up y x) means y is UP from x.
                # So if fact is (up tile_y tile_x), tile_y is above tile_x.
                # From tile_x, the neighbor tile_y is in the 'up' direction.
                # From tile_y, the neighbor tile_x is in the 'down' direction.

                direction_from_x_to_y = parts[0] # e.g., 'up'
                direction_from_y_to_x = None
                if direction_from_x_to_y == 'up': direction_from_y_to_x = 'down'
                elif direction_from_x_to_y == 'down': direction_from_y_to_x = 'up'
                elif direction_from_x_to_y == 'left': direction_from_y_to_x = 'right'
                elif direction_from_x_to_y == 'right': direction_from_y_to_x = 'left'

                # Add edge from tile2 to tile1 with direction_from_x_to_y
                adj_list.setdefault(tile2, []).append((tile1, direction_from_x_to_y))
                # Add edge from tile1 to tile2 with direction_from_y_to_x
                if direction_from_y_to_x:
                    adj_list.setdefault(tile1, []).append((tile2, direction_from_y_to_x))


        # Find a starting tile (any tile in the graph)
        start_tile = next(iter(all_tiles), None) # Get the first tile, or None if all_tiles is empty

        if start_tile:
            q.append((start_tile, (0, 0)))
            visited.add(start_tile)

            while q:
                current_tile, (r, c) = q.popleft() # Use popleft for deque
                self.tile_coords[current_tile] = (r, c)

                # Explore neighbors using the built adjacency list
                for neighbor_tile, direction_to_neighbor in adj_list.get(current_tile, []):
                    if neighbor_tile not in visited:
                        new_coords = None
                        if direction_to_neighbor == 'up':
                            new_coords = (r - 1, c)
                        elif direction_to_neighbor == 'down':
                            new_coords = (r + 1, c)
                        elif direction_to_neighbor == 'left':
                            new_coords = (r, c - 1)
                        elif direction_to_neighbor == 'right':
                            new_coords = (r, c + 1)

                        if new_coords is not None:
                            visited.add(neighbor_tile)
                            q.append((neighbor_tile, new_coords))

        # Store goal tiles and their required colors
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

    def manhattan_distance(self, coords1, coords2):
        """Calculate Manhattan distance between two coordinate tuples (r, c)."""
        # The check for None coords is handled before calling this function
        return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check for dead ends: goal tile painted with the wrong color
        current_painted_facts = {fact for fact in state if match(fact, 'painted', '*', '*')}
        for tile_G, color_G in self.goal_paintings.items():
            for fact_str in current_painted_facts:
                 parts = get_parts(fact_str)
                 if parts[1] == tile_G and parts[2] != color_G:
                     # Tile is painted with the wrong color - unsolvable state
                     return float('inf')

        # Identify robot locations and colors
        robot_locations = {}
        robot_colors = {}
        robots = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                robots.add(robot)
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                robots.add(robot)

        # Identify unpainted goal tiles
        needed_paintings = [] # List of (tile, color) tuples
        # current_clear_facts = {fact for fact in state if match(fact, 'clear', '*')} # Not strictly needed due to dead end check

        for tile_G, color_G in self.goal_paintings.items():
            # Check if the tile is already painted correctly
            goal_fact_str = f'(painted {tile_G} {color_G})'
            if goal_fact_str in current_painted_facts:
                continue # Already painted correctly

            # If not painted correctly, it must be clear (checked by dead end logic)
            # Add to needed list
            needed_paintings.append((tile_G, color_G))


        # If no tiles need painting, the goal is reached
        if not needed_paintings:
            return 0

        # If there are no robots but tiles need painting, it's unsolvable
        if not robots:
             return float('inf')

        total_cost = 0

        # Calculate cost for each unpainted goal tile independently
        for tile_G, color_G in needed_paintings:
            coords_G = self.tile_coords.get(tile_G)
            if coords_G is None:
                 # This goal tile is not part of the grid defined by static facts. Unsolvable.
                 return float('inf')

            min_cost_for_tile = float('inf')

            # Find the minimum cost for any robot to paint this tile
            for robot in robots:
                tile_R = robot_locations.get(robot)
                color_H = robot_colors.get(robot)

                if tile_R is None or color_H is None:
                    # Robot state is incomplete? Should not happen in valid PDDL state.
                    # Or robot object exists but robot-at/robot-has facts are missing?
                    # Treat as unable to paint this tile.
                    continue

                coords_R = self.tile_coords.get(tile_R)
                if coords_R is None:
                    # Robot is at a tile not in the grid? Unlikely in valid problems.
                    # Treat as unable to paint this tile.
                    continue

                # Cost to change color if needed
                color_cost = 1 if color_H != color_G else 0

                # Moves needed to reach a tile adjacent to tile_G
                d = self.manhattan_distance(coords_R, coords_G)
                # Moves = 1 if robot is at tile_G, 0 if adjacent, d-1 if further
                moves_to_neighbor = 1 if d == 0 else max(0, d - 1)

                # Total cost for this robot to paint this specific tile
                robot_cost_for_tile = color_cost + moves_to_neighbor + 1 # +1 for the paint action

                min_cost_for_tile = min(min_cost_for_tile, robot_cost_for_tile)

            # If min_cost_for_tile is still inf, it means no robot could reach it
            # (e.g., grid is disconnected, or no robots exist - the latter is checked).
            # This implies the state is unsolvable.
            if min_cost_for_tile == float('inf'):
                 return float('inf')

            # Add the minimum cost for this tile to the total heuristic
            total_cost += min_cost_for_tile

        return total_cost
