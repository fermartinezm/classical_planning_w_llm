from collections import deque
# from fnmatch import fnmatch # Not used
from heuristics.heuristic_base import Heuristic
import sys # Needed for float('inf') alternative if preferred, but float('inf') is standard

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and ensure it's a string
    if not isinstance(fact, str):
        return [] # Or raise error, depending on expected input
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        return [] # Not a valid PDDL fact string
    return fact[1:-1].split()

# No need for a generic match function if we parse explicitly

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It considers the number of tiles yet to be painted,
    the number of distinct colors needed across all unpainted goal tiles, and the
    minimum distance from any robot to a valid painting location for each unpainted tile.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - Tile names can be mapped to relative (row, column) coordinates based on
      the grid structure defined by static facts.
    - All goal tiles are assumed to be paintable from at least one adjacent tile
      as defined by the static facts.
    - Multiple robots can work in parallel. The heuristic sums costs per tile,
      considering the closest robot for movement, and counts colors needed globally.
    - The heuristic ignores the 'clear' predicate for movement cost calculation,
      assuming movement is always possible along grid lines. It implicitly assumes
      unpainted goal tiles are clear or will become clear.

    # Heuristic Initialization
    - Parses goal facts to identify target tiles and their required colors, storing
      them in `self.goal_tiles`.
    - Parses static facts ('up', 'down', 'left', 'right') to build a grid
      representation. This involves:
        - Creating an adjacency list (`adj_list`) representing tile connections.
        - Identifying valid 'paint locations' (`self.paint_locations`), mapping
          a tile to be painted to the set of tiles a robot must be at to paint it.
          For a fact `(direction tile_A tile_B)`, `tile_B` is a paint location for `tile_A`.
        - Building a coordinate map (`self.tile_coords`) for all tiles reachable
          from a starting tile using Breadth-First Search (BFS), assigning
          relative (row, column) coordinates.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unpainted Goal Tiles: Iterate through `self.goal_tiles` and check
       if the corresponding `(painted T C)` fact exists in the current state.
       Collect all `(tile, color)` pairs that are goals but not in the state.
    2. Check for Goal State: If the set of unpainted goal tiles is empty, the
       current state is a goal state, and the heuristic value is 0.
    3. Find Robot Information: Extract the current location and color for each
       robot from the state facts.
    4. Calculate Paint Cost: The minimum number of paint actions required is
       equal to the number of unpainted goal tiles. Add this count to the total heuristic.
    5. Calculate Color Change Cost: Determine the set of distinct colors required
       for all unpainted goal tiles (`colors_needed`). Determine the set of colors
       currently held by all robots (`robots_colors`). The minimum number of
       `change_color` actions needed across all robots is estimated as the number
       of colors in `colors_needed` that are *not* present in `robots_colors`.
       Add this count to the total heuristic.
    6. Calculate Movement Cost: For each unpainted goal tile `T`:
       a. Find the set of tiles (`PaintLocs_T`) from which `T` can be painted
          (precomputed in initialization).
       b. For each robot, calculate the Manhattan distance from its current
          location to every tile in `PaintLocs_T`.
       c. Find the minimum distance among all robot-to-paint-location distances
          for tile `T`.
       d. Sum these minimum distances over all unpainted goal tiles. Add this sum
          to the total heuristic. This component estimates the travel effort.
    7. Return the sum of Paint Cost, Color Change Cost, and Movement Cost.
       If the grid could not be initialized, or if any required paint location
       or robot location is not found in the precomputed grid map (implying
       unreachability or a problematic instance/state), return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the grid structure from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goal facts
        self.goal_tiles = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted' and len(parts) == 3:
                 self.goal_tiles.add((parts[1], parts[2]))

        # 2. Parse static facts for grid and paint locations
        self.tile_coords = {}  # tile_name -> (r, c)
        self.paint_locations = {}  # tile_to_be_painted -> set(tiles_robot_must_be_at)
        adj_list = {}  # tile -> list of (neighbor, direction)
        all_tiles_in_connections = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                # (direction tile_A tile_B) means tile_A is in that direction FROM tile_B
                # Robot at tile_B can paint tile_A using paint_direction
                tile_A, tile_B = parts[1], parts[2]
                all_tiles_in_connections.add(tile_A)
                all_tiles_in_connections.add(tile_B)

                # Build adjacency list for BFS (for coordinate mapping)
                # B is in reverse direction from A
                adj_list.setdefault(tile_A, []).append((tile_B, {'up':'down', 'down':'up', 'left':'right', 'right':'left'}[predicate]))
                # A is in direction from B
                adj_list.setdefault(tile_B, []).append((tile_A, predicate))

                # B is a valid paint location for A
                self.paint_locations.setdefault(tile_A, set()).add(tile_B)

        # Build tile_coords using BFS
        if not all_tiles_in_connections:
             # Cannot build grid if no tiles or connections are defined
             self.tile_coords = {}
             self.paint_locations = {}
             print("Warning: No tiles or grid connections found in static facts. Grid heuristic disabled.")
             return

        # Find a starting tile that is present in the adjacency list keys
        start_tile = None
        for tile in all_tiles_in_connections:
            if tile in adj_list:
                start_tile = tile
                break

        if not start_tile:
             # No tile is a source or destination in any connection, implies isolated tiles or empty grid
             self.tile_coords = {}
             self.paint_locations = {}
             print("Warning: Could not find a starting tile for BFS in static facts. Grid heuristic disabled.")
             return


        queue = deque([(start_tile, 0, 0)])
        visited = {start_tile}
        self.tile_coords[start_tile] = (0, 0)

        while queue:
            current_tile, r, c = queue.popleft()
            if current_tile in adj_list:
                for neighbor_tile, direction in adj_list[current_tile]:
                    if neighbor_tile not in visited:
                        visited.add(neighbor_tile)
                        nr, nc = r, c
                        if direction == 'up': nr -= 1
                        elif direction == 'down': nr += 1
                        elif direction == 'left': nc -= 1
                        elif direction == 'right': nc += 1
                        self.tile_coords[neighbor_tile] = (nr, nc)
                        queue.append((neighbor_tile, nr, nc))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings

        # Find robot locations and colors
        robot_info = {} # robot_name -> {'location': tile, 'color': color}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_info.setdefault(robot, {})['location'] = tile
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_info.setdefault(robot, {})['color'] = color

        # Identify unpainted goal tiles
        unpainted_goals = set()
        for tile, color in self.goal_tiles:
            if f'(painted {tile} {color})' not in state:
                unpainted_goals.add((tile, color))

        # If all goals are met, heuristic is 0
        if not unpainted_goals:
            return 0

        # Handle case where grid initialization failed
        if not self.tile_coords or not self.paint_locations:
             # Cannot compute grid-based heuristic
             return float('inf')

        # --- Calculate Heuristic Components ---

        # 1. Paint Cost: One action per unpainted tile
        paint_cost = len(unpainted_goals)

        # 2. Color Change Cost: Minimum changes needed across all robots
        colors_needed = {color for tile, color in unpainted_goals}
        robots_colors = {info.get('color') for info in robot_info.values() if 'color' in info} # Get colors robots currently have
        colors_held_needed = colors_needed.intersection(robots_colors)
        # Minimum changes = number of needed colors not currently held by any robot
        color_cost = max(0, len(colors_needed) - len(colors_held_needed))


        # 3. Movement Cost: Sum of minimum distances for each tile to be painted
        movement_cost = 0
        robot_locations = [info.get('location') for info in robot_info.values() if 'location' in info]

        # If no robots, cannot paint
        if not robot_locations:
             return float('inf')

        for tile_to_paint, required_color in unpainted_goals:
            paint_locs = self.paint_locations.get(tile_to_paint, set())

            if not paint_locs:
                 # This tile cannot be painted based on static facts? Problematic instance.
                 return float('inf')

            min_dist_to_paint_loc = float('inf')

            for robot_loc in robot_locations:
                if robot_loc not in self.tile_coords:
                    # Robot is at a location not in our grid? Problematic state.
                    return float('inf')

                robot_coords = self.tile_coords[robot_loc]

                for p_loc in paint_locs:
                    if p_loc not in self.tile_coords:
                         # Paint location not in our grid? Problematic static facts.
                         return float('inf')

                    p_coords = self.tile_coords[p_loc]
                    dist = abs(robot_coords[0] - p_coords[0]) + abs(robot_coords[1] - p_coords[1])
                    min_dist_to_paint_loc = min(min_dist_to_paint_loc, dist)

            # Add the minimum distance for this tile to the total movement cost
            # If min_dist_to_paint_loc is still inf, it means no robot can reach any paint loc for this tile.
            if min_dist_to_paint_loc != float('inf'):
                 movement_cost += min_dist_to_paint_loc
            else:
                 # Cannot reach a paint location for this tile from any robot.
                 return float('inf')


        # Total heuristic is the sum of components
        total_cost = paint_cost + color_cost + movement_cost

        return total_cost
